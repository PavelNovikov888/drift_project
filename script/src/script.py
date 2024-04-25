# pip install prometheus-client -q
# pip install --upgrade dask -q
# pip install -U scikit-learn

import lightgbm as lgb
# from scikit import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, precision_score, recall_score
import joblib
import pandas as pd
import numpy as np

# отключение предупреждений
import warnings
warnings.filterwarnings('ignore')

# импортируем библиотеку для работы с кодировщиками
import category_encoders as ce

import prometheus_client
from prometheus_client import CollectorRegistry, Gauge, start_http_server
import time
    
ans = ''
try:
    while True:
        ans = input('Введите 1, если данные новые или 0, если данные исходные ')
        if int(ans) == 1:
            print('Расчет метрик производится на новых данных с использованием ранее обученной модели, данные будут взяты из файла new_data.csv')
            gbm = joblib.load('loan_model.pkl')
            dataset = pd.read_csv('new_data.csv', sep=';')
            break
        elif int(ans) == 0:
            print('Расчет метрик производится в первый раз, данные будут взяты из файла winequality-red.zip')
            dataset = pd.read_csv('winequality-red.zip', sep=';')
            break
except EOFError:
    print(EOFError)
    dataset = pd.read_csv('winequality-red.zip', sep=';')
    
# Преобразуем целевой признак качества вина
dataset.loc[:,'quality'] = dataset['quality'].apply(
    lambda x: 'bad wine' if x < 6.5 else 'good wine'
    )

# создаем объект OrdinalEncoder, col - имя столбца, mapping - словарь с описанием кодировки
ord_encoder = ce.OrdinalEncoder(mapping=[{
    'col': 'quality',
    'mapping': {'bad wine': 0, 'good wine': 1}
    }])
# Преобразуем категориальный признак в числовой
dataset['quality_n'] = ord_encoder.fit_transform(dataset[['quality']])
dataset.sample(3)

X = dataset.drop(['quality','quality_n'], axis = 1)
y = dataset['quality_n']

#### Реализация стримингового прочтения файлов
#Попробуем воссоздать потом реальных данных. Для этого разобьем данные на батчи и будем их по порядку считывать.

def streaming_reading(X_train, y_train, batch_size=100):
    X = []
    y = []
    current_line = 0
    train_data, train_label = shuffle(X_train, y_train, random_state=0)
    train_data = train_data.to_numpy()
    for row, target in zip(train_data, train_label):
        X.append(row)
        y.append(target)

        current_line += 1
        if current_line >= batch_size:
            X, y = np.array(X), np.array(y)
            yield X, y
            X, y = [], []
            current_line = 0
            
gauge_logloss = Gauge('logloss_gaude', 'Это метрика logloss')
gauge_auc = Gauge('auc_gaude', 'Это метрика AUC')
gauge_batch = Gauge('count_batch', 'Cчетчик пакетов')
gauge_batch.set(1)

train_X, test_X, train_y, test_y = train_test_split(X, y, stratify = y, test_size = 0.2)

def IncrementaLightGbm(X, y):  # Реализация lightgbm 
    gbm = None

    params = {
        'task': 'train',
        'application': 'binary',  
        'boosting_type': 'gbdt', 
        'learning_rate': 0.05,  
        'tree_learner': 'feature',
        'metric': ['binary_logloss', 'auc'], 
        'max_bin': 255,
        'force_col_wise': True,
        'is_unbalance': True
    }
    streaming_train_iterators = streaming_reading(X, y, batch_size=100)

    for i, data in enumerate(streaming_train_iterators):
        X_batch = data[0]
        y_batch = data[1]
        X_train, X_test, y_train, y_test = train_test_split(X_batch, y_batch, test_size=0.1, stratify =y_batch, random_state=0)
        y_train = y_train.ravel()
        lgb_train = lgb.Dataset(X_train, y_train, params={'verbose': -1}, free_raw_data=False)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train, params={'verbose': -1}, free_raw_data=False)

        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=100,
                        valid_sets=lgb_eval,
                        init_model=gbm,
                        keep_training_booster=True)  

        print("{} time".format(i))  
        score_train = dict([(score[1], score[2]) for score in gbm.eval_train()])
        print('The score of the current model in the training set is: logloss=%.4f, auc=%.4f, \n'
              % (score_train['binary_logloss'], score_train['auc']))
        logloss = score_train['binary_logloss']
        auc_sc = score_train['auc']
        
        gauge_logloss.set(logloss)
        gauge_auc.set(auc_sc)
        gauge_batch.inc()
        
        time.sleep(10)
        
    return gbm, gauge_logloss, gauge_auc, gauge_batch

if __name__ == '__main__':
    # Запуск сервера для сбора метрик Prometheus на порту 8000
    start_http_server(8000)
    # Бесконечный цикл с имитацией обработки запросов
    while True:
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2)
        gbm, gauge_logloss, gauge_auc, gauge_butch = IncrementaLightGbm(train_X, train_y)
        pred_y = gbm.predict(test_X)
        pred_classes = np.where(pred_y > 0.5, 1, 0)
        print(f'F1 score: {f1_score(test_y, pred_classes)}')
        print('------------------------------------------')
        print(f'Precision: {precision_score(test_y, pred_classes)}')
        print('------------------------------------------')
        print(f'Recall: {recall_score(test_y, pred_classes)}')

joblib.dump(gbm, './script/src/loan_model.pkl')
