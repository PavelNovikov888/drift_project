# Мониторинги и управление качеством модели  

## Задача:  
1.Построить модель бинарной классификации c возможностью дообучения на новых данных.  
2.Настроить скраппинг(получение веб-данных путём извлечения их со страниц веб-ресурсов) метрик модели с помощью Prometheus.   
3.Подключить Prometheus к Grafana.  
4.Создать дашборд в Grafana.  

## Данные: 
Качество вина в зависимости от его физических свойств.  
Датасет содержит информацию о красных винах и их составе.  
Целевой переменной является столбец "quality".   
Это метрика качества вина по шкале от 3 до 8.  
В процессе feature engeneering целевой признак преобразуется в бинарный признак:хорошее вино(оценка > 6.5) и плохое вино(<6.5).  

## Модель  
Для решения данной задачи используем классификатор LightGBM на основе ансамблевого метода деревьев принятия решений с градиентным бустингом.  

## Реализация возможности обучения на новых данных.  
Данная опция необходима для решения проблемы дрифта данных и, как следствие, устаревания существующей модели классификации.  

При первом запуске метрики рассчитываются на исходных данных winequality-red.zip, после выполнения сериализованная модель сохраняется в корневой директории.  
При наличии в папке /script/src файла new_data.csv и предобученной модели loan_model.pkl расчет метрик производится на новых данных.   

Для имитации процесса получения новых данных в коде реализована функция  streaming_reading() разделяющая данные на пакеты(батчи)   
и поочередно подающая их в модель машинного обучения.  

**DISCLAIMER:** В целях визуальной демонстрации процесса мониторинга данные в модель поступают в процессе бесконечного цикла, 
в реальных условиях, очевидно, данный процесс имеет границы.  

### Инструкция по запуску:
0. Запустить программу Docker  
1. Клонировать данный репозиторий на локальный компьютер  
2. Проверить порты localhost:8000, localhost:9090, localhost:3000 командой sudo netstat -tulpn | grep :<номер порта>   
3. Если данные порты заняты, остановить процессы или программы, которые их заняли.  
4. Перейти в папку проекта cd <путь на локальной машине к клону репозитория>+/drift_project/   
5. Собрать и запустить docker-compose файл командами: docker-compose build и docker-compose up -d    
6. Проверить страницы в браузере:  
- http://localhost:8000 На экране должны появиться метрики logloss_gaude, auc_gaude, count_batch  
 [Скриншот](https://github.com/PavelNovikov888/drift_project/blob/master/picture/port8000.png)  
- http://localhost:9090/targets В колонке "endpoint" напротив строки http://drift:8000/metrics должна быть зелёная пометка UP    
[Скриншот](https://github.com/PavelNovikov888/drift_project/blob/master/picture/port9090.png) 
7. Перейти на страницу http://localhost:3000 и авторизоваться в программе Grafana
- Ввести логин: admin
- Ввести пароль: 123456
8. http://localhost:3000/connections/datasources выбрать источник данных Prometheus.   
9. В настройках источника данных в диалоговом окне "Connection Prometheus server URL" ввести http://prom:9090  
[Скриншот](https://github.com/PavelNovikov888/drift_project/blob/master/picture/datasourse3000.png)   
Внизу окна настроек нажать Save&Test , убедится что появилась надпись на зеленом фоне Successfully queried the Prometheus API.  
[Скриншот](https://github.com/PavelNovikov888/drift_project/blob/master/picture/save_test3000.png)  
9. Перейти к настройкам дашборда http://localhost:3000/dashboards  
10. Нажать слева на синюю кнопку New, в выпадающем окне выбрать пункт Import  
11. В среднее поле ввести 20943 и нажать синюю кнопку Load   
Это ID дашборда в GrafanaLabs https://grafana.com/grafana/dashboards/20943   
Также можно напрямую загрузить дашборд. Он находится в репозитории ./drift_project/grafana/dashboard_logloss.json  
12. Выбрать появившийся дашборд Logloss & AUC в списке дашбордов   
13. На экране появится 3 поля с графиками метрик.   
[Скриншот](https://github.com/PavelNovikov888/drift_project/blob/master/picture/dashboard3000.png)  
14. После окончания работы с программой остановить контейнеры командой docker-compose down  






