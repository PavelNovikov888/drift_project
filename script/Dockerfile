FROM python:latest
WORKDIR /usr/src/app
COPY ./src/ .
COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
CMD [ "python", "./script.py" ]