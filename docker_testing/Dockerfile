FROM python:3.10.10

WORKDIR /pythonapi

COPY ./src ./src
COPY requirements.txt ./

EXPOSE 6000

RUN pip install -r requirements.txt

CMD [ "python", "-u" ,"./src/listener.py" ]