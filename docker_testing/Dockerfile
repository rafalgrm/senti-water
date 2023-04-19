FROM python:3.10.10

WORKDIR /pythonapi

COPY listener.py ./

EXPOSE 6000

RUN pip install Flask Flask-RESTful python-dotenv

CMD [ "python", "./listener.py" ]