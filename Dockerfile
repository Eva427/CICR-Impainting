# syntax=docker/dockerfile:1

FROM python:3.10-slim-buster

WORKDIR /cicr-docker

COPY . .

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["hypercorn", "--bind", "localhost:5000", "QuartApp.py"]