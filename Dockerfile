# syntax=docker/dockerfile:1

FROM python:3.10-slim-buster

WORKDIR /cicr-docker

COPY . /app/

RUN pip install -r /app/requirements.txt

CMD ["hypercorn", "--bind", "0.0.0.0:5000", "myservice:app"]