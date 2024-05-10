FROM python:3.11.0-slim-buster

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

RUN pip install poetry

COPY pyproject.toml poetry.lock /app

RUN poetry config virtualenvs.create false && \
    poetry install --without=dev --no-interaction --no-ansi

COPY . /app
