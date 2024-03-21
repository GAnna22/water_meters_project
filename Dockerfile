# syntax=docker/dockerfile:1
FROM python:3.11-bookworm
SHELL ["/bin/bash", "-c"]

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt update && \
    apt upgrade -y && \
    apt install -y -qq --force-yes --no-install-recommends \
      curl ca-certificates tree vim \
      software-properties-common \
	ffmpeg libsm6 libxext6 && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /app
RUN mkdir data

RUN python3 -m pip install --upgrade pip
COPY ./requirements.txt .
RUN python3 -m pip install -r requirements.txt
COPY ./src/ .
COPY entrypoint.sh .

EXPOSE 8501

ENTRYPOINT ["./entrypoint.sh"]
