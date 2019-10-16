FROM python:3.6-jessie

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    yasm \
    pkg-config \
    libswscale-dev \
    libtbb2 \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libjasper-dev \
    libxvidcore-dev \
    libavformat-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    libpq-dev \
    bgtk2.0-dev \
    && rm -rf /var/lib/apt/lists

WORKDIR /

RUN pip install --upgrade pip

ADD $PWD/requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

CMD ["/bin/bash"]