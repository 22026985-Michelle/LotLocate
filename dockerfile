
FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04


WORKDIR /app


ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Singapore

COPY . /app
COPY requirements.txt /app/

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libgtk2.0-dev \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libdc1394-22-dev \
    libv4l-dev \
    python3-dev \
    python3-numpy \
    python3-pip \
    wget \
    tzdata \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ca-certificates \
    && apt-get clean

RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python3", "main.py"]