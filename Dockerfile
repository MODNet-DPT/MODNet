FROM nvidia/cuda:11.3.0-runtime-ubuntu18.04

RUN apt update && \
    apt install -y unzip && \
    apt install -y python3-pip && \
    apt install -y python3-venv

RUN apt install -y ffmpeg libsm6 libxext6

RUN mkdir -p /opt/modnet
WORKDIR /opt/modnet

RUN mkdir data
RUN mkdir data/dataset
RUN pip3 install gdown
RUN gdown https://drive.google.com/uc?id=13V2LoSz9PvO8M6rrm3QwoDGtGx02Zc_Z
RUN unzip -q MODNet_CocoDataset.zip -d data/dataset/

COPY ./src ./src
COPY ./dockerRequirements.txt ./dockerRequirements.txt

RUN pip3 install --user virtualenv
RUN python3 -m venv /opt/modnet/env
ENV PATH /opt/modnet/env/bin:$PATH

RUN python3 -m pip install --upgrade pip

RUN pip3 install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install -r dockerRequirements.txt

RUN mkdir data/models
CMD python3 -m src.coco.trainOnCoco --dataset-path data/dataset/ --models-path data/models/
