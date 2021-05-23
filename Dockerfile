FROM nvidia/cuda:11.3.0-runtime-ubuntu18.04
WORKDIR /modnet
COPY . .

RUN pip3 install -r requirements.txt

RUN mkdir data
RUN mkdir data/dataset
RUN pip3 install gdown
# RUN wget https://drive.google.com/u/0/uc?export=download&confirm=FGWW&id=13V2LoSz9PvO8M6rrm3QwoDGtGx02Zc_Z
RUN gdown https://drive.google.com/uc?id=13V2LoSz9PvO8M6rrm3QwoDGtGx02Zc_Z
RUN unzip -q MODNet_CocoDataset.zip -d data/dataset/

RUN mkdir data/models
CMD python3 -m src.coco.trainOnCoco --dataset-path /data/dataset/ --models-path data/models/
