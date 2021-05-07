# Usefull commands

## Inference on traind model

```
python3 -m src.coco.inference --input-path data/images/image1.jpg  --ckpt-path data/modnet.ckpt 
```

## Fetch dataset to Colab

```
!wget http://repo.yandex.ru/yandex-disk/yandex-disk_latest_amd64.deb
!sudo dpkg -i yandex-disk_latest_amd64.deb
!yandex-disk setup
!yandex-disk sync
!unzip ~/Yandex.Disk/data/MODNet_CocoDataset.zip -d /content/dataset
```

