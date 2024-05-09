# YOLOmultitask
This tutorial give instruction how to run and setup Yolomultitask

## Installed package
```
pip install -r yoloxyz/multitasks/requirements/requirements.txt -q
```

## Data
Prepare the data follow by structure ([Download](https://drive.google.com/drive/folders/1vjAJUxpThYOlOp4bSLDg6rFZyYd4ZsHc?usp=sharing) data to test model)
```
|__ setup.py
|__ README.md
|__ yoloxyz
|__ datahub
   |__widerface
      |__images
      |   |__train
      |   |__val
      |__labels
         |__train
         |__val
```

## Download pretrain weights
- Prepare the checkpoint file in path (Download checkpoints [here](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt))
```
|__ setup.py
|__ README.md
|__ yoloxyz
|__ weights
   |__ yolov7-tiny.pt 
```

## Training

1. Single GPU
```
python yoloxyz/train.py \
  --epochs 300 \
  --workers 8 \
  --device 0 \
  --batch-size 64 \
  --data yoloxyz/multitasks/cfg/data/widerface.yaml \
  --img 640 640 \
  --cfg yoloxyz/multitasks/cfg/training/yolov7-tiny-multitask.yaml \
  --name yolov7-tiny-pretrain \
  --hyp yoloxyz/multitasks/cfg/hyp/hyp.yolov7.tiny.yaml \
  --weight weights/yolov7-tiny.pt \
  --sync-bn \
  --kpt-label 5 \
  --multilosses True \
  --detect-layer 'IKeypoint'
```

2. Multi GPUs
```
python -m torch.distributed.launch --nproc_per_node 2 --master_port 9527 yoloxyz/train.py \
  --epochs 300 \
  --workers 8 \
  --device 0,1 \
  --batch-size 64 \
  --data yoloxyz/multitasks/cfg/data/widerface.yaml \
  --img 640 640 \
  --cfg yoloxyz/multitasks/cfg/training/yolov7-tiny-multitask.yaml \
  --name yolov7-tiny-pretrain \
  --hyp yoloxyz/multitasks/cfg/hyp/hyp.yolov7.tiny.yaml \
  --weight weights/yolov7-tiny.pt \
  --sync-bn \
  --kpt-label 5 \
  --multilosses True \
  --detect-layer 'IKeypoint'
```