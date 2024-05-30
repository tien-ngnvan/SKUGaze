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
  --iou-loss EIoU \
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
  --iou-loss EIoU \
  --multilosses True \
  --detect-layer 'IKeypoint'
```

3. [Fully Sharded Data Parallel (FSDP)](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- In DistributedDataParallel, (DDP) training, each process/ worker owns a replica of the model and processes a batch of data, finally it uses all-reduce to sum up gradients over different workers. In DDP the model weights and optimizer states are replicated across all workers. FSDP is a type of data parallelism that shards model parameters, optimizer states and gradients across DDP ranks.

- When training with FSDP, the GPU memory footprint is smaller than when training with DDP across all workers. This makes the training of some very large models feasible by allowing larger models or batch sizes to fit on device. This comes with the cost of increased communication volume. The communication overhead is reduced by internal optimizations like overlapping communication and computation.

- For this reason, Yoloxyz integration [FSDP](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel) optimize time and training cost which several features comparable [Deepspeed](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/) as

|   | FSDP | Deepspeed |
| ------------- | ------------- | ------------- |
|  FSDP's version of DDP |  no_shard | _  |
| Sharding both the optimizer state and gradients  | grad_op  | ZeRO2  |
| Sharding optimizer state, gradients, and model parameters  | full  | ZeRO3 |
| Offloads the optimizer memory and computation from the GPU->CPU | full + cpu_offload | ZeRO3 + offload |


- Easy running with several lines
```
python -m torch.distributed.launch --nproc_per_node 2 --master_port 9527 yoloxyz/train_accelerate.py \
  --epochs 100 \
  --workers 8 \
  --device 0,1 \
  --batch-size 64 \
  --data yoloxyz/multitasks/cfg/data/widerface.yaml \
  --img 640 640 \
  --cfg yoloxyz/multitasks/cfg/training/yolov7-tiny-multitask.yaml \
  --name yolov7-tiny-pretrain \
  --hyp yoloxyz/multitasks/cfg/hyp/hyp.yolov7.tiny.yaml \
  --weight weights/yolov7-tiny.pt \
  --kpt-label 5 \
  --iou-loss EIoU \
  --multilosses True \
  --detect-layer 'IKeypoint' \
  --warmup \
  --use_fsdp \
  --adam \
  --sharding 'no_shard'
```
(*) As any [Accelerator](https://github.com/huggingface/accelerate) , FSDP sensitive with `learning rate` easy get [NaN](https://github.com/huggingface/accelerate/issues/2402) loss in training when run with MixedPrecision (FP16, BF16,...), we recommend you should start with `1e-3` or `1e-4`.

## Pytorch Inference
```
python yoloxyz/detect.py \
    --weights weights/best.pt \
    --device 0 \
    --source 'samples' \
    --detect-layer face \
    --kpt-label 5
```

## ONNX 
### ONNX export
```
python yoloxyz/multitasks/onnx_sp/onnx_export.py \
  --weights weights/best.pt \
  --img-size 640 --batch-size 10 \
  --dynamic-batch --grid --end2end --max-wh 640 --topk-all 100 \
  --iou-thres 0.5 --conf-thres 0.2 --device 'cpu' --simplify --cleanup

```
### ONNX Inference
```
python yoloxyz/multitasks/onnx_sp/onnx_inference.py \
  --model-path 'weights/best.onnx' \
  --img-path 'samples' \
  --dst-path 'predicts/output' \
  --get-layer 'face' \
  --face-thres 0.78
```