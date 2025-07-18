# Detectron2 with HuggingFace Accelerate

This document explains how to use HuggingFace Accelerate with Detectron2 for distributed training, mixed precision, and other training optimizations.

## Installation

First, make sure you have HuggingFace Accelerate installed:

```bash
pip install accelerate
```

## Quick Start

### 1. Single GPU Training

For single GPU training, you can still use the standard training script but with Accelerate enabled:

```bash
python tools/train_net_accelerate.py \
    --config-file configs/COCO-Detection/faster_rcnn_R_50_FPN_1x_accelerate.yaml \
    --num-gpus 1
```

### 2. Multi-GPU Training

For multi-GPU training, use the `accelerate launch` command:

```bash
accelerate launch --num_processes=2 tools/train_net_accelerate.py \
    --config-file configs/COCO-Detection/faster_rcnn_R_50_FPN_1x_accelerate.yaml
```

### 3. Mixed Precision Training

Enable mixed precision with Accelerate:

```bash
accelerate launch --mixed_precision=fp16 --num_processes=2 tools/train_net_accelerate.py \
    --config-file configs/COCO-Detection/faster_rcnn_R_50_FPN_1x_accelerate.yaml
```

### 4. Multi-Node Training

For multi-node training:

```bash
# On main node (machine_rank=0)
accelerate launch --num_processes=4 --num_machines=2 --main_process_ip=MAIN_NODE_IP --main_process_port=29500 tools/train_net_accelerate.py \
    --config-file configs/COCO-Detection/faster_rcnn_R_50_FPN_1x_accelerate.yaml

# On other nodes (machine_rank=1,2,...)
accelerate launch --num_processes=4 --num_machines=2 --main_process_ip=MAIN_NODE_IP --main_process_port=29500 --machine_rank=1 tools/train_net_accelerate.py \
    --config-file configs/COCO-Detection/faster_rcnn_R_50_FPN_1x_accelerate.yaml
```

## Configuration

### Accelerate Configuration in Config Files

Add the following to your config file to enable Accelerate:

```yaml
SOLVER:
  ACCELERATE:
    ENABLED: True
    MIXED_PRECISION: "fp16"  # Options: "no", "fp16", "bf16"
    GRADIENT_ACCUMULATION_STEPS: 1
    DATALOADER_CONFIG:
      SPLIT_BATCHES: False
      EVEN_BATCHES: True
    # Additional accelerate kwargs
    KWARGS:
      # Custom accelerate configuration can go here
```

### Accelerate Config File

You can also create an accelerate config file using:

```bash
accelerate config
```

This will create a config file (usually at `~/.cache/huggingface/accelerate/default_config.yaml`) that will be automatically used by accelerate.

## Features Supported

- **Distributed Training**: Multi-GPU and multi-node training
- **Mixed Precision**: FP16 and BF16 training
- **Gradient Accumulation**: Simulate larger batch sizes
- **Automatic Device Placement**: Automatic model and data placement
- **DeepSpeed Integration**: Use DeepSpeed optimizations (configure via accelerate config)
- **CPU Training**: Train on CPU if needed

## Differences from Standard Detectron2

1. **No Manual DDP**: When using AccelerateTrainer, don't wrap your model with DistributedDataParallel manually. Accelerate handles this.

2. **Process Spawning**: Use `accelerate launch` instead of relying on Detectron2's internal process spawning.

3. **Mixed Precision**: Use Accelerate's mixed precision instead of PyTorch's AMP when ACCELERATE.ENABLED=True.

4. **LR Scheduler**: The learning rate scheduler is automatically prepared by Accelerate.

## Backward Compatibility

You can still use the standard training workflow. When `SOLVER.ACCELERATE.ENABLED=False` (default), Detectron2 will use the standard SimpleTrainer or AMPTrainer.

## Environment Variables

Accelerate sets several environment variables that Detectron2 will detect:

- `LOCAL_RANK`: Local process rank (set by accelerate launch)
- `ACCELERATE_MIXED_PRECISION`: Mixed precision mode
- `ACCELERATE_GRADIENT_ACCUMULATION_STEPS`: Gradient accumulation steps

## Examples

### Example 1: Basic Multi-GPU Training

```bash
accelerate launch --num_processes=4 tools/train_net_accelerate.py \
    --config-file configs/COCO-Detection/faster_rcnn_R_50_FPN_1x_accelerate.yaml \
    MODEL.WEIGHTS detectron2://ImageNetPretrained/MSRA/R-50.pkl \
    SOLVER.IMS_PER_BATCH 16 \
    SOLVER.BASE_LR 0.001
```

### Example 2: Mixed Precision with Gradient Accumulation

```bash
accelerate launch --mixed_precision=fp16 --num_processes=2 tools/train_net_accelerate.py \
    --config-file configs/COCO-Detection/faster_rcnn_R_50_FPN_1x_accelerate.yaml \
    SOLVER.ACCELERATE.GRADIENT_ACCUMULATION_STEPS 2 \
    SOLVER.IMS_PER_BATCH 8  # Effective batch size will be 8 * 2 * 2 = 32
```

### Example 3: Using with Existing train_net.py

You can also enable Accelerate with the standard train_net.py by setting the config:

```bash
python tools/train_net.py \
    --config-file configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml \
    SOLVER.ACCELERATE.ENABLED True \
    SOLVER.ACCELERATE.MIXED_PRECISION fp16
```

Note: When using the standard train_net.py with Accelerate, make sure to use `accelerate launch` if you want multi-GPU training.

## Troubleshooting

1. **ImportError: No module named 'accelerate'**: Install accelerate with `pip install accelerate`

2. **CUDA out of memory**: Try reducing batch size or enabling gradient accumulation:
   ```yaml
   SOLVER:
     IMS_PER_BATCH: 8  # Reduce batch size
     ACCELERATE:
       GRADIENT_ACCUMULATION_STEPS: 2  # Accumulate gradients
   ```

3. **Different results with Accelerate**: This can happen due to different random number generation or synchronization. Ensure deterministic training by setting appropriate seeds.

4. **Multi-node training not working**: Make sure all nodes can communicate and use the same IP/port configuration.

## Performance Tips

1. **Use Mixed Precision**: Enable fp16 or bf16 for faster training and reduced memory usage
2. **Gradient Accumulation**: Use gradient accumulation to simulate larger batch sizes without increasing memory usage
3. **Optimize Data Loading**: Consider increasing the number of data loader workers
4. **Profile Your Training**: Use accelerate's profiling tools to identify bottlenecks
