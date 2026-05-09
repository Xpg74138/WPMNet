# WPMNet: Weight-posture Joint Modeling for Pig Body Weight Estimation


## Repository Structure

```text
.
├── configs/                  # Exp configs 
├── src/  
│   ├── core/                 # Trainer, dataset, metrics, loss, model builder
│   ├── models/               # Backbones, fusion blocks, heads
│   └── utils/                # Label normalization and visualization helpers
├── train.py                  # Hydra training entry
├── test.py                   # Test / evaluation entry
├── run_experiments.py        # Sequential experiment runner
└── requirements.txt          # Python dependencies
```

## Installation

The code was developed with Python 3.11 and PyTorch 2.6.0.

```bash
conda create -n wpmnet python=3.11 -y
conda activate wpmnet

pip install -r requirements.txt
```

If you need to install PyTorch manually for a specific CUDA version, install PyTorch first from the official PyTorch instructions, then install the remaining packages:

```bash
pip install -r requirements.txt
```

## Dataset
The dataset used in this study will be made publicly available after publication.

After downloading the dataset, update the paths in:

```text
configs/best.yaml
```


## Pretrained Weights

Pretrained weights are available from Baidu Netdisk:

Link: https://pan.baidu.com/s/1dyg9hPV2zK6EhUsFtXA3hg?pwd=jc59
Extraction code: jc59

Suggested local layout:

```text
weights/
├── wpmnet_best_model.pth
├── swrh_best_model.pth
├── c_wpmh_best_model.pth
└── d_wpmh_best_model.pth
```

When evaluating a released checkpoint, set the checkpoint path in:

```text
configs/test_config.yaml
```

```yaml
testing:
  checkpoint: weights/wpmnet_best_model.pth
```

## MLflow Logging

Training logs metrics and checkpoints with MLflow. The default config expects an MLflow server at:

```text
http://localhost:2000
```

Start a local MLflow server before training:

```bash
mlflow server \
  --host 127.0.0.1 \
  --port 2000 \
  --backend-store-uri /path/to/mlflow/backend \
  --default-artifact-root /path/to/mlflow/artifacts
```

Then open:

```text
http://127.0.0.1:2000
```

If you use a different MLflow URI, update:

```text
configs/best.yaml
```

## Training

Training uses Hydra configs. The public release exposes the best model config as `configs/best.yaml`.

For one GPU:

```bash
torchrun --nproc_per_node=1 train.py --config-path=configs --config-name=best
```

For two GPUs:

```bash
torchrun --nproc_per_node=2 train.py --config-path=configs --config-name=best
```

You can override training parameters from the command line:

```bash
torchrun --nproc_per_node=2 train.py \
  --config-path=configs \
  --config-name=best \
  training.parameter.epochs=100 \
  training.parameter.batch_size=64 \
  optimizer.lr=0.00005
```

Useful visualization/logging overrides for faster training:

```bash
torchrun --nproc_per_node=2 train.py \
  --config-path=configs \
  --config-name=best \
  visualization.log_images=false \
  visualization.log_model_graph=false \
  visualization.visualization_interval=0 \
  visualization.log_interval=0
```

Checkpoints are saved by the trainer as MLflow artifacts, depending on the active run configuration.

## Testing

Set the test config and checkpoint in:

```text
configs/test_config.yaml
```

Example:

```yaml
defaults:
  - best
  - _self_

testing:
  checkpoint: weights/wpmnet_best_model.pth
  metrics:
    regression:
      - mae
      - rmse
      - r2
      - mape
    classification:
      - accuracy
      - f1
```

Run:

```bash
python test.py
```

Important notes:

- The experiment config in `configs/test_config.yaml` must match the checkpoint architecture.
- The test script uses the training split weights to initialize label normalization, matching the training procedure.
- Reported regression metrics include MAE, RMSE, R2, and MAPE.
- Reported posture metrics include accuracy and macro-F1.


## Citation

If this repository is useful for your research, please cite:

```bibtex
@article{TODO_WPMNet,
  title   = {Weight-posture joint modeling for body weight estimation of unconstrained pigs},
  author  = {TODO},
  journal = {TODO},
  year    = {TODO}
}
```


## Contact

For questions, please contact: B20243090880@cau.edu.cn.
