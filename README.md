# WPMNet: Weight-posture Joint Modeling for Pig Body Weight Estimation

This repository contains the code for the paper:

**Weight-posture joint modeling for body weight estimation of unconstrained pigs**

The project trains and evaluates deep learning models for pig body weight estimation from RGB-D images. It supports single-task weight regression and joint weight-posture learning, including SWRH, C-WPMH, D-WPMH, and WPMNet-style RGB-D fusion models.

> Data and trained weights will be released separately. Placeholder links are kept below.

## News

- **Data download:** TODO: add public dataset URL.
- **Pretrained weights:** TODO: add release / cloud storage URL.
- **Paper / citation:** TODO: add DOI, arXiv, or journal link after publication.

## Repository Structure

```text
.
├── configs/                  # Hydra configs for data, models, optimizer, scheduler, experiments
│   ├── data/                 # Dataset split and loader configs
│   ├── model/                # Backbone/head/fusion architecture configs
│   ├── training/             # Training task and metric configs
│   ├── optimizer/            # Optimizer configs
│   └── scheduler/            # LR scheduler configs
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

## Data Preparation

The dataset is expected to be organized with fixed train/validation/test split files. The split must not be randomly regenerated if you want to reproduce the reported results.

Recommended layout:

```text
data/
└── 700x700/
    ├── train.txt
    ├── val.txt
    ├── test.txt
    ├── train/
    │   ├── rgb/
    │   └── depth/
    ├── val/
    │   ├── rgb/
    │   └── depth/
    └── test/
        ├── rgb/
        └── depth/
```

Each line in `train.txt`, `val.txt`, and `test.txt` should follow:

```text
/path/to/rgb_image.jpg,/path/to/depth_image.png,weight_kg,posture_name
```

Example:

```text
data/700x700/train/rgb/000001.jpg,data/700x700/train/depth/000001.png,82.4,standing
```

Supported posture names are defined in the data config:

```yaml
class_names:
  - standing
  - lyingonstomach
  - lyingonside
```

After downloading the dataset, update the paths in:

```text
configs/data/pig_weight_truedepth.yaml
```

For example:

```yaml
train:
  path: data/700x700/train.txt
val:
  path: data/700x700/val.txt
test:
  path: data/700x700/test.txt
```

## Pretrained Weights

Pretrained weights will be released later.

Suggested local layout:

```text
weights/
├── wpmnet_best_model.pth        # TODO: add download link
├── swrh_best_model.pth          # TODO: add download link
├── c_wpmh_best_model.pth        # TODO: add download link
└── d_wpmh_best_model.pth        # TODO: add download link
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
configs/default.yaml
```

## Training

Training uses Hydra configs. The default training entry currently uses `configs/exp31.yaml`.

For one GPU:

```bash
torchrun --nproc_per_node=1 train.py --config-path=configs --config-name=exp31
```

For two GPUs:

```bash
torchrun --nproc_per_node=2 train.py --config-path=configs --config-name=exp31
```

You can override training parameters from the command line:

```bash
torchrun --nproc_per_node=2 train.py \
  --config-path=configs \
  --config-name=exp31 \
  training.parameter.epochs=100 \
  training.parameter.batch_size=64 \
  optimizer.lr=0.00005
```

Useful visualization/logging overrides for faster training:

```bash
torchrun --nproc_per_node=2 train.py \
  --config-path=configs \
  --config-name=exp31 \
  visualization.log_images=false \
  visualization.log_model_graph=false \
  visualization.visualization_interval=0 \
  visualization.log_interval=0
```

Checkpoints are saved by the trainer as MLflow artifacts, depending on the active run configuration.

## Batch Experiments

`run_experiments.py` runs selected experiment configs sequentially and keeps printing logs.

```bash
NPROC_PER_NODE=2 TRAIN_EPOCHS=30 SCHEDULER_T_MAX=100 python run_experiments.py
```

To change which experiments are run, edit:

```text
run_experiments.py
```

especially `get_experiment_configs()`.

## Testing

Set the test config and checkpoint in:

```text
configs/test_config.yaml
```

Example:

```yaml
defaults:
  - exp31
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

## Main Configuration Files

Common experiment configs:

```text
configs/exp31.yaml    # WPMNet-style RGB-D fusion model
configs/exp32.yaml    # Coupled-head variant
configs/exp1.yaml     # Baseline config example
configs/exp5.yaml     # C-WPMH-style config example
configs/exp9.yaml     # D-WPMH-style config example
```

Model configs are under:

```text
configs/model/
```

Data configs are under:

```text
configs/data/
```

## Metrics

Regression metrics:

- MAE
- RMSE
- R2, coefficient of determination: `1 - SS_res / SS_tot`
- MAPE

Posture classification metrics:

- Accuracy
- Macro-F1

## Reproducibility Notes

- Keep the original `train/val/test` split unchanged.
- Do not use the validation set as the test set.
- Record the config file, checkpoint path, random seed, code version, and hardware for each experiment.
- When testing a checkpoint, ensure `configs/test_config.yaml` inherits the same experiment config used during training.
- For multi-seed experiments, use the same split and only change the random seed.

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

## License

TODO: add license information before public release.

## Contact

TODO: add maintainer email or project page.
