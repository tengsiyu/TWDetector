# TWDetector

**TWDetector: Diffusion-Based Two-Wheeler Detection in Low-Light Environments**

TWDetector is an object detection framework designed for two-wheeler perception
under challenging illumination conditions. The framework follows the iterative
box-denoising formulation of DiffusionDet and integrates a Swin-DyT backbone
with a bidirectional feature pyramid network (BiFPN). This design is intended to
improve feature representation and multiscale fusion in scenes affected by low
illumination, image noise, blur, occlusion, and small object size.

The detector recognizes three object categories:

- `rider`
- `bike`
- `motor`

## Quantitative Results

The principal result on the Data-TWC validation set is reported below.

| Sampling steps | AP | AP50 | AP75 | APs | APm | APl |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 25.540 | 55.104 | 20.015 | 11.231 | 28.992 | 41.859 |

The ablation study on the number of diffusion sampling steps demonstrates the
accuracy-efficiency trade-off:

| Sampling steps | AP | APs |
| ---: | ---: | ---: |
| 1 | 24.162 | 10.229 |
| 4 | 25.068 | 11.191 |
| 8 | 24.674 | 11.132 |
| 16 | 25.540 | 11.231 |

The experiment configurations, logs, metrics, COCO-format predictions, and
qualitative results are provided in `results/`.

## Repository Structure

```text
TWDetector/
├── environment.yml
├── LICENSE
├── results/
│   ├── sampling_steps/             # Sampling-step ablation results
│   └── visualization/              # Qualitative and comparative results
└── DiffusionDet/
    ├── train.py                     # Training and evaluation entry point
    ├── demo.py                      # Image, video, and webcam inference
    ├── configs/
    │   ├── Base-DiffusionDet.yaml
    │   └── diffdet_demo_swinbase_DyT_bifpn.yaml
    ├── diffusiondet/                # Model implementation
    └── datasets/coco/               # COCO-format datasets and annotations
```

## Environment

The recommended platform is Linux or WSL2 with an NVIDIA GPU and Conda. The
reference environment uses:

- Python 3.8
- PyTorch 2.2.0
- torchvision 0.17.0
- CUDA 12.1
- Detectron2 0.6

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate env_yang
```

Verify the principal dependencies:

```bash
python -c "import torch, detectron2, cv2; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

If Detectron2 cannot be resolved from the environment specification, install
the corresponding release from source after installing the remaining
dependencies:

```bash
python -m pip install "git+https://github.com/facebookresearch/detectron2.git@v0.6"
```

The provided environment was exported from a Linux CUDA installation. Native
Windows users are therefore advised to use WSL2.

## Dataset Organization

The project uses the COCO object detection format. The prepared data splits and
their annotations are located under `DiffusionDet/datasets/coco/`:

```text
coco/
├── annotations/
│   ├── instances_train2017.json
│   ├── instances_val2017.json
│   └── instances_val2017_main.json
├── train2017/
├── val2017/
└── val2017_main/
```

| Split | Images | Source | Purpose |
| --- | ---: | --- | --- |
| `train2017` | 6,394 | BDD100K | Data-TWC training set |
| `val2017` | 343 | BDD100K | Primary Data-TWC validation set |
| `val2017_main` | 883 | CODA | Supplementary Data-Lowlight test set |

The default dataset root can be replaced with an alternative COCO-format
directory:

```bash
export TWDETECTOR_DATASET=/absolute/path/to/coco
```

The default evaluation split is `val2017`. Select the supplementary low-light
split with:

```bash
export TWDETECTOR_VAL_SPLIT=val2017_main
```

The images originate from
[BDD100K](https://bdd-data.berkeley.edu/download.html) and
[CODA](https://coda-dataset.github.io/download.html). Users are responsible for
complying with the licenses, terms of use, and citation requirements of the
original datasets.

## Configuration

The primary configuration is:

```text
DiffusionDet/configs/diffdet_demo_swinbase_DyT_bifpn.yaml
```

Its principal settings are:

- Backbone: `build_swintransformer_DyT_bifpn_backbone`
- Number of classes: `3`
- Number of initial proposals: `100`
- Default sampling steps: `16`
- Images per batch: `1`
- Maximum training iterations: `45,500`

All commands below are executed from the `DiffusionDet` directory:

```bash
cd DiffusionDet
```

The checkpoint path may be supplied through `MODEL.WEIGHTS`:

```text
/absolute/path/to/model.pth
```

## Evaluation

Evaluate a checkpoint on the selected validation split:

```bash
python train.py \
  --num-gpus 1 \
  --eval-only \
  --config-file configs/diffdet_demo_swinbase_DyT_bifpn.yaml \
  MODEL.WEIGHTS /absolute/path/to/model.pth \
  OUTPUT_DIR ./OUTPUT/evaluation
```

The evaluator reports the standard COCO metrics, including AP, AP50, AP75, APs,
APm, and APl. Prediction artifacts are written to the `inference/` directory
under the specified output path.

## Inference

Create an output directory:

```bash
mkdir -p output
```

### Single-Image Inference

```bash
python demo.py \
  --config-file configs/diffdet_demo_swinbase_DyT_bifpn.yaml \
  --input /absolute/path/to/image.jpg \
  --output ./output/image_result.jpg \
  --confidence-threshold 0.5 \
  --opts MODEL.WEIGHTS /absolute/path/to/model.pth
```

### Batch Image Inference

```bash
python demo.py \
  --config-file configs/diffdet_demo_swinbase_DyT_bifpn.yaml \
  --input "/absolute/path/to/images/*.jpg" \
  --output ./output \
  --confidence-threshold 0.5 \
  --opts MODEL.WEIGHTS /absolute/path/to/model.pth
```

### Video Inference

```bash
python demo.py \
  --config-file configs/diffdet_demo_swinbase_DyT_bifpn.yaml \
  --video-input /absolute/path/to/input.mp4 \
  --output ./output/output.mp4 \
  --confidence-threshold 0.5 \
  --opts MODEL.WEIGHTS /absolute/path/to/model.pth
```

### Webcam Inference

```bash
python demo.py \
  --config-file configs/diffdet_demo_swinbase_DyT_bifpn.yaml \
  --webcam \
  --confidence-threshold 0.5 \
  --opts MODEL.WEIGHTS /absolute/path/to/model.pth
```

## Training

Train the model with the reference configuration:

```bash
python train.py \
  --num-gpus 1 \
  --config-file configs/diffdet_demo_swinbase_DyT_bifpn.yaml \
  MODEL.WEIGHTS ""
```

Continue training or fine-tune from a specified checkpoint:

```bash
python train.py \
  --num-gpus 1 \
  --config-file configs/diffdet_demo_swinbase_DyT_bifpn.yaml \
  MODEL.WEIGHTS /absolute/path/to/model.pth
```

Detectron2 configuration values can be overridden from the command line:

```bash
python train.py \
  --num-gpus 1 \
  --config-file configs/diffdet_demo_swinbase_DyT_bifpn.yaml \
  SOLVER.IMS_PER_BATCH 2 \
  SOLVER.MAX_ITER 20000 \
  OUTPUT_DIR ./OUTPUT/my_experiment \
  MODEL.WEIGHTS /absolute/path/to/model.pth
```

Training outputs are written to the configured `OUTPUT_DIR`.

## Sampling-Step Analysis

The default number of diffusion sampling steps is 16. For example, one-step
inference can be evaluated as follows:

```bash
python train.py \
  --num-gpus 1 \
  --eval-only \
  --config-file configs/diffdet_demo_swinbase_DyT_bifpn.yaml \
  MODEL.WEIGHTS /absolute/path/to/model.pth \
  MODEL.DiffusionDet.SAMPLE_STEP 1 \
  OUTPUT_DIR ./OUTPUT/sample_step_1
```

Typical values are `1`, `4`, `8`, and `16`. A controlled comparison should keep
the checkpoint, dataset, and all remaining hyperparameters fixed.

## Qualitative Results

Model comparison:

![TWDetector model comparison](results/visualization/效果图.png)

Representative TWDetector predictions:

![TWDetector result 0181](results/visualization/our%20model/0181.png)

![TWDetector result 0371](results/visualization/our%20model/0371.png)

## License

The source code is distributed under the terms provided in [LICENSE](LICENSE).
The datasets and incorporated third-party components remain subject to their
respective licenses and terms of use.
