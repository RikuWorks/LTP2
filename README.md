# LiteThermPose

Thermal top-down pose estimation project with:

- strict grayscale preprocessing for thermal images
- custom lightweight CNN person detector
- 17-keypoint pose estimation with visibility-aware supervision
- model zoo with around 10 lightweight variants
- COCO pretraining and OpenThermalPose2 finetuning
- detector-only / pose-only / joint pipeline evaluation
- experiment suite that runs pretrain -> finetune -> test -> report

## Setup

```bash
pip install -e .
```

For Ubuntu with Conda:

```bash
chmod +x scripts/setup_conda.sh
./scripts/setup_conda.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lite-therm-pose
python scripts/verify_gpu.py
```

## Datasets

Download datasets with:

```bash
chmod +x scripts/download_datasets.sh
./scripts/download_datasets.sh all
```

Expected layout:

- `data/coco/train2017`
- `data/coco/val2017`
- `data/coco/annotations/person_keypoints_train2017.json`
- `data/coco/annotations/person_keypoints_val2017.json`
- `data/openthermalpose2/images`
- `data/openthermalpose2/annotations/train.json`
- `data/openthermalpose2/annotations/val.json`

OpenThermalPose2 annotations are expected to be COCO-like. Optional `body_parts` fields are supported for partial detection supervision.

## Model Zoo

List supported models:

```bash
python list_models.py
```

Current variants:

- `dsconv_xs`
- `dsconv_s`
- `dsconv_m`
- `dsconv_l`
- `dsconv_xl`
- `resds_xs`
- `resds_s`
- `resds_m`
- `resds_l`
- `resds_xl`

## Train One Model

Detector pretraining:

```bash
python train_detector.py --config configs/coco_pretrain.yaml --model dsconv_s --output outputs/coco_detector_dsconv_s
```

Pose finetuning:

```bash
python train_pose.py --config configs/openthermalpose2_finetune.yaml --model dsconv_s --output outputs/thermal_pose_dsconv_s
```

## Run Full Suite

This runs all requested models through:

1. detector pretraining
2. pose pretraining
3. detector finetuning
4. pose finetuning
5. detector-only test
6. pose-only test
7. joint pipeline test
8. CSV / Markdown / PNG summary export

```bash
python run_model_suite.py \
  --pretrain-config configs/coco_pretrain.yaml \
  --finetune-config configs/openthermalpose2_finetune.yaml \
  --output outputs/model_suite
```

Run a subset only:

```bash
python run_model_suite.py \
  --pretrain-config configs/coco_pretrain.yaml \
  --finetune-config configs/openthermalpose2_finetune.yaml \
  --models dsconv_s,resds_m,resds_l \
  --output outputs/model_suite_subset
```

## Evaluate Existing Weights

```bash
python evaluate_models.py \
  --config configs/openthermalpose2_finetune.yaml \
  --model dsconv_s \
  --detector-weights outputs/thermal_detector/detector_last.pt \
  --pose-weights outputs/thermal_pose/pose_last.pt
```

## Realtime Inference

```bash
python infer_realtime.py \
  --config configs/openthermalpose2_finetune.yaml \
  --model dsconv_s \
  --detector outputs/thermal_detector/detector_last.pt \
  --pose outputs/thermal_pose/pose_last.pt \
  --source 0
```

## Multi-GPU

Use separate GPUs for detector and pose in config files:

```yaml
runtime:
  device: auto
  detector_device: cuda:0
  pose_device: cuda:1
```

## Reports

`run_model_suite.py` writes:

- `summary.csv`
- `summary.md`
- `summary.png`

Typical metrics:

- `det_recall50`
- `det_mean_iou`
- `pose_pck20`
- `pose_visibility_acc`
- `joint_score`
- `det_fps`
- `pose_fps`
- `joint_fps`
