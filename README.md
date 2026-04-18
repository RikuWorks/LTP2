# LiteThermPose

サーマル画像向けの Top-Down 姿勢推定プロジェクトです。以下の機能を含みます。

- サーマル画像を厳密に白黒化して扱う前処理
- 軽量 CNN ベースの自作人物検出器
- 17 キーポイント + 可視性情報を考慮した姿勢推定
- 10 種類前後の軽量モデル切り替え
- COCO 事前学習 + OpenThermalPose2 ファインチューニング
- 人検出のみ / 骨格推定のみ / 結合パイプラインの個別評価
- `pretrain -> finetune -> test -> report` をまとめて実行する一括実験

## セットアップ

```bash
pip install -e .
```

## Ubuntu + Conda 環境構築

```bash
chmod +x scripts/setup_conda.sh
./scripts/setup_conda.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lite-therm-pose
python scripts/verify_gpu.py
```

RTX 3090 x2 環境では、デフォルトで Conda 版 PyTorch + CUDA 12.4 を使うようにしてあります。  
この環境では `pip install torch torchvision` のような入れ方はしないでください。

既存環境が壊れている場合は、次で修復できます。

```bash
chmod +x scripts/repair_torch_conda.sh
./scripts/repair_torch_conda.sh
```

## データセット取得

```bash
chmod +x scripts/download_datasets.sh
./scripts/download_datasets.sh all
```

ダウンロード並列数を増やしたい場合:

```bash
DOWNLOAD_JOBS=16 DOWNLOAD_SPLITS=16 ./scripts/download_datasets.sh all
```

- `DOWNLOAD_JOBS`: 同時ダウンロード数
- `DOWNLOAD_SPLITS`: 1ファイルあたりの分割接続数
- `aria2c` がある場合は分割並列ダウンロードを優先使用
- ない場合は `wget` または `curl` にフォールバック

想定ディレクトリ構成:

- `data/coco/train2017`
- `data/coco/val2017`
- `data/coco/annotations/person_keypoints_train2017.json`
- `data/coco/annotations/person_keypoints_val2017.json`
- `data/openthermalpose2/images`
- `data/openthermalpose2/annotations/train.json`
- `data/openthermalpose2/annotations/val.json`

OpenThermalPose2 の注釈は COCO 互換を想定しています。`body_parts` を含めると部分検出学習にも使えます。

## モデル一覧

利用可能なモデル一覧:

```bash
python list_models.py
```

現在のモデル:

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

## 単体学習

人物検出器の事前学習:

```bash
python train_detector.py --config configs/coco_pretrain.yaml --model dsconv_s --output outputs/coco_detector_dsconv_s
```

姿勢推定器のファインチューニング:

```bash
python train_pose.py --config configs/openthermalpose2_finetune.yaml --model dsconv_s --output outputs/thermal_pose_dsconv_s
```

## 全モデル一括実験

次をモデルごとにまとめて実行します。

1. 人物検出器の事前学習
2. 姿勢推定器の事前学習
3. 人物検出器のファインチューニング
4. 姿勢推定器のファインチューニング
5. 人検出のみテスト
6. 骨格推定のみテスト
7. 結合パイプラインテスト
8. CSV / Markdown / PNG レポート出力

```bash
python run_model_suite.py \
  --pretrain-config configs/coco_pretrain.yaml \
  --finetune-config configs/openthermalpose2_finetune.yaml \
  --output outputs/model_suite
```

一部モデルだけ回す場合:

```bash
python run_model_suite.py \
  --pretrain-config configs/coco_pretrain.yaml \
  --finetune-config configs/openthermalpose2_finetune.yaml \
  --models dsconv_s,resds_m,resds_l \
  --output outputs/model_suite_subset
```

## 学習済み重みの評価

```bash
python evaluate_models.py \
  --config configs/openthermalpose2_finetune.yaml \
  --model dsconv_s \
  --detector-weights outputs/thermal_detector/detector_last.pt \
  --pose-weights outputs/thermal_pose/pose_last.pt
```

## リアルタイム推論

```bash
python infer_realtime.py \
  --config configs/openthermalpose2_finetune.yaml \
  --model dsconv_s \
  --detector outputs/thermal_detector/detector_last.pt \
  --pose outputs/thermal_pose/pose_last.pt \
  --source 0
```

## マルチ GPU

設定ファイルで検出器と姿勢推定器を別 GPU に分けられます。

```yaml
runtime:
  device: auto
  detector_device: cuda:0
  pose_device: cuda:1
```

`scripts/verify_gpu.py` では見えている全 CUDA デバイスを表示します。  
RTX 3090 x2 の正常な環境なら、少なくとも次のようになります。

- `device_count: 2`
- `device_0: NVIDIA GeForce RTX 3090`
- `device_1: NVIDIA GeForce RTX 3090`

## 出力レポート

`run_model_suite.py` は以下を出力します。

- `summary.csv`
- `summary.md`
- `summary.png`

主な指標:

- `det_recall50`
- `det_mean_iou`
- `pose_pck20`
- `pose_visibility_acc`
- `joint_score`
- `det_fps`
- `pose_fps`
- `joint_fps`
