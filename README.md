# LiteThermPose

CPUでもリアルタイム動作を狙える、CNNベースのサーマル向けTop-Down姿勢推定実装です。

- 自作の軽量CNN人検出器
- Top-Down型の姿勢推定器
- COCOでの事前学習からOpenThermalPose2へのファインチューニング
- 部分観測でも扱いやすい `part_heatmaps` 出力

## モデル構成

### 人検出器

`TinyPersonDetector` は depthwise separable convolution を用いた軽量検出器です。

- 入力: 320x320
- 出力: 人中心ヒートマップ、bbox サイズ回帰、center offset 回帰
- 復号: top-k + NMS

### 姿勢推定器

`TopDownPoseCNN` は検出bboxごとにクロップして推定するTop-Downモデルです。

- 入力: 256x192 人物クロップ
- 出力: COCO互換キーポイントヒートマップ、6部位の `part_heatmaps`、各キーポイントの可視性

`part_heatmaps` により、全身が見えないケースでも頭部、胴体、腕、脚単位での部分検出を扱えます。

## データセット

### COCO事前学習

- 画像: `data/coco/train2017`
- アノテーション: `data/coco/annotations/person_keypoints_train2017.json`

### OpenThermalPose2

- 画像: `data/openthermalpose2/images`
- アノテーション: `data/openthermalpose2/annotations/train.json`

アノテーションはCOCO互換を想定しつつ、必要なら `body_parts` を各人物に追加できます。

```json
{
  "id": 1,
  "image_id": 10,
  "category_id": 1,
  "bbox": [30, 40, 120, 220],
  "keypoints": [],
  "body_parts": {
    "head": [90, 55],
    "torso": [92, 120],
    "left_arm": [60, 125],
    "right_arm": [123, 122],
    "left_leg": [78, 210],
    "right_leg": [105, 212]
  }
}
```

## セットアップ

```bash
pip install -e .
```

Anaconda を使う場合は次でも構築できます。

```bash
chmod +x scripts/setup_conda.sh
./scripts/setup_conda.sh
conda activate lite-therm-pose
```

## データセット取得

```bash
chmod +x scripts/download_datasets.sh
./scripts/download_datasets.sh all
```

個別取得もできます。

```bash
./scripts/download_datasets.sh coco
./scripts/download_datasets.sh openthermalpose2
```

このスクリプトは次を行います。

- COCO train2017 / val2017 / annotations を `data/coco` に展開
- OpenThermalPose2 を `data/raw` に保存して展開
- 既知レイアウトなら `data/openthermalpose2/images` と `data/openthermalpose2/annotations/train.json` に配置

OpenThermalPose2 の配布形式が COCO JSON でない場合は、raw データを保持したまま変換が必要です。

## 学習

```bash
python train_detector.py --config configs/coco_pretrain.yaml --output outputs/coco_detector
python train_pose.py --config configs/coco_pretrain.yaml --output outputs/coco_pose
python train_detector.py --config configs/openthermalpose2_finetune.yaml --weights outputs/coco_detector/detector_last.pt --output outputs/thermal_detector
python train_pose.py --config configs/openthermalpose2_finetune.yaml --weights outputs/coco_pose/pose_last.pt --output outputs/thermal_pose
```

## リアルタイム推論

```bash
python infer_realtime.py --config configs/openthermalpose2_finetune.yaml --detector outputs/thermal_detector/detector_last.pt --pose outputs/thermal_pose/pose_last.pt --source 0
```

## 実装メモ

- CPUリアルタイム重視のため、backbone は軽量CNNです
- Top-Downなので、検出器と姿勢推定器を独立に改善できます
- サーマル画像は `grayscale: true` にして1ch入力で扱えます
- `partial_crop_prob` により、部分しか映らない訓練サンプルを意図的に増やせます

## 今後の改善候補

- OKS mAP 評価スクリプトの追加
- INT8量子化とONNX書き出し
- OpenThermalPose2の実注釈仕様に合わせたローダ最適化
- tracker を加えた時系列安定化
