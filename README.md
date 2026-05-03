# LiteThermPose

サーマル画像向けの Top-Down 姿勢推定プロジェクトです。以下の機能を含みます。

- サーマル画像を厳密に白黒化して扱う前処理
- 軽量 CNN ベースの自作人物検出器
- 17 キーポイント + 可視性情報を考慮した姿勢推定
- 20 種類の軽量モデル切り替え
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
COCO 事前学習時も、サーマル向けに合わせて入力画像は白黒化して学習します。

## LWIRPOSE変換
LWIRPOSE は `S1 ... S7` の被験者フォルダと、各動作フォルダ内の `IR` / `IR_posepoints` から成る構造なので、
学習前に COCO 互換 JSON へ変換します。

S1 を test、S2-S7 を train/fine-tune 用に使う場合:

```bash
python scripts/convert_lwirpose_to_coco.py \
  --source /home/motomochi/LWIRPose \
  --target /home/motomochi/LTP2/data/lwirpose \
  --test-subjects S1
```

これで次ができます。

- `data/lwirpose/images/train/...`
- `data/lwirpose/images/test/...`
- `data/lwirpose/annotations/train.json`
- `data/lwirpose/annotations/val.json`
- `data/lwirpose/annotations/test.json`

`val.json` と `test.json` には S1 が入り、既存の評価系からそのまま test 相当として使えます。

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
- `dsconv_nano`
- `dsconv_pico`
- `resds_xs`
- `resds_s`
- `resds_m`
- `resds_l`
- `resds_xl`
- `hrnet_w18_small`
- `hrnet_w18`
- `hrnet_w32`
- `unet_tiny`
- `unet_small`
- `hourglass_tiny`
- `hourglass_small`
- `fpn_tiny`
- `fpn_small`
- `csp_pico`
- `csp_tiny`

## 単体学習

人物検出器の事前学習:

```bash
python train_detector.py --config configs/coco_pretrain.yaml --model dsconv_s --output outputs/coco_detector_dsconv_s
```

姿勢推定器のファインチューニング:

```bash
python train_pose.py --config configs/openthermalpose2_finetune.yaml --model dsconv_s --output outputs/thermal_pose_dsconv_s
```

同じ `output` ディレクトリに `detector_last.pt` または `pose_last.pt` がある場合は、自動で途中から再開します。

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

## 2台で10モデルずつ実行

20 モデルを 2 台で均等に分けて回せます。  
分割内容の確認:

```bash
python list_model_assignments.py --num-machines 2
```

1台目:

```bash
python run_model_suite_distributed.py \
  --pretrain-config configs/coco_pretrain.yaml \
  --finetune-config configs/openthermalpose2_finetune.yaml \
  --machine-index 0 \
  --num-machines 2 \
  --output outputs/model_suite_distributed
```

2台目:

```bash
python run_model_suite_distributed.py \
  --pretrain-config configs/coco_pretrain.yaml \
  --finetune-config configs/openthermalpose2_finetune.yaml \
  --machine-index 1 \
  --num-machines 2 \
  --output outputs/model_suite_distributed
```

この設定では、モデル順を固定したまま先頭 10 モデルを machine 0、残り 10 モデルを machine 1 に割り当てます。
各マシン内では、`detector_device: cuda:0` と `pose_device: cuda:1` が別 GPU なら、
全モデル一括実験の `pretrain` と `finetune` の各段階で人物検出器学習と姿勢推定器学習を同時実行します。

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

追加で、研究比較向けの基本スペックも記録します。

- `det_latency_ms`
- `pose_latency_ms`
- `joint_latency_ms`
- `det_peak_memory_mb`
- `pose_peak_memory_mb`
- `joint_peak_memory_mb`
- `det_train_peak_memory_mb`
- `pose_train_peak_memory_mb`
- `detector_params_m`
- `pose_params_m`
- `total_params_m`
- `detector_model_size_mb`
- `pose_model_size_mb`
- `detector_checkpoint_mb`
- `pose_checkpoint_mb`
- `det_pre_train_seconds`
- `pose_pre_train_seconds`
- `det_fine_train_seconds`
- `pose_fine_train_seconds`
- `det_epoch_seconds_mean`
- `pose_epoch_seconds_mean`

## 論文級精度を狙う 5 モデル

精度重視で回すための 5 モデルセットも追加しています。

- `pose_resnet50`
- `pose_resnet101`
- `hrnet_w48`
- `higherhrnet_w32`
- `stacked_hourglass_large`

これらは軽量性よりも表現力を優先した構成です。既存の軽量 20 モデル群とは別に、
高精度候補だけをまとめて実験できます。

専用一括実験:

```bash
python run_paper_grade_suite.py \
  --pretrain-config configs/coco_pretrain.yaml \
  --finetune-config configs/openthermalpose2_finetune.yaml \
  --output outputs/paper_grade_suite
```

この 5 モデル専用の実験も、設定で `detector_device: cuda:0` と `pose_device: cuda:1` が分かれていれば、
各モデルごとの `pretrain` / `finetune` で 2 GPU を同時活用します。

## 超軽量 3 モデル

軽さと CPU / 低 VRAM 実行を優先した 3 モデルも追加しています。

- `dsconv_nano`
- `dsconv_pico`
- `csp_pico`

専用一括実験:

```bash
python run_ultralight_suite.py \
  --pretrain-config configs/coco_pretrain.yaml \
  --finetune-config configs/openthermalpose2_finetune.yaml \
  --output outputs/ultralight_suite
```

この 3 モデルは、既存の軽量群よりさらに小さい構成で、CPU 側の実行速度や VRAM 使用量を抑えたいときの候補です。

## Paper Grade Suite Update

`run_paper_grade_suite.py` は 6 モデル構成になりました。

- `pose_resnet50`
- `pose_resnet101`
- `hrnet_w48`
- `higherhrnet_w32`
- `thermhr_csp_bifpn`
- `stacked_hourglass_large`

追加した `thermhr_csp_bifpn` は、サーマル画像向けに

- 高解像度分岐
- CSP 系の軽量融合
- BiFPN 風の双方向融合
- チャネル注意

を組み合わせたハイブリッド構成です。  
既存の HRNet 系より少し軽くしつつ、細かい関節位置精度を落としすぎないことを狙っています。
## Paper Grade OOM 対策

paper-grade 6 モデルは通常設定だと GPU メモリ不足になりやすいため、
専用の低メモリ config を追加しています。

- `configs/coco_pretrain_paper_grade.yaml`
- `configs/openthermalpose2_finetune_paper_grade.yaml`

推奨実行例:

```bash
python run_paper_grade_suite.py \
  --pretrain-config configs/coco_pretrain_paper_grade.yaml \
  --finetune-config configs/openthermalpose2_finetune_paper_grade.yaml \
  --output outputs/paper_grade_suite
```

この設定では batch size を `8` に下げ、batch size に合わせて learning rate も
安全側に調整しています。通常の 20 モデル実験には既存 config をそのまま使えます。

## ResNet Plus Suite

ResNet 系の改良版も追加しています。

- `pose_resnet50_se`
- `pose_resnet101_se`
- `pose_resnet101_se_fuse`

実行例:

```bash
python run_resnet_plus_suite.py \
  --pretrain-config configs/coco_pretrain_paper_grade_fast.yaml \
  --finetune-config configs/openthermalpose2_finetune_paper_grade_fast.yaml \
  --output outputs/resnet_plus_suite
```

## Novel Thermal ResNet

SE 版をさらに改良した、サーマル向けの新規モデルも追加しています。

- `pose_resnet50_se_thermal`
- `pose_resnet101_se_thermal`
- `pose_resnet101_se_thermal_ppm`

追加した要素:

- `SE` に加えた `spatial contrast gate`
- 浅い特徴と深い特徴を重み付きで混ぜる `thermal fusion gate`
- `ppm` 版では簡易 `pyramid context`

実行例:

```bash
python run_resnet_novel_suite.py \
  --pretrain-config configs/coco_pretrain_paper_grade_fast.yaml \
  --finetune-config configs/openthermalpose2_finetune_paper_grade_fast.yaml \
  --output outputs/resnet_novel_suite
```

## Paper Grade 高速設定

速度優先で paper-grade 6 モデルを回すための config も追加しています。

- `configs/coco_pretrain_paper_grade_fast.yaml`
- `configs/openthermalpose2_finetune_paper_grade_fast.yaml`

この設定では以下を高速側に寄せています。

- pose 入力を `256x192` から `224x160` に縮小
- detector 入力を `320x320` から `256x256` に縮小
- COCO pretrain を `24 epoch`
- OpenThermalPose2 finetune を `30 epoch`

速度優先の実行例:

```bash
python run_paper_grade_suite.py \
  --pretrain-config configs/coco_pretrain_paper_grade_fast.yaml \
  --finetune-config configs/openthermalpose2_finetune_paper_grade_fast.yaml \
  --output outputs/paper_grade_suite_fast
```
## 静止画推論

1 枚画像を読み込んで、人物検出と 17 点姿勢推定の結果を書き出せます。

```bash
python infer_image.py \
  --config configs/openthermalpose2_finetune_paper_grade_fast.yaml \
  --model pose_resnet101_se_thermal_ppm \
  --detector outputs/resnet_novel_suite/pose_resnet101_se_thermal_ppm/finetune_detector/detector_last.pt \
  --pose outputs/resnet_novel_suite/pose_resnet101_se_thermal_ppm/finetune_pose/pose_last.pt \
  --input path/to/input.png \
  --output outputs/inference/result_pose_resnet101_se_thermal_ppm.png
```
## ディレクトリ推論

フォルダ内の画像をまとめて推論したいときは `infer_directory.py` を使います。

```bash
python infer_directory.py \
  --config configs/openthermalpose2_finetune_paper_grade_fast.yaml \
  --model pose_resnet101_se_thermal_ppm \
  --detector outputs/resnet_novel_suite/pose_resnet101_se_thermal_ppm/finetune_detector/detector_last.pt \
  --pose outputs/resnet_novel_suite/pose_resnet101_se_thermal_ppm/finetune_pose/pose_last.pt \
  --input-dir path/to/input_dir \
  --output-dir outputs/inference_dir/pose_resnet101_se_thermal_ppm
```
## OpenThermalPose2 Test ベンチ

`data/raw` 配下の OpenThermalPose2 `testimage` / `testlabel` を自動探索して、
精度・速度・可視化比較をまとめて行うスクリプトです。

取得する主な項目:

- `det_recall50_test`, `det_precision50_test`, `det_mean_iou_test`
- `det_map50_95_test`, `det_map50_test`, `det_map75_test`
- `pose_map50_95_test`, `pose_map50_test`, `pose_map75_test`
- `pose_mean_oks_test`
- `pose_pck20_test`, `pose_pck10_test`, `pose_pck05_test`
- `pose_pckh50_test`, `pose_pckh30_test`
- `pose_mean_error_px_test`, `pose_mean_error_bbox_test`
- `joint_score_test`, `joint_pckh50_test`
- CPU / GPU の detector, pose, joint の `fps` と `latency`
- パラメータ数、checkpoint サイズ、GPU peak memory
- モデルごとの推論画像と比較シート

実行例:

```bash
python benchmark_otp2_testset.py \
  --config configs/openthermalpose2_finetune_paper_grade_fast.yaml \
  --suite-dir outputs/resnet_plus_suite \
  --suite-dir outputs/resnet_novel_suite \
  --models pose_resnet101_se,pose_resnet50_se_thermal,pose_resnet101_se_thermal_ppm \
  --output outputs/otp2_test_benchmark
```

## Pose高精度モデル
Pose を高精度寄りで回すときの本命は `pose_resnet101_se_thermal_ppm` です。
このモデル向けに、学習解像度と epoch を少し厚めにした専用 config を用意しています。

- `configs/coco_pretrain_pose_highacc.yaml`
- `configs/openthermalpose2_finetune_pose_highacc.yaml`
- `run_pose_high_accuracy_pipeline.py`

個別に回す場合:

```bash
python train_detector.py \
  --config configs/coco_pretrain_pose_highacc.yaml \
  --model pose_resnet101_se_thermal_ppm \
  --output outputs/pose_high_accuracy/pose_resnet101_se_thermal_ppm/pretrain_detector
```

```bash
python train_pose.py \
  --config configs/coco_pretrain_pose_highacc.yaml \
  --model pose_resnet101_se_thermal_ppm \
  --output outputs/pose_high_accuracy/pose_resnet101_se_thermal_ppm/pretrain_pose
```

```bash
python train_detector.py \
  --config configs/openthermalpose2_finetune_pose_highacc.yaml \
  --model pose_resnet101_se_thermal_ppm \
  --weights outputs/pose_high_accuracy/pose_resnet101_se_thermal_ppm/pretrain_detector/detector_last.pt \
  --output outputs/pose_high_accuracy/pose_resnet101_se_thermal_ppm/finetune_detector
```

```bash
python train_pose.py \
  --config configs/openthermalpose2_finetune_pose_highacc.yaml \
  --model pose_resnet101_se_thermal_ppm \
  --weights outputs/pose_high_accuracy/pose_resnet101_se_thermal_ppm/pretrain_pose/pose_last.pt \
  --output outputs/pose_high_accuracy/pose_resnet101_se_thermal_ppm/finetune_pose
```

OpenThermalPose2 test で精度・速度・可視化まで含めて測る場合:

```bash
python benchmark_otp2_testset.py \
  --config configs/openthermalpose2_finetune_pose_highacc.yaml \
  --suite-dir outputs/pose_high_accuracy \
  --models pose_resnet101_se_thermal_ppm \
  --test-images-dir data/raw/openthermalpose2_extracted/otp2_dataset/test/images \
  --test-labels-dir data/raw/openthermalpose2_extracted/otp2_dataset/test/labels \
  --output outputs/otp2_test_benchmark_pose_high_accuracy
```

最初から最後まで一発で回す場合:

```bash
python run_pose_high_accuracy_pipeline.py \
  --pretrain-config configs/coco_pretrain_pose_highacc.yaml \
  --finetune-config configs/openthermalpose2_finetune_pose_highacc.yaml \
  --train-output outputs/pose_high_accuracy \
  --benchmark-output outputs/otp2_test_benchmark_pose_high_accuracy \
  --test-images-dir data/raw/openthermalpose2_extracted/otp2_dataset/test/images \
  --test-labels-dir data/raw/openthermalpose2_extracted/otp2_dataset/test/labels
```

## Direct Pose回帰モデル
heatmap ではなく、人物 crop から 17 点の正規化座標と可視性を直接回帰する pose モデルも使えます。
既存の heatmap 系はそのまま残してあり、direct 系は別ファイル実装です。

- `src/lite_therm_pose/models/pose_direct.py`
- `configs/coco_pretrain_pose_direct.yaml`
- `configs/openthermalpose2_finetune_pose_direct.yaml`
- `run_pose_direct_pipeline.py`

本命の direct モデル:

- `pose_resnet101_se_thermal_direct`

一発実行:

```bash
python run_pose_direct_pipeline.py \
  --pretrain-config configs/coco_pretrain_pose_direct.yaml \
  --finetune-config configs/openthermalpose2_finetune_pose_direct.yaml \
  --train-output outputs/pose_direct \
  --benchmark-output outputs/otp2_test_benchmark_pose_direct \
  --test-images-dir data/raw/openthermalpose2_extracted/otp2_dataset/test/images \
  --test-labels-dir data/raw/openthermalpose2_extracted/otp2_dataset/test/labels
```

個別に回す場合:

```bash
python train_detector.py \
  --config configs/coco_pretrain_pose_direct.yaml \
  --model pose_resnet101_se_thermal_direct \
  --output outputs/pose_direct/pose_resnet101_se_thermal_direct/pretrain_detector
```

```bash
python train_pose.py \
  --config configs/coco_pretrain_pose_direct.yaml \
  --model pose_resnet101_se_thermal_direct \
  --output outputs/pose_direct/pose_resnet101_se_thermal_direct/pretrain_pose
```

```bash
python train_detector.py \
  --config configs/openthermalpose2_finetune_pose_direct.yaml \
  --model pose_resnet101_se_thermal_direct \
  --weights outputs/pose_direct/pose_resnet101_se_thermal_direct/pretrain_detector/detector_last.pt \
  --output outputs/pose_direct/pose_resnet101_se_thermal_direct/finetune_detector
```

```bash
python train_pose.py \
  --config configs/openthermalpose2_finetune_pose_direct.yaml \
  --model pose_resnet101_se_thermal_direct \
  --weights outputs/pose_direct/pose_resnet101_se_thermal_direct/pretrain_pose/pose_last.pt \
  --output outputs/pose_direct/pose_resnet101_se_thermal_direct/finetune_pose
```

## LWIRPOSEでの最終ファインチューニング
OpenThermalPose2 / COCO とは別に、LWIRPOSE の `S2-S7` を最終 fine-tune、`S1` を test として使うときは、
direct 本命モデルを次で回せます。

- `configs/lwirpose_finetune_pose_direct.yaml`
- `run_lwirpose_direct_pipeline.py`

```bash
python run_lwirpose_direct_pipeline.py \
  --pretrain-config configs/coco_pretrain_pose_direct.yaml \
  --finetune-config configs/lwirpose_finetune_pose_direct.yaml \
  --output outputs/lwirpose_pose_direct
```

この出力の `summary.csv` / `summary.md` は、`val_annotation_file=data/lwirpose/annotations/val.json`
を使うので、そのまま `S1` に対する精度になります。

OpenThermalPose2 で fine-tune 済みの direct 重みを初期値に使いたい場合は、こちらを使います。

- `run_lwirpose_direct_from_otp.py`

```bash
python run_lwirpose_direct_from_otp.py \
  --config configs/lwirpose_finetune_pose_direct.yaml \
  --source-suite outputs/pose_direct \
  --output outputs/lwirpose_pose_direct_from_otp
```

この流れでは、`outputs/pose_direct/pose_resnet101_se_thermal_direct/finetune_detector/detector_last.pt` と
`outputs/pose_direct/pose_resnet101_se_thermal_direct/finetune_pose/pose_last.pt` を初期重みにして、
LWIRPOSE の `S2-S7` で追加 fine-tune し、`S1` で評価します。
