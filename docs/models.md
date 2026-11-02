# 学習済みモデルの仕様と取得方法

本検証で配布する学習済みモデルは、ランダム初期化からの公式不整地歩行学習（`baseline`）を起点として段階的に溝地形で継続学習した主系列6段階のチェックポイント（計約41 MB）である。すべてのモデルは物理シミュレーション（Isaac Lab / Newton）上の四脚ロボット Unitree Go2 向けに学習されたものである。

## 1. 配布モデル一覧（Release `v1.0.0`）

6つのモデルファイルは GitHub Releases の [`v1.0.0`](https://github.com/kiri-i/pai-go2-32667/releases/tag/v1.0.0) で配布しており、各ファイルのサイズ・SHA-256ハッシュ・親段階・乱数シード・反復数は [`../models/manifest.json`](../models/manifest.json) に記録されている。

| モデルID (`<id>`) | ファイル名 | 親段階（開始重み） | 学習地形 | 反復数・並列環境数 | 乱数シード | 本検証における役割 |
|---|---|---|---|---|---|---|
| `baseline` | [`baseline.pt`](https://github.com/kiri-i/pai-go2-32667/releases/download/v1.0.0/baseline.pt) | なし（ランダム初期化） | 公式不整地 | 1,500・4,096 | `42` | 公式不整地タスクで初期獲得した基礎歩行方策 |
| `first` | [`first.pt`](https://github.com/kiri-i/pai-go2-32667/releases/download/v1.0.0/first.pt) | `baseline` | 溝 $20\text{--}45\,\text{cm}$ | 400・1,024 | `42` | 小幅な溝地形での継続学習（標準報酬） |
| `second` | [`second.pt`](https://github.com/kiri-i/pai-go2-32667/releases/download/v1.0.0/second.pt) | `first` | 溝 $40\text{--}70\,\text{cm}$ | 400・1,024 | `42` | 中幅の溝地形への段階的拡張（標準報酬） |
| `stage3-exposure` | [`stage3-exposure.pt`](https://github.com/kiri-i/pai-go2-32667/releases/download/v1.0.0/stage3-exposure.pt) | `second` | 溝 $55\text{--}90\,\text{cm}$ | 400・1,024 | `42` | 広幅の溝地形への適応（標準報酬）。**主比較2系統の共通の親モデル** |
| `stage4-standard` | [`stage4-standard.pt`](https://github.com/kiri-i/pai-go2-32667/releases/download/v1.0.0/stage4-standard.pt) | `stage3-exposure` | 溝 $55\text{--}90\,\text{cm}$ | 400・1,024 | `42` | **標準報酬モデル（対照群）**。標準報酬を維持して400反復追加学習 |
| `stage4-balance` | [`stage4-balance.pt`](https://github.com/kiri-i/pai-go2-32667/releases/download/v1.0.0/stage4-balance.pt) | `stage3-exposure` | 溝 $55\text{--}90\,\text{cm}$ | 400・1,024 | `42` | **報酬変更モデル（変更群）**。5係数緩和＋7項目（5加点・2減点）で400反復追加学習 |

主比較（[`report/report.md`](report/report.md)）では、同じ `stage3-exposure` を親とし、同じ溝幅分布（$55\text{--}90\,\text{cm}$・深さ $40\,\text{cm}$）と反復数（400反復、`seed=42`）で分岐させた `stage4-standard`（対照群）と `stage4-balance`（変更群）を比較する。

## 2. チェックポイントの構造と継続学習時の扱い

- **保存されている状態**：配布している各 `.pt` ファイルは、元の実験で出力されたチェックポイントそのものであり、方策ネットワーク（`actor`）、価値ネットワーク（`critic`）、および最適化器（`optimizer`）の状態辞書と学習イテレーション情報を含む。
- **継続学習における継承仕様**：`scripts/terrain_train.py` で親モデルを指定して次段階の継続学習を開始する際は、親チェックポイントから `actor` と `critic` の重みのみを読み込み、`optimizer` の状態および更新回数（イテレーションカウンタ）は引き継がずに初期化する。

## 3. 取得方法と保存先

モデルの取得には `scripts/download_models.py` を使用する。取得されたファイルはすべて `.local/models/<id>.pt` に保存され、ダウンロード直後に [`../models/manifest.json`](../models/manifest.json) に記載されたファイルサイズおよびSHA-256ハッシュとの一致が検証される。

モデルのダウンロードとハッシュ照合自体はGPUを必要とせず、`scripts/setup_sim.sh` によるシミュレータ環境（`.local/sim`）やロボットUSD資産（`.local/assets`）のセットアップとは独立してCPU環境のみで実行できる。一方、取得したモデルを用いて評価推論（`scripts/evaluate.py`）や継続学習を実行する際には、CUDA対応GPUおよび構築済みのシミュレータ環境が必要となる（手順は [`reproduction.md`](reproduction.md) を参照）。

### 既定の取得（主比較の2モデル）

引数を指定しない場合、主評価で使用する `stage4-standard.pt` と `stage4-balance.pt` の2ファイルを取得する。

```bash
uv run python scripts/download_models.py
```

### 全6モデルの一括取得

`baseline` から `stage4-balance` までの全6段階のモデルを一括で取得する場合は `--all` を指定する。

```bash
uv run python scripts/download_models.py --all
```

### 単一モデルの指定取得

特定の段階（例えば主比較の親モデルである `stage3-exposure`）のみを取得する場合は `--model` オプションでモデルIDを指定する。

```bash
uv run python scripts/download_models.py --model stage3-exposure
```

### 直接URLからの取得

各ファイルは以下のURL形式から直接取得することも可能である。

```text
https://github.com/kiri-i/pai-go2-32667/releases/download/v1.0.0/<filename>
```

### 既存ファイルの上書き保護

保存先（`.local/models/<id>.pt`）に同名ファイルが既に存在する場合、`download_models.py` はそのSHA-256ハッシュを計算する。期待されるハッシュと一致すれば再ダウンロードを省略し、ハッシュが異なる場合は既存データの意図しない消失を防ぐため上書きを拒否してエラー終了する。

## 4. ハッシュ管理と利用上の制約

- **マニフェストによる真正性確認**：各モデルのSHA-256ハッシュとバイト数は [`../models/manifest.json`](../models/manifest.json) を正本とする。元実験のステージ別記録である `../evidence/stage3_manifest.json` および `../evidence/stage4_manifest.json` とも対応している。
- **再学習時のハッシュ非保証**：同じコードと `seed=42` を用いてゼロまたは途中段階から新規に学習を実行した場合、生成されるチェックポイントのSHA-256ハッシュが配布モデルと一致することは保証しない。報告書記載の評価結果や動作を点検する場合は、配布モデルの重みおよび収録済みの評価記録（[`../evidence/evaluation_manifest.json`](../evidence/evaluation_manifest.json)）を使用する。
- **観測・行動空間と適用範囲の制約**：すべてのモデルは235次元観測（胴体・関節状態・直前行動・速度指令の48次元＋機体周囲の地面高さ走査187次元）を入力とし、12関節の位置目標を出力するシミュレーション専用の方策である。シミュレーション上の地面高さ走査を前提としており、実機ロボットへそのまま適用することはできない。また、`stage4-balance` の基準開始位置（$0.0\,\text{m}$）で確認された通過挙動は両岸同時支持（$0.16\,\text{s}$ / $8\,\text{step}$）を伴う跨ぎ動作であり、4脚同時離地によるジャンプ動作ではない。
