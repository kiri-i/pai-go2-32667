# 学習と評価の再現手順

本検証の確認・再現手順は、(1) 収録済み評価記録JSONのハッシュ照合と主結果のCPU再集計、(2) 配布済みの学習済み重みを用いたGPUシミュレータ上での幅 $85\,\text{cm}$ 溝の評価実行、(3) ランダム初期化からの主系列6段階の学習、(4) GPUを用いないCPU単体検査に分かれる。実行手順の全体像とコマンド仕様を以下に示す。

## 1. 実行環境の構築

### システム要件

- OS / アーキテクチャ：Linux x86_64
- 必須ソフトウェア：Python 3.12、`uv`（収録済み評価JSONの再集計・CPU単体検査・モデル取得はCPUのみで実行可能。シミュレータ上での評価実行および学習にはCUDA対応GPU・ドライバ・`ffmpeg` が必須）
- シミュレータ構成（[`../sim/uv.lock`](../sim/uv.lock) で固定）：Isaac Lab 3.0.0、Isaac Sim 6.2、Newton 1.6.1、PyTorch 2.11.0

Isaac Simを初めて起動する際はライセンス同意が求められる。ライセンス条項を確認し同意した環境では、非対話実行の前に以下の環境変数を設定できる。

```bash
export OMNI_KIT_ACCEPT_EULA=YES
```

### シミュレータ環境のセットアップ

```bash
bash scripts/setup_sim.sh
```

このスクリプトは以下の処理を順に実行する。
1. `sim/uv.lock` に従い、通常のPython環境から分離された仮想環境を `.local/sim` に構築する。
2. 公式タスク設定ファイル群のSHA-256ハッシュを [`../evidence/baseline_source_audit.json`](../evidence/baseline_source_audit.json)（基準コミット `7529e35ff98a851d62015a369b38e29827676f46`）と照合した上で、材質APIクラス名の互換修正（`PhysxRigidBodyMaterialCfg` から `RigidBodyMaterialCfg` への置換）を適用する。
3. Unitree Go2 の公式ロボットUSDアセット2件をNVIDIA配布元から `.local/assets` へ取得し、SHA-256ハッシュを検証する。

なお、公式USDアセット2件の取得とハッシュ照合および既存の固定 `.local/sim` 環境での動作は確認済みであるが、新規のクリーン環境における `.local/sim` 全体のゼロからの構築完了までは本検証の確認範囲として主張しない。環境を再構築する場合も、既存の `.local/sim` や成果物を直接削除せず、別のチェックアウトディレクトリを使用することを推奨する。

## 2. 配布済み重みを用いた評価と動画生成

配布されている学習済み重みを取得し、幅 $85\,\text{cm}$ の固定溝で標準報酬モデル（`stage4-standard`）と報酬変更モデル（`stage4-balance`）を評価する手順は以下の通りである。モデルの取得（`download_models.py`）はGPUや `.local/sim` の構築を必要とせず独立して実行できるが、評価の実行（`evaluate.py`）にはGPUシミュレータ環境が必要である。

### モデルの取得（GPU不要）

```bash
uv run python scripts/download_models.py
```

既定で `.local/models/stage4-standard.pt` と `.local/models/stage4-balance.pt` が取得され、[`../models/manifest.json`](../models/manifest.json) のハッシュと照合される（全6段階のモデルを取得する場合は `--all` を付与する。詳細は [`models.md`](models.md) を参照）。

### 評価の実行

```bash
uv run python scripts/evaluate.py \
  --checkpoint .local/models/stage4-standard.pt --gap-cm 85 \
  --output runs/evaluation-85-standard --view readable
uv run python scripts/evaluate.py \
  --checkpoint .local/models/stage4-balance.pt --gap-cm 85 \
  --output runs/evaluation-85-balance --view readable
```

### 評価仕様と判定基準

- **溝幅と開始位置**：`--gap-cm` には `75`, `80`, `85`, `90` を指定できる（主比較は `85`）。各溝幅において、開始位置を基準（$0.0\,\text{m}$）、$15\,\text{cm}$ 遠い位置（手前、$-0.15\,\text{m}$）、$15\,\text{cm}$ 近い位置（奥、$+0.15\,\text{m}$）の3条件に変えて評価する。
- **速度指令**：エピソード開始直後の $1.0\,\text{s}$ 間は初期姿勢を安定させるため静止指令を与え、その後は前進速度 $1.0\,\text{m/s}$ と初期方位保持指令を与える。
- **描画オプション**：動画は制御周期と同じ $50\,\text{fps}$・等速で出力される。`--view readable` は地形配色とカメラ位置を調整した判読用視点、`--view original` は元の評価描画、`--view side` は真横からの側面視点である。
- **成功判定（`scripts/course_metrics.py`）**：全4脚が対岸の所定範囲内に入った後の連続 $1.0\,\text{s}$ 間において、胴体姿勢と位置・足先高さ条件を維持し、4脚それぞれが期間中に1回以上対岸に接地（$F_z > 1\,\text{N}$）し、かつ $80\%$ 以上のフレームで少なくとも1脚が接地支持している場合に成功と判定する（常時4脚同時接地は要求しない）。溝への転落、側方逸脱、制限時間内未完了は失敗となる。
- **エラー時の挙動**：個別の開始位置で失敗した場合も結果JSONを保存して次の開始位置へ進む。シミュレータの起動失敗などにより結果JSONが生成されなかった場合は、その時点で処理を停止する。

## 3. ゼロからの主系列6段階の学習

ランダム初期化から主比較の2モデルに至る主系列は6段階で構成される。各段階は前段（親段階）の出力重みを引き継いで学習する。

| 順序 | 段階名 (`stage`) | 開始重み（親） | 学習地形 | 更新回数・並列環境数 | 報酬設定 |
|---:|---|---|---|---|---|
| 1 | `baseline` | ランダム初期化 | 公式不整地 | 1,500・4,096 | 公式不整地歩行報酬 |
| 2 | `first` | `baseline` | 溝 $20\text{--}45\,\text{cm}$ | 400・1,024 | 標準報酬 |
| 3 | `second` | `first` | 溝 $40\text{--}70\,\text{cm}$ | 400・1,024 | 標準報酬 |
| 4 | `stage3-exposure` | `second` | 溝 $55\text{--}90\,\text{cm}$ | 400・1,024 | 標準報酬 |
| 5 | `stage4-standard` | `stage3-exposure` | 溝 $55\text{--}90\,\text{cm}$ | 400・1,024 | 標準報酬（対照群） |
| 6 | `stage4-balance` | `stage3-exposure` | 溝 $55\text{--}90\,\text{cm}$ | 400・1,024 | 報酬変更（変更群：5緩和＋5加点・2減点） |

すべての段階において乱数シードは `seed=42` である。継続学習（`first` 以降）では親段階の `actor` と `critic` の重みのみを読み込み、`optimizer` の状態と更新回数は引き継がない。

### コマンド表示と `--execute` による実行の区別

`scripts/reproduce.py` は、既定では実行されるコマンド列と依存関係を表示するのみであり、実際の学習処理は起動しない。学習を実行するには `--execute` フラグを付与する。既存の出力ディレクトリが存在する場合、上書きは拒否される。

```bash
# 主系列6段階の実行コマンド一覧を表示（実行はしない）
uv run python scripts/reproduce.py all

# 主系列6段階を順番に実行
uv run python scripts/reproduce.py baseline --execute
uv run python scripts/reproduce.py first --execute
uv run python scripts/reproduce.py second --execute
uv run python scripts/reproduce.py stage3-exposure --execute
uv run python scripts/reproduce.py stage4-standard --execute
uv run python scripts/reproduce.py stage4-balance --execute
```

なお、リポジトリには補足探索用の段階定義も残されているが、`reproduce.py all` が扱う主系列は上記6段階のみである。補足探索条件は段階名を明示指定した場合にのみ使用される。

新規に学習したチェックポイントを評価する場合は、`evaluate.py` の `--checkpoint` に `runs/stage4-standard/final.pt` または `runs/stage4-balance/final.pt` を指定する。

## 4. 収録済み評価記録の検証・配布重みでの再推論・新規学習の違い

本リポジトリでは、「収録済み評価記録の数値検証」と「シミュレータ上での新規推論・新規学習」を明確に区別して扱う。

- **収録済み評価記録のCPU再集計（報告書数値の確定検証）**：`evidence/evaluation/*.json`（計7本）は、主評価で得られた評価JSONから公開時に特定ホストの絶対パスのみを置換した記録であり、[`../evidence/evaluation_manifest.json`](../evidence/evaluation_manifest.json) に元ハッシュ・公開ハッシュ・実カメラ設定・報告書本文および比較動画との試行対応が記録されている。`uv run python scripts/summarize_results.py` を実行すると、GPUを使用せずにJSONのSHA-256ハッシュを照合した上で、主評価の通過成功数（`stage4-standard` が $0/3$、`stage4-balance` が $2/3$）および基準開始位置（$0.0\,\text{m}$、`close` 視点の1試行）における両岸同時支持（$0.16\,\text{s}$ / $8\,\text{step}$）を再集計する（両岸支持時間は基準位置1試行の定量値であり、成功2試行全体への一般化ではない）。
- **配布重みでの新規推論（別試行としての扱いと確認範囲）**：`.local/models/` に取得した配布重みは元の実験で得られたチェックポイントそのものであるが、同じ重みファイルと評価コードを用いても新規推論は元の収録とは独立した試行として扱う。実際、`stage4-balance.pt` は学習直後の元評価では成功 $3/3$ であったのに対し、映像・接触記録用の実行では開始位置が最も近い $+0.15\,\text{m}$ 条件で失敗し成功 $2/3$ となった（詳細は [`report/appendix.md`](report/appendix.md) および [`provenance.md`](provenance.md) を参照）。この実行間差が生じた原因は未確定であり、特定の原因を断定しない。なお、公開スクリプトの動作点検として、既存の固定 `.local/sim` 環境を `--python` で指定し、配布 `stage4-balance.pt`（ハッシュ一致確認済み）を用いて描画なし・$250\,\text{step}$・基準開始位置（$0.0\,\text{m}$）の新規試行を1回実行した際には `course_outcome: "success"`（`error: null`）となることを確認しているが、この単発確認は主評価成績（$2/3$）へは加算せず、また新規構築環境での再現を保証するものでもない。
- **ゼロまたは途中段階からの新規学習**：`seed=42` を固定して `reproduce.py <stage> --execute` を実行した場合でも、得られる重みファイルのSHA-256ハッシュが [`../models/manifest.json`](../models/manifest.json) の記録値と一致することは保証しない。新規学習は、同一の学習手順と報酬設計から同様の溝渡り挙動が獲得されるかを確認する用途として扱う。

## 5. CPU検査と単体テスト

GPUやIsaac Simを起動せずに、収録済み評価JSONのハッシュ照合・再集計、ならびに地形形状の計算、成功判定ロジック、学習段階の依存定義を検証するコマンドを用意している。

```bash
uv sync --locked --link-mode copy
uv run python scripts/summarize_results.py
uv run python scripts/check_actions_policy.py
uv run python -m compileall -q scripts
(cd scripts && uv run --project .. python -m unittest test_course_suite test_terrain_geometry test_recipe)
```

また、シミュレータ環境（`.local/sim`）のPyTorchを使用する接触状態機械の補足テストは以下のコマンドで実行できる。

```bash
(cd scripts && ../.local/sim/bin/python -m unittest test_jump_sequence)
```

これらのCPU検査は収録済み評価記録の集計整合性およびスクリプトの構文・判定・設定ロジックを確認するものであり、GPU上での学習収束や第三者環境での新規推論の数値一致を保証するものではない。
