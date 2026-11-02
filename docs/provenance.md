# コードと実験の来歴

本検証のシミュレーション基盤、公式設定からの変更範囲、ゼロからの初期学習と重み継承の経路、および評価記録と証拠ハッシュの対応を以下に示す。第三者ソフトウェアのライセンスと権利関係は [`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) に記載している。

## 1. シミュレーション基盤と公式タスクとの対応

本検証は Isaac Lab 3.0.0、Isaac Sim 6.2、Newton 1.6.1、PyTorch 2.11.0 を基盤とする。公式不整地歩行タスクの設定照合基準は Isaac Lab のコミット `7529e35ff98a851d62015a369b38e29827676f46` である。

- 公式タスク本体はリポジトリに同梱せず、[`../sim/uv.lock`](../sim/uv.lock) で固定したwheelパッケージから `.local/sim` へ導入する。wheel全体を当該Gitコミットと同一とみなすものではなく、使用する設定ファイル群のSHA-256ハッシュを [`../evidence/baseline_source_audit.json`](../evidence/baseline_source_audit.json) によって個別に検証している。
- 公式設定に対して適用した変更は、`scripts/setup_sim.sh` における地形材質クラス名 `PhysxRigidBodyMaterialCfg` から `RigidBodyMaterialCfg` へのAPI互換置換のみであり、公式不整地歩行タスクの報酬係数や物理設定は変更していない。

## 2. ランダム初期化からの学習と段階的継承

本検証の学習済みモデルは、外部で配布されている事前学習済み歩行モデルを流用したものではなく、すべてランダム初期化から学習を開始している。

1. **公式不整地での初期学習（`baseline`）**：`scripts/train_baseline.py` は、公式CLIの設定読み込みとPPO学習器の呼び出しを、ローカルのUSD配置（`.local/assets`）および相対出力パスに対応させた実行入口である。ランダム初期化から公式不整地タスクで1,500反復（4,096並列環境、`seed=42`）学習し、`baseline.pt` を得た。
2. **溝地形での段階的継続学習（`first` → `second` → `stage3-exposure`）**：`scripts/terrain_train.py` により、標準の不整地歩行報酬を維持したまま、溝幅分布を $20\text{--}45\,\text{cm}$（`first`）、$40\text{--}70\,\text{cm}$（`second`）、$55\text{--}90\,\text{cm}$（`stage3-exposure`）へと広げて各400反復（1,024並列環境、`seed=42`）継続学習した。
3. **主比較2系統への分岐（`stage4-standard` / `stage4-balance`）**：共通の親モデル `stage3-exposure.pt` から、同じ溝地形分布（幅 $55\text{--}90\,\text{cm}$・深さ $40\,\text{cm}$、カリキュラム無効）・400反復・`seed=42` の条件で、標準報酬を維持した対照群（`stage4-standard`）と、5項目の安定性ペナルティ緩和＋7項目（加点5・減点2）を追加した変更群（`stage4-balance`）をそれぞれ追加学習した。

各段階の継続学習では親チェックポイントの `actor` と `critic` の重みのみを引き継ぎ、`optimizer` の状態と更新回数はリセットしている。

## 3. 独自実装と実験コードの変更点

元実験のスクリプトと本リポジトリ内のファイルとの対応関係およびハッシュは [`../evidence/source_manifest.json`](../evidence/source_manifest.json) に記録している。

- **溝地形学習と報酬設計（`scripts/terrain_train.py`）**：深さ $40\,\text{cm}$ の溝地形生成、カリキュラム無効化、および標準報酬と変更報酬の切り替えを実装している。なお、コードや設定内に含まれる `jump_balance` などの名称は実装上の条件識別子であり、4脚が同時に離地する空中ジャンプの獲得を意味しない。幅の大きな溝を通過することと空中移動で溝を越えることは区別して評価しており、本検証で確認された通過挙動は両岸同時支持を伴う跨ぎ動作である。
- **固定幅評価器と成功判定（`scripts/evaluate.py`, `scripts/course_metrics.py`, `scripts/probe_*.py`）**：評価器は `Robot-sim-runtime`（コミット `0c45f63`）の `evaluate_lab30.py` を起点に、固定溝地形、3開始位置（$-0.15,\ 0.0,\ +0.15\,\text{m}$）、足先接触反力（$F_z > 1\,\text{N}$）と胴体姿勢に基づく連続 $1.0\,\text{s}$ の成功判定、および視点別の動画記録（`probe_step.py`, `probe_readable.py`, `probe_side.py`）を追加したものである。
- **環境非依存化**：ローカル資産管理（`.local/assets`, `.local/models`）と明示的な `--checkpoint` 指定を導入し、特定ホストの絶対パスへの依存や未使用のメタデータ欄を除去した。地形形状、報酬係数、成功判定の数値基準は元実験から変更していない。

## 4. 学習直後の元評価と映像・接触記録用評価の別集計

同一の学習済みチェックポイント（`stage4-standard.pt` および `stage4-balance.pt`）を用いた幅 $85\,\text{cm}$ 溝・3開始位置の評価には、学習完了直後の元評価と、比較映像および足先接触解析を記録した主評価の2回の実行記録が存在する。両者は合算せず別集計として扱う。

| 評価実行 | 標準報酬モデル (`stage4-standard`) | 報酬変更モデル (`stage4-balance`) | 備考 |
|---|---|---|---|
| **本検証の主評価（映像・接触記録用実行）** | 成功 **0 / 3**（全3位置で失敗） | 成功 **2 / 3**（$-0.15\,\text{m},\ 0.0\,\text{m}$ で成功、$+0.15\,\text{m}$ で溝内接触失敗） | [`report/report.md`](report/report.md) の主結果、比較動画 `comparison.mp4`、および基準位置（$0.0\,\text{m}$、`close` 視点1試行）での両岸同時支持 $0.16\,\text{s}$（$8\,\text{step}$）の解析に対応する実行 |
| **学習直後の元評価** | 成功 **0 / 3**（全3位置で失敗） | 成功 **3 / 3**（全3位置で成功） | 学習完了直後に実施した評価記録（詳細は [`report/appendix.md`](report/appendix.md) Appendix A.3） |

同じ重みファイルと評価コードであっても、助走距離が最も短い $+0.15\,\text{m}$（$15\,\text{cm}$ 近い位置）条件において実行間で成否が分かれた。この差が生じた具体的な原因は未確定であり、特定の要因を断定せず観測事実として留保し、実行ごとに別試行として扱っている。報告書本文（[`report/report.md`](report/report.md)）では、動画および接触反力時系列データと完全に対応する「成功 $0/3$ 対 $2/3$」を主結果として報告している。

主評価で記録された数値JSON（公開時に特定ホストの絶対パスのみを置換した計7本）は `../evidence/evaluation/*.json` に収められており、[`../evidence/evaluation_manifest.json`](../evidence/evaluation_manifest.json) に置換前後のSHA-256ハッシュ、実際のカメラ設定、および報告書本文・比較動画との試行対応を記録している。内訳は主比較2モデル×3開始位置の6試行と、比較動画の側面映像用に別試行として収録された `side` 視点1試行の計7本であり、両岸同時支持 $0.16\,\text{s}$（$8\,\text{step}$）の定量解析は基準開始位置（$0.0\,\text{m}$）の `close` 視点1試行に対応する（`uv run python scripts/summarize_results.py` でハッシュ照合と再集計が可能）。

## 5. 証拠ファイルとハッシュ一覧

| ファイルパス | 記録内容 |
|---|---|
| [`../models/manifest.json`](../models/manifest.json) | Release `v1.0.0` で配布する全6モデルのファイルサイズ、SHA-256ハッシュ、親段階、乱数シード（`seed=42`）、反復数 |
| [`../evidence/evaluation_manifest.json`](../evidence/evaluation_manifest.json) | 収録済み評価JSON（`../evidence/evaluation/*.json`、計7本）の元ハッシュ、絶対パス置換後の公開ハッシュ、実カメラ設定、本文解析（`close`）および比較動画（`side` 別試行を含む）との試行対応 |
| [`../evidence/baseline_source_audit.json`](../evidence/baseline_source_audit.json) | Isaac Lab 基準コミット（`7529e35ff98a851d62015a369b38e29827676f46`）に対する公式設定ファイルのSHA-256照合結果 |
| [`../evidence/stage3_manifest.json`](../evidence/stage3_manifest.json) | 第3段階（`stage3-exposure` 等）の元実験チェックポイントのハッシュと学習条件記録 |
| [`../evidence/stage4_manifest.json`](../evidence/stage4_manifest.json) | 第4段階（`stage4-standard`, `stage4-balance` 等）の元実験チェックポイントのハッシュと学習条件記録 |
| [`../evidence/source_manifest.json`](../evidence/source_manifest.json) | 元実験コードとリポジトリ内スクリプトの対応関係およびSHA-256ハッシュ |

`stage3_manifest.json`、`stage4_manifest.json`、および `models/manifest.json` に記録されたチェックポイントのSHA-256ハッシュは、元の実験で生成された重みファイルを識別するためのものであり、新規に再学習した際に同一ハッシュが生成されることは保証しない。
