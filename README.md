# フィジカルAI講座 最終課題 — Go2の溝踏破（32667）

Isaac Labの四足ロボットGo2を対象に、学習地形の変更と報酬の再設計を比較する実験コードです。地形高スキャンを含む235次元の観測を使用し、入力や行動空間を追加せずに溝の踏破を学習します。シミュレーションによる評価であり、実機検証ではありません。

## 構成

| ファイル | 内容 |
|---|---|
| `scripts/train_baseline.py` | 公式不整地タスクのランダム初期化学習 |
| `scripts/terrain_train.py` | 溝地形での継続学習、報酬条件の比較 |
| `scripts/reproduce.py` | 学習の依存順序とコマンド生成 |
| `scripts/evaluate.py` | 固定幅・3開始位置での評価と動画記録 |
| `scripts/course_metrics.py` | 接触履歴と姿勢を含む成功判定 |
| `scripts/probe_step.py` | 元の評価描画 |
| `scripts/probe_readable.py`, `scripts/probe_side.py` | 地形の配色とカメラを変更した評価描画 |
| `sim/uv.lock` | 実験に用いたシミュレータ依存の固定版 |
| `evidence/` | 公式タスクの照合結果、比較条件、元ファイルのハッシュ |

## 実行環境

Linux x86_64、Python 3.12、CUDA対応GPU、uv、ffmpegが必要です。実験環境はIsaac Lab 3.0.0、Isaac Sim 6.2、Newton 1.6.1、PyTorch 2.11.0です。GPUドライバはインストール処理に含めません。通常のPython環境とは分離して構築します。依存の配布条件は各提供元に従ってください。

Isaac Simを初めて起動するとライセンス同意を求められます。内容を確認し同意した環境では、実行前に`export OMNI_KIT_ACCEPT_EULA=YES`を設定できます。

```bash
bash scripts/setup_sim.sh
```

この処理は`sim/uv.lock`から`.local/sim`を構築し、記録済みの材質API互換修正をハッシュ照合後に適用します。ロボットUSDはNVIDIA配布元から`.local/assets`へ取得し、SHA-256を検証します。USD、学習済み重み、収録データはリポジトリに含めません。環境を再構築する場合も、既存環境を削除せず別のチェックアウトを使用してください。

公式設定の照合基準はIsaac Labコミット`7529e35ff98a851d62015a369b38e29827676f46`です。インストールには固定wheelを使用し、`evidence/baseline_source_audit.json`の各設定ファイルのハッシュを検証します。wheel全体を当該Gitコミットと同一とみなすものではありません。

## 学習

以下はコマンドを表示するだけです。実行には`--execute`を付けます。既存の出力ディレクトリへの上書きは拒否します。

```bash
uv run python scripts/reproduce.py all
uv run python scripts/reproduce.py baseline --execute
uv run python scripts/reproduce.py first --execute
uv run python scripts/reproduce.py second --execute
uv run python scripts/reproduce.py stage3-continued --execute
uv run python scripts/reproduce.py stage3-exposure --execute
uv run python scripts/reproduce.py stage4-standard --execute
uv run python scripts/reproduce.py stage4-balance --execute
```

| 段階 | 開始重み | 地形 | 更新回数・並列環境数 |
|---|---|---|---|
| baseline | ランダム | 公式不整地 | 1,500・4,096 |
| first | baseline | 溝20–45 cm | 400・1,024 |
| second | first | 溝40–70 cm | 400・1,024 |
| stage3-continued | second | 溝40–70 cm | 400・1,024 |
| stage3-exposure | second | 溝55–90 cm | 400・1,024 |
| stage4-standard | stage3-exposure | 溝55–90 cm、標準報酬 | 400・1,024 |
| stage4-balance | stage3-exposure | 同じ地形、複合報酬 | 400・1,024 |

各段階の乱数seedは42です。継続学習ではactor・criticを読み込み、optimizerと更新回数は引き継ぎません。独立学習seedを変えた比較ではありません。`stage3-shaped`と`stage4-sequence`は補足探索を再実行する条件です。

## 評価と動画

```bash
uv run python scripts/evaluate.py \
  --checkpoint runs/stage4-standard/final.pt --gap-cm 85 \
  --output runs/evaluation-85-standard --view readable
uv run python scripts/evaluate.py \
  --checkpoint runs/stage4-balance/final.pt --gap-cm 85 \
  --output runs/evaluation-85-balance --view readable
```

溝幅75・80・85・90 cmを指定できます。各条件で開始位置を基準、15 cm手前、15 cm奥に変えて評価します。開始直後の1秒は静止指令、その後は前進速度1 m/sを指令します。描画は50 fps、等速です。`--view original`は元の描画、`--view side`は側面視点です。描画環境による非決定性があるため、新規推論は元の収録と別の試行として扱ってください。

成功は対岸の所定範囲内で姿勢・接触条件を1秒間満たすことです。各足がその間に一度以上接触し、80%以上のフレームで少なくとも1足が支持する条件を含みます。4足の同時接地を常に要求する条件ではありません。転落、側方逸脱、時間内未完了は失敗です。失敗試行でもJSONを残し、次の開始位置へ進みます。起動失敗などJSONが残らない場合は処理を止めます。

## CPU検査

```bash
uv sync --locked --link-mode copy
uv run python scripts/check_actions_policy.py
uv run python -m compileall -q scripts
(cd scripts && uv run --project .. python -m unittest test_course_suite test_terrain_geometry test_recipe)
```

補足の接触状態機械テストはシミュレータ環境のPyTorchを使用します。

```bash
(cd scripts && ../.local/sim/bin/python -m unittest test_jump_sequence)
```

CPU検査は地形生成、成功判定、学習条件の依存関係を対象とします。GPUでの学習完了や第三者環境での数値再現を保証するものではありません。変更点と出典は[コードの来歴](docs/provenance.md)、第三者の権利関係は[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)を参照してください。
