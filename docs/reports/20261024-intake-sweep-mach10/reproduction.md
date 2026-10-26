# 再現手順

この記録には、レビューした図、コンパクトなCSV、実行したcfgの写しだけを含める。格子・保存場・ログは含めない。

## 実行

設定は[scripts/examples/intake-sweep-ramp4.json](../../../scripts/examples/intake-sweep-ramp4.json)と[intake-sweep-ramp2.json](../../../scripts/examples/intake-sweep-ramp2.json)。今回は `execution` だけを次のように変えた（それ以外は同一）。

| 項目 | 値 |
|---|---|
| `backend` | `local`（PBSを使わず、その計算機で順に実行） |
| `ranks` | 12（Mac mini の既定値、`scripts/mpi_defaults.py`） |
| `mpirun` | `mpirun --bind-to none -np {ranks}` |
| `su2_cfd`、`python`、`output_root` | 計算機のパス（共有しない） |

```sh
python3 scripts/intake_sweep.py prepare <設定>.json
python3 <run>/scripts/intake_sweep.py submit <run> --backend local
python3 <run>/scripts/intake_sweep.py compare <2段のrun> <4段のrun> --output <出力先>
```

実行には、このリポジトリで公開する前の版（非公開リポジトリのコミット `8675ab6`）のスクリプトを使った（各runの`scripts/`に写したもののハッシュは[manifest.json](manifest.json)の`source_runs.*.scripts`）。図はその後、`theory-diff.png`の凡例の位置だけを直したスクリプトで集計し直した。値は変わらない。

公開版の `intake_sweep.py` ほかは、その後に (1) 境界条件図と格子図の追加、(2) 出力形式の既定値の変更（2026年10月26日、[掃引の説明](../../intake-euler/intake-sweep-pbs.md)の `output` の行）と実行後の整理（`sweep_retention.py`）を加えたものである。`intake_euler.py` にはバイナリ restart の読み込みを足しただけで、`intake_sweep_theory.py` と形状・格子を作るスクリプトは同一である。当時と同じ出力を得るには、設定の `output` を `{"restart_ascii": true, "restart_compact": false, "checkpoint_volume": true, "adopted_volume": false}` にする。

## 出典

- 表と図は各runの`analysis/`（`aggregate`の出力）から写した。
- `configs/<形状>/<case>/<段>.cfg`は実行したcfgそのもの。`MESH_FILENAME= mesh.su2`は各段のフォルダに置いた格子へのリンク、`SOLUTION_FILENAME= seed_restart`は初期解をその段へ写したもの（SU2 8.5.0のパス長の問題を避けるため）。
- 格子、run.json、diagnosis.md、ログは計算機のパスを含むため載せていない。
