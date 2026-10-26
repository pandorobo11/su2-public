# SU2 公開ノート

[SU2](https://su2code.github.io/) を使った数値流体解析のスクリプトと、その調査記録のうち公開できるものをまとめる。調査は後から追加する。

## 内容

| 分野 | 説明 | 記録 |
|---|---|---|
| 2次元Eulerインテーク | 混合圧縮型ランプインテークの形状設計・格子・SU2計算・リップ断面とアイソレータの集計を、設定JSON 1つから行うマッハ数・迎角掃引（OpenPBS またはローカル実行）。[使い方](docs/intake-euler/intake-sweep-pbs.md) | [Mach 10 設計の2段・4段ランプ、Mach 8〜12 × 迎角 −4〜+4°（2026-10-24）](docs/reports/20261024-intake-sweep-mach10/report.md) |

## 配置

- `scripts/`：スクリプトと試験。`scripts/examples/` に設定例。
- `docs/<分野>/`：分野ごとの説明。
- `docs/reports/<YYYYMMDD>-<名前>/`：調査ごとの記録（レビューした図、コンパクトなCSV、実行したcfg、再現手順）。格子・保存場・ログは含めない。

## 必要な環境

Python 3.9 以上と `requirements.txt` のパッケージ（numpy、matplotlib。gmsh は掃引には不要）、SU2 8.5.0、MPI。

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest discover -s scripts -p 'test_*.py'
```

試験は擬似ソルバー（`scripts/intake_sweep_fake_su2.py`、CFDではない）で、判定・やり直し・集計・図の作成を通しで確かめる。SU2 は不要。

## 注意

- 結果は記録に書いた条件での観察であり、格子独立性や実機での妥当性は保証しない。
- ライセンスは付けていない。閲覧は自由だが、再利用・再配布の許諾は与えていない。
