# Eulerインテークのマッハ数・迎角掃引（PBSワークステーション用）

`scripts/intake_sweep.py` は、設定JSONを1つ与えると、形状設計から格子、SU2の計算、リップ断面とアイソレータの集計、図の作成までを自動で行う。Claude Code が接続できないワークステーションで、OpenPBS にジョブを投入して使うことを想定している。計算中の停止・やり直し・次の段への進行はすべてスクリプトが決め、モデルは関与しない。

2026年10月23日作成。手法はオフデザイン掃引の記録（10/13、非公開）を踏襲する。

## 使い方

```sh
# 1. 設定を用意する（例をコピーし、podman のイメージ名などを書き換える）
cp scripts/examples/intake-sweep-ramp4.json my-ramp4.json
# 2. 形状・格子・各ケースの設定を作る（CFDは動かさない）
python3 scripts/intake_sweep.py prepare my-ramp4.json
# 3. PBSへ投入する（ケースごとに1ジョブ、最後に集計ジョブ）
python3 runs/ramp4-m10-tr2.5-cr1.3/scripts/intake_sweep.py submit runs/ramp4-m10-tr2.5-cr1.3
# 進み具合
python3 runs/ramp4-m10-tr2.5-cr1.3/scripts/intake_sweep.py status runs/ramp4-m10-tr2.5-cr1.3
# 2段と4段の比較図
python3 scripts/intake_sweep.py compare runs/ramp2-m10-tr2.5-cr1.3 runs/ramp4-m10-tr2.5-cr1.3 --output runs/compare-ramp2-ramp4
```

- **実行ごとにフォルダを分ける**：`prepare` は `<output_root>/<name>/` を新しく作る。同じ名前のフォルダがあれば止まり、上書きしない。2段と4段は `name` が違うので別のフォルダになる。
- **スクリプトの写し**：`prepare` は使うスクリプトを実行フォルダの `scripts/` に写し、ハッシュを `plan.json` に記録する。ジョブはこの写しを実行する。計算中にリポジトリを更新しても、実行中の掃引は変わらない。
- `submit --dry-run` は、qsub せずにジョブスクリプトを表示する。最初に一度確認する。
- `prepare` の後、`analysis/figures/boundary-conditions.png` と `mesh.png` で境界と格子を確認してから投入する。
- 必要な環境：Python 3.9 以上と numpy・matplotlib（ホスト側）、SU2 8.5.0 と mpirun（podman イメージの中、または `container.image` を `null` にしてホスト側）。インターネット接続は不要。

## 設定ファイル

`scripts/examples/intake-sweep-ramp2.json` と `intake-sweep-ramp4.json` が、今回の2段・4段の設定である。

| 項目 | 内容 |
|---|---|
| `name` | 実行フォルダの名前。英数字と `- _ .` |
| `output_root` | 実行フォルダを作る場所。相対パスは設定ファイルの場所から |
| `design` | 設計 Mach、ランプ後の静温比、段数、捕獲高さ、収縮比（`contraction_ratio`）または流路高さ（`duct_gap_height_m`）、流路長（リップから出口まで）、カウル厚さ・先端長さ、上流長さ |
| `sweep` | 計算する Mach と迎角の一覧。全組合せを計算する |
| `mesh.length_scale` | 省略時は捕獲高さ÷0.35 m。前回の fine-quad 格子と同じ相対解像度になる |
| `stages`、`gate` | 反復数・CFL・limiter・受入判定。省略時は前回と同じ値（下表） |
| `output` | 出力の形式。既定は `restart_ascii=false`（restart は DAT だけ）、`restart_compact=true`（座標と保存量だけ）、`checkpoint_volume=false`（チェックポイントごとの VTU なし）、`adopted_volume=true`（解析後に採用点の VTU を 1 個作る）。2026年10月26日より前の形式は 4 つを反転した値。データ容量を抑えるための設定 |
| `execution` | `ranks`（MPI数）、`su2_cfd`、`mpirun`（`{ranks}` が置き換わる）、`python`、`setup`（ジョブ内で先に実行する行。例：`module load ...`）、`container`、`pbs` |
| `execution.container` | `engine`（既定 `podman`）、`image`（イメージ名。`null` ならホストの `su2_cfd` を直接実行）、`options`（`podman run` に足す引数）。次節を参照 |
| `execution.pbs` | `queue`、`resources`（`#PBS -l` の行）、`job_prefix`、`extra_directives`、集計ジョブ用の `analysis_queue`・`analysis_resources` |

## podman イメージの SU2 を使う

ワークステーションでは、podman で作ったイメージの中の SU2 を使う。`execution.container.image` を設定すると、各段の計算と採用点の VTU 作成を次のコマンドで実行する（`submit --dry-run` で実際のコマンドを表示する）。

```sh
podman run --rm --init --name su2-<name>-<case>-<stage>-<pid> -v <実行フォルダ>:<実行フォルダ> -w <段のフォルダ> \
  -e OMP_NUM_THREADS=1 ... <options> <イメージID> mpirun -np 8 SU2_CFD case.cfg
```

- **MPI はイメージの中で動かす**。`mpirun` と `su2_cfd` はイメージの中のコマンド名またはパスで書く（例の設定は `mpirun`、`SU2_CFD`）。1ジョブ1ノード（`select=1`）の前提で、ホストの mpirun からランクごとにコンテナを起動する方式には対応していない。Python の解析はホスト側で動く。
- **実行フォルダは同じパスでマウントする**。格子へのシンボリックリンクと初期解のハードリンクがそのまま使える。実行フォルダの外のファイルは使わない。
- **イメージは ID で固定する**。`prepare` がイメージ名を `podman image inspect` で調べ、ID・タグ・ダイジェスト・podman のバージョンを `plan.json` の `execution.container` に記録する。ジョブはタグではなくこの ID を実行するので、掃引中にイメージを作り直しても計算中の掃引は変わらない。`prepare` の時点でイメージが無ければ止まる。`run-case` は始める前に ID がそのホストにあるか確かめ、無ければ `state.json` を作らずに終わる（未開始のままなので、イメージを戻してから再投入できる）。各段の `run.json` にコマンド、コンテナ名、イメージ ID を残す。
- **停止**：qdel などで SIGTERM が届くと、podman を通して SU2 を止め、`podman rm -f <コンテナ名>` でコンテナを消してから `cancelled` と記録する。各段の終わりにも同じコマンドを実行し、前の段のコンテナが残ったまま次の段を始めない。
- `options` の例：`--shm-size=2g`（podman の既定 64 MB の `/dev/shm` では MPI の共有メモリ通信が足りないことがある）、`--security-opt=label=disable`（SELinux が有効なホストでマウントしたフォルダへ書けるようにする。無効なホストでは影響しない）。

ワークステーションで最初に確かめること：

1. `podman run --rm <イメージ> SU2_CFD` で、SU2 8.5.0 の表示が出る（設定ファイルが無いというエラーで終わってよい）。
2. イメージの中の利用者が root で MPI が Open MPI なら、`mpirun` に `--allow-run-as-root` を足す（または `options` に `-e OMPI_ALLOW_RUN_AS_ROOT=1 -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1`）。rootless podman ではコンテナの root はホストの利用者に対応するので、出力ファイルの所有者はホストの利用者になる。
3. PBS のジョブの中から podman が動く。rootless podman は `XDG_RUNTIME_DIR` が無いと警告や失敗になることがあるので、必要なら `setup` に `export XDG_RUNTIME_DIR=...` を書く。
4. 1ケースだけ投入し（`submit --only <case>`）、計算中に `podman ps` と `top` で8プロセスが動いていること、`qdel` でコンテナが消えることを確かめてから全体を投入する。podman の cgroup の設定によっては、コンテナが PBS ジョブの cgroup の外で動き、PBS の CPU 割り当てや強制終了が効かないことがある。その場合は `options` に `--cgroups=disabled` などを足して確かめる。

今回の形状：設計 Mach 10、ランプ後静温比2.5、捕獲高さ1 m、鋭角カウル、収縮比1.3。掃引は Mach 8〜12（設計±2）×迎角 −4〜+4°（2°刻み）の25条件。

| 形状 | ランプ角 | ランプ長 | リップ開口 | 流路高さ | 格子 | 設計点の理論総圧回復率 |
|---|---|---:|---:|---:|---:|---:|
| 2段 | 6.34、14.43° | 5.27 m | 177.1 mm | 136.2 mm | 168,692節点 | 0.613 |
| 4段 | 3.29、7.01、11.21、15.96° | 7.05 m | 130.4 mm | 100.3 mm | 204,340節点 | 0.839 |

4段は、10/13 の形状（捕獲高さ0.35 m）を1/0.35倍に拡大したもので、ランプ角は同じ。違うのは流路高さで、収縮比が1.141から1.3になった。

Kantrowitzの余裕（許される収縮比÷1.3）は、最も厳しい Mach 8・−4° で両形状とも約1.18倍。理論の直線衝撃波がリップ手前で交差する条件（理論が当てはまらない条件）は、2段で8条件、4段で15条件ある（`analysis/theory.csv`）。

## 計算の流れ（1ケース＝1ジョブ）

1. 一様流から一次精度で1000反復（CFL 1）。失敗したら CFL 0.5 で1回やり直す。
2. 二次精度（HLLC、MUSCL、WLS、Venkatakrishnan K=0.05、最大4万反復）。2000反復ごとの保存点で受入判定を行い、満たしたら止める。
3. 受入にならなければ、次の順に自動でやり直す（二次精度は最大4回まで）。
   - 非物理点・発散・ソルバーの異常終了（`failed`）：同じ初期解から、limiter の固定を10000反復目に遅らせる。それでも失敗したら CFL 0.5 にする。
   - 健全だが受入判定に届かない（`unconverged`）：最後の保存点から、limiter の固定を遅らせて続ける（1回だけ）。
4. 結果の保存点で、リップ断面、5 mm上流面（捕獲高さで拡大）、ランプ・アイソレータ・出口の断面を評価し、ケースの図と `diagnosis.md` を作る。
5. 全ケースのジョブの後（`depend=afterany`）、集計ジョブが表と図を作る。`aggregate` は手動でも何度でも実行できる。

数値スキーム・境界条件・後処理の詳細は[掃引の記録の「解析手法」](../reports/20261024-intake-sweep-mach10/report.md#解析手法)にまとめた。

受入判定（前回と同じ）：

| 項目 | 条件 |
|---|---|
| 健全性 | 二次精度の全反復で非物理点ゼロ、密度・圧力・温度が正 |
| 残差 | 密度・エネルギー残差の直近500反復の最大値が、一次精度の開始時から5桁以上低い |
| 収支 | 開境界の質量・エネルギー収支が0.1%以内 |
| リップ値の安定 | リップ面とその上流面の総圧回復率・静温比・静圧比・Mach・流量の、前の保存点からの変化が0.01%未満 |
| 反復数 | その試行で二次精度6000反復以上 |

始動の確認：リップ上流面に逆流がなく、亜音速の流量が1%未満なら「始動」とする。受入でも始動を確認できなければ、性能値には使わない（`diagnostic`）。

### 安全のための決まり

- 一度始まったケース（`state.json` がある）は、`submit` も `run-case` も再実行を拒否する。結果が分からないケースを自動で再投入しない。
- qdel などでジョブに SIGTERM が届くと、SU2 を止めてから `cancelled` と記録して終わる。
- 確認した上でやり直すときは、`archive-case` で記録を `cases/<case>/archive-<n>/` へ移してから、そのケースだけ再投入する。`running` のまま止まったケースは、qstat でジョブが無いことを確かめてから `--confirm-not-running` を付ける。

```sh
python3 runs/<name>/scripts/intake_sweep.py archive-case runs/<name> m07_a-4
python3 runs/<name>/scripts/intake_sweep.py submit runs/<name> --only m07_a-4 --resubmit
```

- `--backend local`（PBSを使わず、その計算機で順に実行）は試験用。同じ計算機で別の SU2 が動いていれば始めない（`execution.local_exclusive`）。
- 計算中は保存場を削除しない。整理は検討が終わった後に `scripts/sweep_retention.py` で行う。`plan` は何も変更せず対象と容量を表示し、`apply --review-complete` は全対象の sha256 を `sweep-retention.json` に記録してから整理する。
- SU2 8.5.0 は ASCII 初期解のパスが約100文字を超えると異常終了するため、初期解は各段のフォルダへ `seed_restart.dat`（旧形式では `.csv`）という短い名前で置く。ハードリンクにするので容量は増えない（できなければ APFS クローン、最後に複写）。方法は各段の `run.json` の `seed_link` に残す。
- 採用点の VTU は、採用点の DAT から `ITER=0` の SU2_CFD を 1 プロセスで動かして `cases/<case>/volume/flow.vtu` に作る（反復しないことを history で確かめる）。保存量と圧力・温度・Mach などは本計算の VTU とビット単位で同じになる。limiter の列は保存状態から計算し直した値になる。`SU2_SOL` は restart にある列しか書かないので、compact restart からは保存量だけの VTU になる。`volume.json` に入力のハッシュ、コマンド、終了コードを残す。作れなかった場合も記録だけ残し、解析は止めない。

## 出力

### 判定と用途

| outcome | use | 意味 |
|---|---|---|
| `qualified` | `performance` | 受入、かつ始動を確認 |
| `qualified` | `diagnostic` | 受入だが始動を確認できない |
| `unconverged` | `diagnostic` | 非物理点はないが受入判定に届かない。最後の健全な保存点の値を参考として載せる |
| `failed` / `error` | `none` | 使える二次精度解がない |
| `unknown` | `none` | ジョブが途中で止まった（`running` のまま） |

うまくいかなかったケースは、`analysis/failures.csv` と `cases/<case>/diagnosis.md` に、各試行の初期解・limiter・CFL・結果・停止理由・最初の非物理点の反復・残差の低下桁数・リップ値の変化を記録する。

### 図（`analysis/figures/`）

| ファイル | 内容 |
|---|---|
| `lip-vs-mach.png`、`lip-vs-aoa.png` | リップ断面の6量（総圧回復率の質量流束平均とミックスアウト平均、静温比、静圧比、Mach、流量捕獲率）。横軸 Mach（迎角ごとの線）と横軸迎角（Machごとの線） |
| `theory-diff.png` | CFDと逐次斜め衝撃波理論の差（総圧回復率、静温比）。横軸 Mach と横軸迎角。衝撃波がリップ手前で交差する条件は■ |
| `lip-profiles-m<M>.png`、`lip-profiles-a<α>.png` | リップ面の高さ方向の分布（総圧回復率、静温比、Mach） |
| `isolator-vs-mach.png`、`isolator-vs-aoa.png` | アイソレータ出口の総圧回復率・静温比・静圧比・Mach・逆流と亜音速の割合（診断用） |
| `streamwise-m<M>.png`、`streamwise-a<α>.png` | ランプ→リップ→アイソレータの断面平均（Mach、静圧比、静温比、密度比、総圧回復率、流量） |
| `geometry.png` | 形状、設計の衝撃波線、検査面の位置 |
| `boundary-conditions.png` | 境界の種類（超音速流入口、farfield、Euler滑り壁、超音速流出口）とリップ面の位置。`prepare` の時点で作る |
| `mesh.png` | 格子の全体図（8本に1本）と、第1ランプ先端・最終ランプの角・カウル先端の拡大（全格子線）。`prepare` の時点で作る |
| `outcome-map.png` | Mach×迎角の判定マップ |
| `kantrowitz-margin.png` | Mach×迎角の Kantrowitz の余裕 |
| `residuals.png` | 全ケースの残差の履歴 |
| `cases/<case>/plots/` | 各ケースの残差と、リップ面・アイソレータの代表断面の横断分布（9量） |

塗りつぶしの印は性能値、白抜きは診断用、印なしは解なし。

### CSV（`analysis/`）

| ファイル | 内容 |
|---|---|
| `lip-results.csv` | 1ケース1行。判定、用途、始動、試行回数、リップ断面の質量流束平均（総圧回復率、静温比、静圧比、Mach、流れ角）、流量捕獲率、ミックスアウト平均、総圧回復率と静温比の5%・95%値、一様流のまま入る流量の割合、逆流と亜音速の割合、全温誤差、上流面の総圧回復率と静温比、理論値、CFDと理論の差、衝撃波の交差、Kantrowitz の余裕 |
| `lip-profiles.csv` | リップ面の分布（壁からの高さ、総圧回復率、静温比、静圧比、局所Mach、流れ角） |
| `isolator-sections.csv` | アイソレータ断面（最終ランプ終端の下流に、前回を拡大した間隔で並べた I01… と出口 OUT）。x、断面の上下端、流量、質量流束平均（総圧回復率、静温比、静圧比、密度比、Mach）、流れ角、ミックスアウト平均、逆流と亜音速の割合、最小法線Mach |
| `streamwise-sections.csv` | 全断面（ランプ R*、リップ上流 C-pre、リップ C-lip、アイソレータ、出口）の同じ列 |
| `theory.csv` | 理論値（リップのMach、総圧回復率、静温比、静圧比、衝撃波の交差、Kantrowitz） |
| `case-settings.csv` | 1ケース1行の設定と、実行したcfgへの相対パス |
| `execution.csv` | 段ごとに、初期解、limiter、CFL、理由、結果、終了コード、最終反復、最初の非物理点、ソルバー時間、PBSジョブID |
| `failures.csv` | 性能値にならなかったケースと理由 |
| `summary.md`、`manifest.json` | 要約表と、CSV・格子・スクリプトのハッシュ |

値はすべて無次元（一様流の静圧・静温・全圧で割った値）である。定比熱のEulerでは、高度に相当する寸法を変えても無次元の結果は変わらない。

## 評価の定義

- リップ断面：x=リップ先端の鉛直線で、ランプ壁からリップ高さまで。順方向の質量流束で重み付けした平均（5点Gauss）。前回の記録と同じ方法で、10/13の保存場から同じ値を再現することを確認した。
- ランプ断面は、リップの流量と同じ流量を運ぶ壁側の流管で平均する。アイソレータ断面は下壁からカウル内面まで。
- 理論値：ランプ面に沿った逐次一様の斜め衝撃波（負の転向は Prandtl–Meyer 膨張）。迎角は SU2 の符号で、+α で圧縮が弱まる。直線の衝撃波がリップ手前で交差する条件では、理論は当てはまらない。
- Kantrowitz：理論のリップ Mach で、垂直衝撃波の後の等エントロピー収縮が M=1 になる面積比。リップ断面の流れを一様とみなした近似。

## 格子

`intake_structured_mesh.py` の二ブロック構造四角形格子を使う。流れ方向の節点列は、既存格子から取る代わりに規則（`rule_columns`）で作る。ランプ・流路で3 mm、角とリップ付近で2 mm、リップ±10 mmで0.5 mm、上流は徐々に粗くする（いずれも捕獲高さ0.35 m のとき）。`length_scale` で全ての長さ（間隔、壁の第一層、法線方向の帯、角のなめらかさ）を拡大するので、捕獲高さ1 m の形状でも前回と同じ相対解像度になる。前回の形状に適用すると201,414節点（前回の格子は195,130節点）。

## Mac miniでの確認計算（2026-10-23）

同じスクリプトを `--backend local`、12 MPI で実行した。結果の場はローカルの `results/` にあり、共有しない。

**格子の作り方の確認**：10/13 と同じ4段・Mach 10・40 mm 流路の形状を、規則で作った節点列（201,414節点）で計算し、10/13 の記録と比べた。

| case | 総圧回復率（今回 / 10/13） | 差 | 静温比（今回 / 10/13） | 差 |
|---|---|---:|---|---:|
| Mach 10・0° | 0.82912 / 0.82757 | +0.19% | 2.5663 / 2.5735 | −0.28% |
| Mach 10・+2° | 0.81193 / 0.81228 | −0.04% | 2.3217 / 2.3183 | +0.15% |

差は節点列の違いによるもので、fine-quad の既知の格子誤差（細格子との差0.7〜0.9%）より小さい。

**今回の形状（設計 Mach 10）の四隅**（2026-10-24）：2段・4段の Mach 8・12 × 迎角 −4°・+4° の8ケースは、すべて受入、始動も確認した。Kantrowitzの余裕が最も小さい Mach 8・−4° も不始動にならなかった。2段の Mach 12・+4° は、最初の試行で3290反復目（limiter固定の約1800反復後）に非物理点が出て、limiter の固定を10000反復目に遅らせた2回目の試行で、18,000反復で受入になった。

| 形状 | Mach 8・−4° | Mach 8・+4° | Mach 12・−4° | Mach 12・+4° |
|---|---:|---:|---:|---:|
| 2段の総圧回復率（理論） | 0.552（0.568） | 0.784（0.798） | 0.262（0.285、交差） | 0.608（0.578） |
| 4段の総圧回復率（理論） | 0.752（0.774） | 0.879（0.898） | 0.454（0.529、交差） | 0.619（0.749、交差） |

「交差」は、理論の直線衝撃波がリップ手前で交差する条件。二次精度のソルバー時間は1試行140〜310秒（12 MPI）。

**設計 Mach 9 の形状の四隅**（Mach 10 に変更する前の設定。スクリプトの動作確認として残す）：2段・4段（収縮比1.3）の Mach 7・9 × 迎角 −4°・0° の8ケースは、すべて最初の試行で受入、始動も確認した。Kantrowitzの余裕が最も小さい Mach 7・−4° も、不始動にならなかった。

| 形状 | Mach 7・−4° | Mach 7・0° | Mach 9・−4° | Mach 9・0° |
|---|---:|---:|---:|---:|
| 2段の総圧回復率 | 0.592 | 0.739 | 0.424 | 0.609 |
| 4段の総圧回復率 | 0.782 | 0.885 | 0.655 | 0.830 |

設計点（Mach 9・0°）の静温比は2段2.54、4段2.55（設計値2.5）。

**計算時間（観察値。ベンチマークではない）**：二次精度は8,400〜10,300反復で受入になり、ソルバー時間は2段で約130秒、4段で約160〜190秒だった（12 MPI）。ワークステーションの8 MPIでは、これより長くなる見込み。

Mac miniで全50ケースを流した結果は[掃引の記録（2026-10-24）](../reports/20261024-intake-sweep-mach10/report.md)。

## 確定していないこと

格子独立性、自然始動（上流面の判定以外）、粘性・実在気体の影響、物理的妥当性は確認していない。CR 1.3 は Kantrowitz の限界に近いため、低 Mach・負の迎角で不始動になれば、そのケースは `diagnostic` として記録される。
