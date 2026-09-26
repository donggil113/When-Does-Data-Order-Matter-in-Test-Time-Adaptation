# STATUS: P1 "When Does Data Order Matter in Test-Time Adaptation?"

최종 갱신: 2026-09-26 (stage 2). branch는 `claude/data-order-test-time-adaptation-runqs4`이다.

## 요약

| 구분 | 상태 |
|---|---|
| 소프트웨어 | **TECHNICAL_TEST_PASS**: unit test 83/83 통과, skip 0. config 검증 7/7 통과(stage 파일 1개 포함). stage 2 실행 `STAGE_COMPLETED` (576/576 cell OK) |
| 과학 (stage 2 toy 진단) | **ORDER_PENALTY_ADDED_UTILITY_NOT_SUPPORTED** (이 합성 toy 환경에 한정). `TOY_DIAGNOSTIC_ONLY_NOT_EVIDENCE_ABOUT_REAL_TTA` |
| candidate | **CANDIDATE_PAUSED**: λ, seed, model sweep로 되살리지 않는다. 추가 학습도 보류한다 |
| 실데이터 | `BLOCKED` (아래 차단 요인). 자동 실행하지 않음 |

## Stage 2: 감독정보와 배치 효과를 통제한 제한적 TTA 진단 (commit `0119a34`)

질문: "같은 meta 지도정보와 적응 예산에서 order penalty가 추가 효용을 주는가?"

### 인수 확인

- 인수 기준 `82ed1a8`은 로컬, 원격, working tree(clean)와 일치했다. 이후 커밋은 없었다.
- `runs/cpu_smoke_b7edd9f/`와 기존 negative smoke 기록은 수정하지 않고 보존했다.
- 기존 config 3개(`cpu_smoke`, `pilot_toy_stationary`, `pilot_toy_drift`)는 수정하지 않았다. 원래 pilot config 두 개는 여전히 `WRITTEN_NOT_RUN`이며, 이번 stage-2 설계로 대체되었다.

### 실행 전에 고친 비교 설계

1. **Utility-only 대조군 (`tent_utility_only`)**
   - candidate와 같은 meta batch, meta label, pool, HVP/JVP를 공유한다. 통계는 한 번만 계산하고, 비용도 order_aware 항목에 한 번만 기록한다.
   - 같은 rank(candidate의 실현 rank), 같은 lr grid와 tuning 순서, 같은 step 수를 쓴다.
   - order penalty만 λ = 0이다.
   - 이번 stage에서는 random/PCA 대조군을 빼서 그것과의 차이로 주장하지 않는다.
2. **Primary 순서 비교**
   - batch 구성원을 `partition_seed = 5000`으로 한 번 고정하고, 순서마다 batch 순서만 바꾼다(`batch_permutation`).
   - drift는 단일 도메인 batch를 고정한 뒤 블록 순서와 블록 안 batch 순서만 바꾼다(`domain_blocked_batches`).
   - 샘플을 섞은 뒤 batch를 다시 나누는 조건(`uniform_permutation`)은 secondary로 따로 돌렸고, 판정에는 쓰지 않았다.
3. **상태 복원 검사**
   - arm/order마다 source 가중치와 θ0에서 시작하는지, frozen 파라미터가 불변인지, 전역 RNG를 소비하지 않았는지를 매 run마다 검사한다. 위반하면 FAILED로 기록된다.
   - optimizer 상태는 없다(plain SGD). normalization buffer도 없다(고정 source 통계).
   - 적응 손실은 모든 적응 arm이 entropy로 같다.
   - step 수는 stationary 30, drift 32이다. 조건 안에서는 arm 간에 같다.
4. **Causal step-norm matching (`norm_matched_per_step_causal`)**
   - candidate 복제본을 같은 batch에서 lockstep으로 진행시켜, step t에는 step t의 update norm만 읽는다. 미래 step을 요청하면 `FutureTraceAccessError`가 난다.
   - 복제본의 gradient 비용(순서당 30/32회)은 control의 비용으로 기록했다.
   - 사후 감사에서 복제본 궤적은 candidate와 모든 order에서 일치했다(조건마다 24/24, 합계 72/72).
5. **Calibration 대조군 (`norm_matched_global`)**: calibration은 이제 **label을 전혀 읽지 않는다**. calibration phase의 label 사용은 False로 기록된다.
6. **이론 설명 정리 (`quadratic.py` docstring)**
   - quadratic 두 step에서는 η²(H_b g_a − H_a g_b)가 정확한 순서 차이다.
   - 비-quadratic에서는 O(η³) 잔차가 있고, η·곡률이 작을 때만 유효하다.
   - 둘을 분리해 서술했다. 기존 fixture를 새 이론 기여로 쓰지 않는다.

### 기존 테스트와 코드의 수정 (명시)

- `tests/test_runner.py::test_all_shipped_configs_validate_and_dry_run`과 `scripts/check.sh`
  - 새 stage 파일(조건 목록)을 run config로 dry-run하다 실패했다.
  - 그래서 stage 파일은 조건 config를 통해 검증하도록 바꿨다. 다른 기대값은 바꾸지 않았다.
- 기존 config에도 영향을 주는 동작 변경이 두 가지 있다.
  - norm calibration이 dev_holdout label을 더는 읽지 않는다. displacement만 쓰던 기존 동작의 수치는 바뀌지 않는다.
  - 상태 복원 위반은 예외로 처리된다.
- 기존 smoke run은 재실행하지 않았다.

### 고정한 config (실행 전 커밋 `0119a34`)

| 파일 | 역할 | canonical sha256 앞 16자 |
|---|---|---|
| `configs/stage2_order_penalty_diagnostic.json` | stage: 조건 목록, 예산, 판정 규칙 | (manifest에 기록) |
| `configs/stage2_stationary_fixed_batches.json` | primary stationary | `3dfd753d85b056c8` |
| `configs/stage2_drift_fixed_batches.json` | primary drift | `623939d3cdd14d05` |
| `configs/stage2_stationary_reshuffle.json` | secondary | `ef7f0a2f856a0b4b` |

- **seed**: 사전등록 replicate 목록 [0..4]의 첫 3개 [0, 1, 2]와 order 목록 [1000..1015]의 첫 8개를 썼다. 성능으로 고르지 않았다.
  - 독립 단위는 seed 3개뿐이다. order는 seed 안에 중첩된다.
  - pilot world의 test 데이터는 이번 stage 전에 평가된 적이 없다. smoke와 test는 다른 world 사양을 쓴다.
- **split과 MIE**: split은 기존 pilot과 같다(source_train / meta_train / development / dev_holdout / calibration / test_stream / test_holdout). MIE는 spread 0.005, gain 0.01이고, tolerance는 0.0025와 0.005이다.
- **tuning**: 기존 lr grid {0.05, …, 2.0}를 그대로 쓰고 확대하지 않았다. tuned arm은 tent_full, tent_orderaware, tent_utility_only이다.
  - stationary는 dev mixture terminal error, drift는 dev regime-end error로 lr을 선택한다.
  - λ = 1.0(정규화)과 k_max = 4는 고정이다.
- **자원 상한**: CPU 누적 3,600초(tuning, calibration, 자식 프로세스 포함), RLIMIT_AS 3 GiB, worker 1개, thread 1개.
- **판정 규칙**: `ordertta.analysis.decide_stage`. seed와 order를 합친 CI는 만들지 않는다. seed별 paired 차이와 3개 seed 모두에서의 부호 일관성으로 판정한다.

### 실행

| 항목 | 값 |
|---|---|
| 검사 | `sh scripts/check.sh`: exit 0, 83 tests OK, config 7개 OK. wall 29.6초, CPU 29.3초, peak RSS 27.8 MiB. 로그: `runs/test_logs/check_0119a34.log` |
| 명령 | `PYTHONPATH=src python3 -m ordertta.stage --stage configs/stage2_order_penalty_diagnostic.json --run-id stage2_order_penalty_0119a34` |
| 결과 위치 | `runs/stage2_order_penalty_0119a34/`: `stage_summary.json`, `stage_manifest.json`, 조건별 `{raw_log.jsonl, summary.json, manifest.json}` |
| 환경 | git `0119a34` (dirty=False), Python 3.11.15, stdlib only. 설치와 다운로드 없음(비용 0) |
| 자원 | **CPU 128.9초 / 3,600초**, wall 130.3초, peak RSS 79.7 MiB. RLIMIT_AS 3,221,225,472 B, RLIMIT_CPU 3605/3610초. thread는 시작과 끝 모두 1개 |
| 조건별 CPU | stationary_fixed 41.1초, drift_fixed 47.2초, reshuffle 40.6초 |
| cell | 조건마다 3 seed × 8 order × 8 arm = 192개, 합계 576개가 **전부 OK**. FAILED, CAP_EXCEEDED, NOT_RUN은 0 |
| label 접근 | adapter의 stream label 접근 0회. calibration label 0회. tuning은 dev_holdout label을 사용(기록됨). drift의 regime holdout 읽기는 기록됨 |
| data hash | r0 `c780ca280870…`, r1 `f4515834278c…`, r2 `179f7b5e2fa7…`. 세 조건이 같은 world를 썼다 |

### 결과: stationary, fixed batches (primary; terminal common-holdout error)

값은 seed r0 / r1 / r2 순서다. 순서 간 통계는 **관측한 8개 order**에 대한 것이다.

| arm | 평균 error | 순서 간 std | 관측 최댓값 | no-adapt 대비 gain |
|---|---|---|---|---|
| no_adapt | .2083 / .1817 / .3683 | 0 / 0 / 0 | .2083 / .1817 / .3683 | 0 |
| tent_full | .2325 / .1965 / .3698 | .0084 / .0240 / .0011 | .2400 / .2533 / .3717 | −.0242 / −.0148 / −.0015 |
| tent_full_lr_x0.3 | .2279 / .1794 / .3635 | .0032 / .0038 / .0006 | .2317 / .1867 / .3650 | −.0196 / +.0023 / +.0048 |
| tent_full_lr_x0.1 | .2200 / .1767 / .3646 | .0013 / .0009 / .0008 | .2217 / .1783 / .3650 | −.0117 / +.0050 / +.0038 |
| **tent_orderaware** | .2054 / .2127 / .4425 | .0008 / .0056 / .0121 | .2067 / .2167 / .4583 | +.0029 / −.0310 / −.0742 |
| tent_utility_only | .2054 / .1767 / .4425 | .0008 / .0000 / .0121 | .2067 / .1767 / .4583 | +.0029 / +.0050 / −.0742 |
| norm_matched_per_step_causal | .2083 / .1796 / .3902 | .0000 / .0284 / .0118 | .2083 / .2100 / .4033 | .0000 / +.0021 / −.0219 |
| norm_matched_global | .2087 / .2008 / .3635 | .0008 / .0318 / .0006 | .2100 / .2750 / .3650 | −.0004 / −.0192 / +.0048 |

paired 평균 error 차이(candidate − 대조군, 같은 seed, 같은 order; + 는 candidate가 나쁨):

- utility-only: 0 / +.036 / 0
- lr×0.3: −.0225 / +.0333 / +.0790
- lr×0.1: −.0146 / +.0360 / +.0779
- norm causal: −.0029 / +.0331 / +.0523
- norm global: −.0033 / +.0119 / +.0790
- tent_full: −.0271 / +.0163 / +.0727. 이때 std 차이는 −.0076 / −.0184 / +.0110이다.

### 결과: drift, fixed batches (primary)

regime-end current-regime holdout error (각 블록 끝에서 그 도메인의 holdout; 순서당 4회):

| arm | r0 / r1 / r2 |
|---|---|
| no_adapt | .2083 / .1817 / .3683 |
| tent_full | .2125 / .1850 / .3723 |
| tent_full_lr_x0.3 | .2167 / .1742 / .3687 |
| tent_full_lr_x0.1 | .2152 / .1858 / .3681 |
| tent_orderaware | .2179 / .1919 / .4490 |
| tent_utility_only | .2179 / .1802 / .4490 |
| norm_matched_per_step_causal | .2210 / .1815 / .4004 |
| norm_matched_global | .2067 / .2437 / .3702 |

predict-then-update online error:

| arm | r0 / r1 / r2 |
|---|---|
| no_adapt | .1833 / .1604 / .3583 |
| tent_full | .1961 / .1586 / .3768 |
| tent_full_lr_x0.1 | .1810 / .1570 / .3555 |
| tent_orderaware | .2003 / .1690 / .4406 |
| tent_utility_only | .2003 / .1503 / .4406 |

- mixture terminal holdout error도 기록했다(`drift_fixed_batches/summary.json`). 다만 최신 regime 적응을 벌점 처리하지 않도록 판정에는 쓰지 않았다.
- drift 결과는 stationary 결과와 합치지 않았다.

### 판정: `ORDER_PENALTY_ADDED_UTILITY_NOT_SUPPORTED` (사전등록 규칙)

충족하지 못한 조건:

- no-adapt 대비 gain: +.0029 / −.0310 / −.0742
- tent_full 대비 std 감소: +.0076 / +.0184 / −.0110 (한 seed에서 음수)
- utility-only, small-LR 두 개, norm-matched 두 개 모두를 지배하지 못함
- drift에서 no-adapt보다 나쁨: seed 평균 gain이 regime-end −.0335, online −.0359이다.
  - candidate − utility-only는 regime-end +.0039(tolerance 이내), online +.0063(tolerance 0.005 초과)이다.

### 해석과 한계 (이 toy 환경에 한정)

- **penalty가 부분공간을 바꾼 seed는 r1 하나뿐이다.**
  - r0와 r2에서는 order-aware greedy가 2개 방향에서 조기 종료했고, 같은 rank의 utility-only가 **똑같은 두 방향**을 골랐다(pool 인덱스 [4,3], [3,5]). 두 arm의 결과가 완전히 같다.
  - r1에서는 방향([1,5] 대 [0,2])과 tuned lr(2.0 대 0.05)이 모두 달랐고, candidate가 utility-only보다 평균 3.6%p 나빴다.
  - 따라서 order penalty의 순수 효과에 대한 정보는 사실상 seed 1개에서만 나왔다.
- **이 toy에서는 baseline Tent 자체가 terminal holdout에서 이득이 없다.** tent_full의 stationary gain은 3개 seed 모두 음수다.
  - "gain을 유지하면서 순서 분산을 줄인다"는 질문을 검정하기에 이 환경은 검정력이 낮다.
  - 이 결과는 실제 이미지 TTA에 대한 증거가 아니다.
- **seed 3개의 비유의성이나 비지지는 동등성이 아니다.** CI는 만들지 않았다.
- **tuned lr이 grid 경계에 걸렸다.** candidate의 tuned lr은 여러 seed에서 grid 상한 2.0이었다(stationary r1·r2, drift 전 seed). 지시에 따라 grid는 확대하지 않았다.
- **부분적 class 쏠림.** r2에서 candidate와 utility-only의 terminal 예측 최대 class 비율은 0.78(stationary)과 0.87(drift)이었다. no-adapt는 0.59였다. 완전 collapse(> 0.9)는 없었다.
- **관측한 최악 order는 모든 순서에 대한 보장이 아니다.**
- **secondary (reshuffle) 조건도 같은 방향이었다.** candidate − utility-only는 0 / +.0363 / 0이었다. 판정에는 쓰지 않았다.

### 비용 (stationary_fixed 조건; 다른 조건도 비슷)

| phase | gradient 평가 | 그 밖의 비용 | CPU/wall | label |
|---|---|---|---|---|
| source 학습 | 600 full-batch epoch | – | 10.4초 | label 사용 |
| meta fit | 432 | HVP 192, JVP 24 | 4.7초 | label 사용. utility-only는 선택 비용만 |
| tuning | 1,728 | – | 4.4초 | dev_holdout label 사용 |
| calibration | 1,120 | – | 2.1초 | label 0 |
| test-time | 5,760 | – | 16.1초 | label 0 |

순서당 test-time gradient는 candidate 30회, causal norm control 60회(자신 30 + 복제본 30)이다.

## Stage 1 기록 (보존)

## 착수 시점의 저장소

- 커밋이 하나도 없는 빈 저장소였다. README, STATUS, 기존 결정, 코드, 테스트, 데이터, 실행 기록이 모두 없었다.
- 그래서 보존하거나 재실행을 피할 기존 산출물, ARCHIVE_METHOD, 보류 결정이 없었다. 이 문서가 첫 기록이다.

## Stage 1: 지시 항목별 상태

| # | 항목 | 상태 | 위치 |
|---|---|---|---|
| 1 | smooth quadratic/plain-SGD에서 두 업데이트의 순서 차이와 국소 2차항 비교 | 구현하고 테스트 통과 | `src/ordertta/quadratic.py`, `tests/test_quadratic_order.py` |
| 2 | 같은 sample-ID multiset을 여러 고정 순서로 replay하는 loader, terminal common-holdout evaluator, 중복·누락·미래 label 접근 검사 | 구현하고 테스트 통과 | `replay.py`, `evaluator.py`, `tests/test_replay.py`, `tests/test_toy_and_controls.py` |
| 3 | no-adaptation, small-lr, update-norm-matched(per-step, calibrated-global) 대조군 연결 | 구현하고 runner에 연결, 테스트 통과 | `methods.py`, `runner.py` |
| 4 | output 수준 순서 차이를 줄이는 adaptation-subspace 최소 API. source/meta 자료와 비용을 따로 기록 | 구현하고 테스트 통과 | `subspace.py` (`CostLedger`의 phase 분리) |
| – | 실행 가능한 pilot config | 작성하고 dry-run 통과. **실행하지 않음** | `configs/pilot_toy_{stationary,drift}.json`, `docs/pilot_plan.md` |

## 실행 기록 (전부 저장소 안에 있음)

| run | 명령 | commit | 결과 |
|---|---|---|---|
| 테스트와 dry-run | `sh scripts/check.sh` | `b7edd9f` (clean) | exit 0. 70개 테스트 OK, dry-run 3개 OK. wall 11.9초, child peak RSS 26.5 MiB. 로그: `runs/test_logs/check_b7edd9f.log` |
| smoke | `PYTHONPATH=src python3 -m ordertta.run --config configs/cpu_smoke.json --run-id cpu_smoke_b7edd9f` | `b7edd9f` (dirty=False) | `RUN_COMPLETED`. 8개 arm × 4개 순서가 모두 OK. wall 3.18초, peak RSS 28.1 MiB, Python heap peak 2.77 MiB. 산출물: `runs/cpu_smoke_b7edd9f/{raw_log.jsonl,summary.json,manifest.json}` |

smoke run의 해시:

- config: canonical sha256 `ea198f77c18977cb…`
- data(world): `56abcd7df483…`
- source model: `c6561f5ba43f…`
- holdout: `08b2649b04cf…`

전체 값은 manifest에 있다.

### smoke 수치 (코드 경로 검증용, 과학적 증거 아님)

규모는 test stream 144개, holdout 180개, 순서 4개, replicate 1개이다. 판정 라벨은 `NO_ADDED_UTILITY_GAIN_BELOW_MIE`이다. candidate의 gain이 −0.011로 no-adapt보다 나빴다.

| arm | 평균 error | 순서 간 std | 관측 최악 | gain |
|---|---|---|---|---|
| no_adapt | 0.2222 | 0.0000 | 0.2222 | +0.0000 |
| tent_full (lr 0.5) | 0.2056 | 0.0045 | 0.2111 | +0.0167 |
| tent_full_lr_x0.1 | 0.2222 | 0.0000 | 0.2222 | +0.0000 |
| tent_orderaware (k=2/3) | 0.2333 | 0.0000 | 0.2333 | −0.0111 |
| tent_gradpca (k=2) | 0.2319 | 0.0028 | 0.2333 | −0.0097 |
| tent_random (k=2) | 0.2389 | 0.0000 | 0.2389 | −0.0167 |
| norm_matched_per_step | 0.2208 | 0.0053 | 0.2278 | +0.0014 |
| norm_matched_global | 0.2167 | 0.0000 | 0.2167 | +0.0056 |

해석의 한계:

- 순서가 4개뿐이고 holdout이 180개라 error 해상도가 1/180이다. 극소형 toy이므로 어떤 방향의 결론도 내리지 않는다.
- 이 수치들이 보여 주는 것은 구현이 의도대로 동작한다는 것뿐이다.
  - 대조군이 연결되어 있다.
  - no_adapt는 순서에 대해 정확히 불변이다.
  - 판정 규칙이 "안정적이지만 적응하지 않은" candidate를 걸러낸다.
- adapter의 stream label 접근은 0회였다. prequential scorer만 commit 후에 접근했다.

## 차단 요인 (정확한 내용)

1. **Python 과학 스택이 없다.** `/usr/local/bin/python3` (3.11.15)과 `/usr/bin/python3.1{0,1,2,3}` 어디에도 numpy, torch, torchvision, scipy, pytest가 없다. 지시에 따라 설치하지 않았다. 그래서 전 구현을 표준 라이브러리만으로 작성했고, 테스트는 `unittest`로 돌린다.
2. **GPU가 없다.** `nvidia-smi`가 없다. CPU는 4코어, RAM은 15 GiB이다.
3. **실데이터와 가중치가 없다.** CIFAR-10-C와 RobustBench 가중치가 로컬에 없다. 다운로드는 승인 대상이다.
4. **사용권이 사람 검증 전이다.** 코드, 데이터, 가중치 라이선스는 sub-agent의 보고만 있다(`docs/prior_art.md` §3).

## 실행 전 설계 변경 기록 (pilot을 돌리기 전, 결과를 보고 고친 것 아님)

1. **테스트 격자.** 비-quadratic(quartic) fixture의 잔차 차수 테스트는 처음에 절대 step 격자(η = 0.04 … 0.0025)를 썼고, 곡률이 큰 seed에서 실패했다(추정 차수 2.79 < 2.8).
   - 2차 전개는 η·L(L은 국소 곡률)에 대해 점근적이다. 그래서 격자를 η·L 단위로 바꿨다. 임계값 2.8~3.2는 **그대로** 두었다.
   - 같은 이유로 "η = 0.01에서 2차항이 잔차의 10배 이상"이라는 즉흥 검사도 실패했다. 이를 "η·L = 1e−3에서 상대 잔차 < 0.05"로 바꿨다.
   - "η·L ≈ 0.3에서는 상대 잔차 > 0.1"이라는 범위 확인 테스트를 추가했다. 이는 실제 TTA 학습률에서 2차항 근사가 깨질 수 있다는 점을 기록한다.
2. **greedy 조기 종료.** 첫 smoke 시도는 scratchpad에서 실행해 저장소에 보존하지 않았다. 설정은 λ = 1.0, 비정규화 점수였다.
   - 이때 order-aware fitter가 entropy gradient가 0인 방향만 골랐다(gain ≈ 1e−32).
   - 그래서 "점수를 올리지 못하는 방향은 추가하지 않는다"는 조기 종료를 넣었다. 빈 부분공간이면 "적응하지 않음"으로 그대로 보고한다.
   - smoke config의 λ는 비어 있지 않은 부분공간 코드 경로를 거치도록 0.1로 바꿨다. smoke는 과학적 판정에 쓰지 않는다.
3. **정규화 점수.** 작은 테스트 config에서 meta 데이터의 pool 전체 1차 gain이 음수가 되어 정규화가 깨졌다(ValueError).
   - 그래서 기준값을 "달성 가능한 gain" G_ref = Σ_j max(0, gain_j)로 바꿨다.
   - pilot의 λ = 1.0은 이 정규화 점수 기준이다.
4. **실패 격리.** subspace fit이나 tuning이 실패해도 run 전체가 중단되지 않는다. 그 부분공간이나 lr에 의존하는 arm만 `NOT_RUN`(사유 포함)으로 기록된다. 이 경우 판정은 `INCOMPLETE`이다.
5. **calibration 0 목표.** norm-matched calibration에서 목표 변위가 0이면(candidate가 움직이지 않음) lr = 0을 정확히 일치한 것으로 처리한다. 이전에는 불일치 ∞로 처리해 NOT_RUN이 되었다.
6. **tuning의 label 기록.** lr tuning은 dev_holdout label로 선택하므로 cost ledger에 `uses_labels=True`로 기록하게 고쳤다. 이전에는 False였다.
7. **테스트 자체의 오류 수정.**
   - (a) `Counter(dict)`로 비교하던 버그를 고쳤다.
   - (b) "도움이 되는 방향이 없으면 빈 부분공간" 테스트의 fixture에 실제로는 교환 가능하고 유익한 방향(축 1)이 들어 있었다. 그래서 기대값 자체가 틀렸다.
   - 모든 방향의 gain이 음수인 fixture로 바꿨다. "교환 가능한 방향은 어떤 λ에서도 선택된다"는 테스트도 따로 추가했다.

## 미검증 주장과 열린 문제

- 모든 선행연구 판정(KNOWN / NOT FOUND)은 AI sub-agent의 문헌 읽기에 근거한다. 사람 검증이 필요하다.
- C5가 "검색 범위에서 발견되지 않음"이라는 것은 신규성의 증거가 아니다.
- order-aware 부분공간이 θ0 근방의 국소 2차 근사와 simulated shift(M*) 가정 밖에서도 의미 있는지는 검증하지 않았다.
- stage-1의 "toy pilot 실행시간 미측정"은 stage 2에서 해소되었다. stage 2 전체 CPU는 128.9초였다.
- stage 2 결론은 이 합성 선형 toy, seed 3개, 관측 order 8개에 한정된다.
- `norm="batch"`(Tent식 batch 통계) 경로는 gradient와 불변성 테스트만 있다. pilot에는 쓰이지 않는다.

## 다음 단계 (각각 승인이 필요함)

- 현재 candidate(order-aware subspace, λ = 1)는 stage 2에서 지지되지 않았다. 그래서 **보류**한다. λ, seed, model sweep로 되살리지 않는다.
- 원래 `pilot_toy_{stationary,drift}` config는 stage-2 설계로 대체되었다. 실행하지 않는다.
- 이 toy에서는 baseline Tent도 이득이 없었다. 따라서 순서 의존성 질문(RQ-A)을 계속하려면 적응이 실제로 도움이 되는 환경이 먼저 필요하다. 이것은 설계 결정이며 승인 대상이다.
- 실데이터 pilot은 여전히 차단 상태이며 자동 실행하지 않는다. 의존성 설치, CIFAR-10-C와 가중치, GPU, 라이선스 사람 검증이 필요하다.
