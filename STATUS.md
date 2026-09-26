# STATUS: P1 "When Does Data Order Matter in Test-Time Adaptation?"

최종 갱신: 2026-09-26. branch는 `claude/data-order-test-time-adaptation-runqs4`이다.

## 요약

| 구분 | 상태 |
|---|---|
| 소프트웨어 | **TECHNICAL_TEST_PASS**: unit test 70/70 통과, skip 0. config dry-run 3/3 통과. smoke run `RUN_COMPLETED` |
| 과학 | **SCIENCE_NOT_EVALUATED**. toy pilot 2개는 `WRITTEN_NOT_RUN`, 실데이터 pilot은 `BLOCKED` |
| 파일럿 준비 | **READY_FOR_PILOT (toy, CPU)**. 코드와 사전등록 config가 준비되었다는 뜻일 뿐이다. 신규성 인증이나 채택 가능성 확인이 아니다 |

## 착수 시점의 저장소

- 커밋이 하나도 없는 빈 저장소였다. README, STATUS, 기존 결정, 코드, 테스트, 데이터, 실행 기록이 모두 없었다.
- 그래서 보존하거나 재실행을 피할 기존 산출물, ARCHIVE_METHOD, 보류 결정이 없었다. 이 문서가 첫 기록이다.

## 이번 단계의 지시 항목별 상태

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
- toy pilot의 실행시간은 측정하지 않았다. 추정뿐이다.
- `norm="batch"`(Tent식 batch 통계) 경로는 gradient와 불변성 테스트만 있다. pilot에는 쓰이지 않는다.

## 다음 단계 (각각 승인이 필요함)

1. toy pilot A/B 실행: CPU, 다운로드 없음. 명령은 `docs/pilot_plan.md`에 있다.
2. 실데이터 pilot 준비: 의존성 설치, CIFAR-10-C와 가중치 다운로드, GPU 확보, 라이선스 사람 검증이 필요하다.
