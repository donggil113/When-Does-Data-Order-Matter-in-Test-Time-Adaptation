# When Does Data Order Matter in Test-Time Adaptation? (P1)

같은 test sample-ID multiset을 여러 고정 순서로 replay하면서 test-time adaptation(TTA)의 순서 의존성을 측정한다. 여기에 필요한 CPU 계측 도구를 모았고, 부분공간 제한이 small-step과 norm-matched 대조군을 넘어서는 효용이 있는지 **반증 가능하게** 판정하는 코드를 담았다.

- 현재 상태: `STATUS.md`
- 연구 질문, 반증 조건, 정보 접근: `docs/research_question.md`
- 선행연구 대조: `docs/prior_art.md`
- 파일럿 계획: `docs/pilot_plan.md`
- AI 사용 기록: `AI_USAGE.md`
- 원고 Working Draft v0 (제출하지 않음): `paper/main.pdf`, 소스 `paper/`, 상태 `paper/PAPER_STATUS.md`, 빌드 `paper/BUILD.md`

순서 의존성 자체, reset, 2-step 교환자 공식, 저차원 적응은 모두 알려진 결과다. 이 저장소는 이것들을 새로운 기여로 주장하지 않는다.

## 요구사항

Python 3.10 이상, **표준 라이브러리만** 필요하다. numpy, torch, pytest는 필요 없다(설치되어 있지 않은 환경을 기준으로 작성했다). GPU, 다운로드, 유료 API는 쓰지 않는다.

## 실행

```sh
# 전체 검사: 83개 unit test + 모든 config dry-run / stage 검증
sh scripts/check.sh

# smoke run (코드 경로 검증용, 약 3초)
PYTHONPATH=src python3 -m ordertta.run --config configs/cpu_smoke.json --run-id cpu_smoke_<tag>

# 파일럿 config 검증만 (적응 없음)
PYTHONPATH=src python3 -m ordertta.run --config configs/pilot_toy_drift.json --dry-run

# stage 2 진단: 세 조건, 공유 CPU 예산 3,600초, RLIMIT_AS 3 GiB, 1 worker, 1 thread.
# 이미 실행했다. 결과는 runs/stage2_order_penalty_0119a34/, 판정은 STATUS.md 참조
PYTHONPATH=src python3 -m ordertta.stage --stage configs/stage2_order_penalty_diagnostic.json --run-id <id>

# (stage-2 설계로 대체됨, 실행하지 않음) 원래 toy pilot과 family 비교
PYTHONPATH=src python3 -m ordertta.run --config configs/pilot_toy_stationary.json
PYTHONPATH=src python3 -m ordertta.run --config configs/pilot_toy_drift.json
PYTHONPATH=src python3 -m ordertta.compare --a runs/<stationary>/summary.json --b runs/<drift>/summary.json
```

run마다 `runs/<run_id>/` 아래에 세 파일이 생긴다.

| 파일 | 내용 |
|---|---|
| `raw_log.jsonl` | 모든 step, subspace fit, tuning, calibration, terminal 평가, 실패와 NOT_RUN cell |
| `summary.json` | 순서 간 통계(평균, 표준편차, 범위, q90, 관측 최악, gain), output 수준 순서 차이, 판정 라벨, arm별 정보 접근 카드, phase별 cost ledger |
| `manifest.json` | git commit과 dirty 여부, config/data/model/subspace/holdout/order 해시, 환경, wall time, peak RSS와 Python heap |

## 구조

| 경로 | 내용 |
|---|---|
| `src/ordertta/quadratic.py` | plain SGD의 순서 차이와 국소 2차항 η²(H_b g_a − H_a g_b), 다단계 합, output 수준 예측. Adam 같은 optimizer는 명시적으로 거부한다 |
| `src/ordertta/replay.py` | 고정 순서 replay loader. multiset의 중복·누락·미지 ID를 검사하고 한 번만 replay한다. label 없는 batch, LabelVault, 미래 label 접근을 막는 prequential oracle을 포함한다 |
| `src/ordertta/evaluator.py` | terminal common-holdout evaluator(완전 replay 후에만 평가, split 서로소 검사), 순서 간 예측 불일치와 TV |
| `src/ordertta/methods.py` | Tent식 entropy SGD, no-adapt, small-lr, update-norm-matched 대조군(per-step, calibration split에서의 global) |
| `src/ordertta/subspace.py` | `AdaptationSubspace` 최소 API, full/random/gradient-PCA/order-aware fitter, source/meta와 test-time 비용을 분리하는 `CostLedger` |
| `src/ordertta/toy.py` | 합성 shifted-Gaussian world(모든 split이 서로소)와 선형 softmax 모델. γ,β를 적응한다 |
| `src/ordertta/runner.py` | CPU runner, config 검증, tuning, calibration, 실패와 cap 보존, 산출물 기록 |
| `src/ordertta/analysis.py` | 순서 통계, 짝지은 계층 bootstrap(stage 1), seed별 paired 요약과 stage-2 판정 규칙(seed와 order를 합친 CI 없음), order family 비교 |
| `src/ordertta/stage.py` | 여러 조건을 하나의 CPU 예산과 메모리 상한 아래에서 실행하고 판정하는 stage driver |
| `configs/` | `cpu_smoke.json`, 사전등록된 `pilot_toy_stationary.json`과 `pilot_toy_drift.json` |
| `tests/` | unittest 83개 |
| `runs/` | 실제 실행 산출물과 검사 로그 |

## 상태 라벨 규약

- **소프트웨어 상태와 과학 상태를 분리한다.** 예: `TECHNICAL_TEST_PASS` + `SCIENCE_NOT_EVALUATED`.
- **`READY_FOR_PILOT`의 의미.** 코드와 사전등록 config가 준비되었다는 뜻이다. 신규성 인증이 아니다.
- **실패도 결과로 남긴다.** cell마다 `OK`, `FAILED`, `CAP_EXCEEDED`, `NOT_RUN` 중 하나가 기록된다. OK가 아닌 cell은 결과에 남고, 판정은 `INCOMPLETE`가 된다. skip은 PASS가 아니다.
- **관측 최악은 보장이 아니다.** `observed_worst_*`는 **실행한** 순서 가운데 최악일 뿐, 모든 순서에 대한 위험 보장이 아니다.
