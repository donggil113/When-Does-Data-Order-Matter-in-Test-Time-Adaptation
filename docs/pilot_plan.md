# 파일럿 계획 (작성만 했고 실행하지 않음)

## A. Toy pilot: CPU, 다운로드 없음, 바로 실행 가능

두 config는 같은 world, 같은 test multiset, 같은 seed, 같은 arm, 같은 판정 규칙을 쓴다. 차이는 순서 family뿐이다.

| config | 순서 family | canonical config sha256 (앞 16자) |
|---|---|---|
| `configs/pilot_toy_stationary.json` | `uniform_permutation` (stationary mixture) | `cf60d8d67687a2eb` |
| `configs/pilot_toy_drift.json` | `domain_blocked` (genuine drift: 도메인 블록 순서) | `cf3ed83fcf9826ab` |

실행 명령은 다음과 같다. **승인 후에만 실행한다.**

```sh
PYTHONPATH=src python3 -m ordertta.run --config configs/pilot_toy_stationary.json --run-id pilot_toy_stationary_<commit>
PYTHONPATH=src python3 -m ordertta.run --config configs/pilot_toy_drift.json      --run-id pilot_toy_drift_<commit>
PYTHONPATH=src python3 -m ordertta.compare --a runs/pilot_toy_stationary_<commit>/summary.json \
                                           --b runs/pilot_toy_drift_<commit>/summary.json > runs/pilot_toy_compare_<commit>.json
```

규모와 비용:

- dry-run 기준으로 replicate당 test-time 적응 step은 3,840이다(8개 적응 arm × 16개 순서 × 30 step).
- 여기에 tuning(4개 arm × 6개 lr × 4개 순서 × 8 step)과 calibration(최대 40회 bisection × 4개 순서)이 더해진다.
- 실행시간은 **측정하지 않았다.** smoke run의 step 속도로 외삽하면 config당 수 분 수준으로 보이지만, 확인되지 않은 추정이다.
- 자원 상한은 wall-clock 3,600초(run 전체, 모든 phase)와 test-time gradient 평가 400,000회(replicate당)이다. 상한은 실지출이 아니며, 실지출은 `cost_ledger`에 기록된다.

사전에 고정한 것(`preregistration` 블록):

- primary metric: terminal common-holdout error
- spread 지표: 순서 간 표준편차
- MIE와 tolerance
- replicate seed, 순서 seed, tuning 순서 seed, calibration 순서 seed. 전부 실행하고 보고하며, 어떤 seed도 제외하지 않는다.
- lr grid: {0.05, 0.1, 0.2, 0.5, 1.0, 2.0}. 네 tuned arm이 같은 예산을 쓴다.
- 고정값: λ = 1.0, k_max = 4, small-lr factor {0.3, 0.1}
- split 정의와 해석의 한계

보고할 항목:

- 평균 error
- 순서별 분산, 범위, q90
- **관측한** 최악 순서 (모든 순서에 대한 보장이 아님)
- no-adapt 대비 gain과 최악 gain
- prequential online error
- 순서 간 예측 불일치율과 TV 거리
- 도메인별 error
- phase별 실제 비용
- 판정 라벨. `INCOMPLETE`도 그대로 보고한다.

해석 규칙: 작은 step이나 적응 중단만으로 안정화가 설명되면 "추가 효용 없음"으로 기록한다. 판정 라벨로는 `NO_ADDED_UTILITY_GAIN_BELOW_MIE` 또는 `NO_ADDED_UTILITY_VS_CONTROLS`에 해당한다.

## B. 실데이터 pilot: BLOCKED, 설계 초안

예정 구성:

- 데이터: CIFAR-10-C, severity 5, 15개 corruption
- 모델: RobustBench `Standard` WRN-28-10
- Tent BN affine 적응
- stationary mixture(corruption 혼합을 균일 셔플)와 genuine drift(corruption 블록 순서)를 분리한다. 각각 고정된 여러 순서를 쓴다(예: 16개).
- 같은 대조군 세트를 쓰고, terminal common holdout은 corruption별로 떼어 둔 이미지로 구성한다.

차단 요인:

| 항목 | 상태 |
|---|---|
| numpy, torch, torchvision | 미설치. 지시에 따라 설치하지 않음 |
| GPU | 없음 (`nvidia-smi` 없음). 이 환경에서는 CPU 전용 |
| CIFAR-10-C 데이터 | 저장소와 로컬 어디에도 없음. 다운로드는 승인 대상 |
| 사전학습 가중치 | 없음. 사용할 특정 RobustBench 가중치의 라이선스를 개별 확인해야 함 |
| 코드와 데이터 사용권 | sub-agent 보고 수준(`docs/prior_art.md` §3). 사람 검증 필요 |

실데이터 pilot 전에 할 일:

1. 실제 기준선의 작은 실행을 먼저 확보한다. `no_adapt`와 `tent_full`을 1개 corruption, 2개 순서로 돌려 공개 수치 범위와 대조한다.
2. train, development, calibration, test split을 파일 ID 목록으로 고정한다.
3. 위 사전등록 항목을 GPU 시간 상한과 함께 새 config로 고정한 뒤 실행한다.
