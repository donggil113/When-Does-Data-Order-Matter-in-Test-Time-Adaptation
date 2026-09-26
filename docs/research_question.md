# 연구 질문, 반증 조건, 정보 접근, 목적함수 가정 (P1)

작성일 2026-09-26. 이 문서는 파일럿 실행 **전에** 고정한다. 바꿀 경우 날짜와 이유를 적은 새 버전으로 추가하고, 이미 실행된 결과를 보고 수정하지 않는다.

## 1. 고정한 연구 질문

**RQ-A (측정).** 같은 test sample-ID multiset을 K개의 고정 순서로 replay한다. 이때 Tent식 entropy-minimization plain SGD(`tent_full`)의 terminal common-holdout error가 순서마다 얼마나 흩어지는지(표준편차)를 본다. 이 흩어짐이 stationary mixture 순서(`uniform_permutation`)와 genuine drift 순서(`domain_blocked`) 사이에서 최소관심효과(MIE) 이상 다른가?

**RQ-B (반증 가능한 개입 질문).** Adaptation을 order-aware 파라미터 부분공간(`tent_orderaware`)으로 제한한다고 하자. 이것이 아래 세 가지 대조군보다 순서 간 흩어짐을 MIE 이상 더 줄이면서, no-adaptation 대비 adaptation gain을 MIE 이상 유지하는가?

- small learning rate (`tent_full_lr_x0.3`, `tent_full_lr_x0.1`)
- update-norm-matched full-space 업데이트 (`norm_matched_per_step`, `norm_matched_global`)
- 차원을 맞춘 부분공간 (`tent_gradpca`, `tent_random`)

## 2. 반증 조건 (사전등록된 `ordertta.analysis.decide`)

bootstrap은 각 replicate 안에서 순서를 복원추출한다. 통계량은 replicate 평균이다.

| 라벨 | 조건 | RQ-B에 대한 의미 |
|---|---|---|
| `INCOMPLETE` | 필요한 arm 중 하나라도 OK가 아닌 cell이 있음 | 판정 불가 (실패를 보존함) |
| `NO_ADDED_UTILITY_GAIN_BELOW_MIE` | candidate gain < MIE_gain, 또는 CI 하한 ≤ 0 | 반증: 적응을 거의 안 해서 안정적인 것 |
| `NO_STABILIZATION_OBSERVED` | baseline spread − candidate spread < MIE_spread, 또는 CI 하한 ≤ 0 | 반증: 안정화 효과 없음 |
| `NO_ADDED_UTILITY_VS_CONTROLS` | 지배(dominate)하지 못한 대조군이 하나라도 있음 | 반증: 작은 step, 덜 움직임, 차원 축소로 설명됨 |
| `ADDED_UTILITY_OBSERVED_ON_SAMPLED_ORDERS` | 모든 대조군을 지배함 | **관측한 순서와 replicate에 한정.** 모든 순서에 대한 위험 보장이 아님 |

candidate가 대조군 K를 "지배"한다는 것은 다음 둘 중 하나가 성립한다는 뜻이다.

- (spread_K − spread_C ≥ MIE_spread, CI 하한 > 0) 이고 gain_C ≥ gain_K − tol_gain
- (gain_C − gain_K ≥ MIE_gain, CI 하한 > 0) 이고 spread_C ≤ spread_K + tol_spread

RQ-A는 `python -m ordertta.compare`로 판정한다. drift − stationary의 `tent_full` spread 차이가 MIE_spread 미만이거나 CI가 0을 포함하면 "이 규모에서는 순서 family 차이가 관측되지 않음"으로 기록한다.

toy pilot의 사전등록값은 다음과 같다. 값은 `configs/pilot_toy_*.json`의 `preregistration`에 있다.

- MIE: spread 0.005, gain 0.01 (error 비율 단위)
- tolerance: spread 0.0025, gain 0.005
- replicate(world) seed 5개, test 순서 16개

## 3. 비교군의 정보 접근

| arm | test label | stream history | source/meta 자료 | 추가 supervision | 적응 파라미터 | lr 출처 |
|---|---|---|---|---|---|---|
| `no_adapt` | 없음 | 없음 | source 학습 (W,b) | 없음 | 0 | – |
| `tent_full` | 없음 | 한 순서 안에서 누적, reset 없음 | source 학습 | 없음 | γ,β (2d) | dev split에서 tuning |
| `tent_full_lr_x{0.3,0.1}` | 없음 | 동일 | 동일 | 없음 | γ,β | `tent_full` lr × factor |
| `tent_orderaware` | 없음 | 동일 | source 학습 + **meta_train 라벨** (supervised gain 항) | meta 라벨 | γ,β, 유효차원 ≤ k | dev tuning |
| `tent_gradpca` | 없음 | 동일 | meta_train 입력만 (entropy gradient) | 없음 | γ,β, 유효차원 = candidate 차원 | dev tuning |
| `tent_random` | 없음 | 동일 | 없음 (seed만) | 없음 | γ,β, 유효차원 = candidate 차원 | dev tuning |
| `norm_matched_per_step` | 없음 | 동일 | candidate의 **같은 순서 step-norm 궤적**(label 없는 양) | 없음 | γ,β | – (크기는 step마다 덮어씀) |
| `norm_matched_global` | 없음 | 동일 | calibration split에서 candidate의 평균 terminal 변위 | 없음 | γ,β | calibration에서 bisection |

공통 사항:

- tuning은 모든 tuned arm에 같은 lr grid와 같은 dev 순서를 쓴다(같은 예산).
- 선택 기준은 dev_holdout의 label을 읽는다. 이 사실은 cost ledger에 `uses_labels=True`로 기록된다.
- test_stream label은 prequential scorer만 읽는다. 예측을 commit한 뒤에만 읽을 수 있다.
- test_holdout label은 terminal evaluator만 읽는다.
- 실제 계산비용은 `summary.json`의 `cost_ledger`에 phase별로 따로 기록된다. phase는 source_training, meta_training, tuning, calibration, test_time이다. cap은 상한일 뿐 실지출이 아니다.

## 4. 목적함수와 가정

- **적응 목적**: 배치 평균 예측 entropy(Tent). 최적화는 plain SGD, 고정 lr이며 momentum, Adam, weight decay는 없다. 2차항 분석의 범위도 이것뿐이다.
- **order-aware 부분공간**: θ0(source 해)에서 국소적으로 계산한다. meta batch 쌍 (a,b)에 대해 output 수준 2차 순서항 `J P Pᵀ(H_b P Pᵀ g_a − H_a P Pᵀ g_b)`의 크기를 추정한다. 이것과 1차 supervised 유용도 `⟨P g_sup, P g_ent⟩`를 정규화해 greedy로 방향을 고른다.
  - 가정 1: θ0 근방의 2차 전개가 실제 적응 궤적을 대표한다. 테스트에서 확인한 대로 이 전개는 η·(곡률)이 작을 때만 정확하다.
  - 가정 2: meta_train의 simulated shift(M*)가 test shift(T*)와 비슷하다. toy에서는 같은 생성기에서 파라미터만 다르게 뽑았고, 이것이 보장되지는 않는다.
  - 가정 3: 유한차분으로 구한 HVP와 JVP의 오차는 무시할 수 있다.
- **정규화 점수**: score = gain(S)/G_ref − λ·sqrt(order(S)/order(pool)). 여기서 G_ref = Σ_j max(0, gain_j)이다. λ = 1.0과 k_max = 4는 tuning하지 않고 고정했다.

## 5. 이 프로젝트가 기여로 주장하지 않는 것

아래는 `docs/prior_art.md`의 선행연구에 이미 있다.

- 순서 의존성 자체와 temporal correlation의 영향
- reset과 복원
- 2-step 교환자 공식 θ_ab − θ_ba = η²(H_b g_a − H_a g_b) + O(η³)
- 저차원 파라미터 적응

candidate는 알려진 요소를 조합한 것이다(Taylor 순서항 + greedy 부분집합 선택 + 부분공간 제한). 검색한 범위에서 같은 조합을 찾지 못했지만, 그것이 신규성 인증은 아니다.

## 6. 범위 밖 (결론을 확대하지 않음)

Adam 상태, 장기(long-horizon) 안정성과 collapse, BN 통계량의 동역학은 다루지 않는다. `norm="batch"` 옵션은 구현했지만 pilot에서는 쓰지 않는다. 심층망, 실제 corruption 데이터도 범위 밖이다. toy 결과는 실제 TTA에 대한 증거가 아니다.
