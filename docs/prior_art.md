# Prior-art claim table (P1)

조사일은 2026-09-26이다.

- **검증 방식**: 이 세션의 AI sub-agent가 arXiv, CVF, OpenReview에서 PDF를 받아 텍스트를 추출했다(pypdf, 보조 설치 환경). 그다음 관련 절(방법, 실험)을 키워드 검색으로 읽었다.
- **직접 확인**: 메인 에이전트가 arXiv abs 페이지에서 두 편(Sweeney 2026, ASR 2026)의 메타데이터를 한 번 더 확인했다.
- **사람은 아직 검증하지 않았다.** 인용하기 전에 원문을 다시 확인해야 한다(`AI_USAGE.md` 참조).

상태 표기:

- `FULL_TEXT_VERIFIED`: 관련 절의 본문을 읽음. 전 페이지 정독을 뜻하지는 않는다.
- `FULL_TEXT_UNVERIFIED`: 초록과 메타데이터만 읽음.

## 1. 가장 가까운 원논문과 후속연구

| # | 논문 | 저자 | 연도/학회 | ID | 상태 | 이 프로젝트와의 관계 |
|---|---|---|---|---|---|---|
| 1 | Tent: Fully Test-time Adaptation by Entropy Minimization | D. Wang, E. Shelhamer, S. Liu, B. Olshausen, T. Darrell | ICLR 2021 | arXiv 2006.10726 | FULL_TEXT_VERIFIED | BN affine(γ,β)만 적응, batch 통계, online. 순서는 "shuffle 후 방법 간 공유"로 통제한다. toy의 `tent_full`이 이 구조를 흉내 낸다. |
| 2 | Continual Test-Time Domain Adaptation (CoTTA) | Q. Wang, O. Fink, L. Van Gool, D. Dai | CVPR 2022 | arXiv 2203.13591 | FULL_TEXT_VERIFIED | stochastic restore(p=0.01)는 reset과 유사하다. ImageNet-C에서 10개 corruption 순서에 대해 63.0±1.8을 보고한다. 순서 간 분산 보고는 이미 있다. |
| 3 | Efficient Test-Time Model Adaptation without Forgetting (EATA) | S. Niu, J. Wu, Y. Zhang, Y. Chen, S. Zheng, P. Zhao, M. Tan | ICML 2022 | arXiv 2204.02610 | FULL_TEXT_VERIFIED | entropy 기반 sample 선택과 Fisher 정규화를 쓴다. Fisher는 in-distribution test 표본의 pseudo-label로 계산한다. 10개 random sample order에서 std ≤ 0.3이다. |
| 4 | Towards Stable Test-Time Adaptation in Dynamic Wild World (SAR) | S. Niu, J. Wu, Y. Zhang, Z. Wen, Y. Chen, P. Zhao, M. Tan | ICLR 2023 | arXiv 2302.12400 | FULL_TEXT_VERIFIED | 샘플 **재정렬**로 online label shift를 만든다. entropy EMA < 0.2이면 reset한다. SAM을 쓴다. |
| 5 | When and Where to Reset Matters for Long-Term Test-Time Adaptation (**ASR**, Adaptive and Selective Reset) | T. Lim, J.-W. Hwang, K. Lee | ICLR 2026 | arXiv 2603.03796 | FULL_TEXT_VERIFIED (sub-agent); 메타데이터는 메인 에이전트가 재확인 | 언제(예측 집중도 EMA)와 어디서(출력 쪽 layer 비율) reset할지 적응적으로 정한다. 지시문의 "ASR"을 이 논문으로 본 것은 추론이다(경쟁 후보는 발견되지 않음). |
| 6 | RDumb: A simple approach that questions our progress in continual TTA | O. Press, S. Schneider, M. Kümmerer, M. Bethge | NeurIPS 2023 | arXiv 2306.05401 | FULL_TEXT_VERIFIED | T=1000 step마다 주기적으로 reset한다. CCC 벤치마크는 조건마다 noise ordering 3개를 쓴다. |
| 7 | NOTE: Robust Continual TTA Against Temporal Correlation | T. Gong, J. Jeong, T. Kim, Y. Kim, J. Shin, S.-J. Lee | NeurIPS 2022 | arXiv 2208.05117 | FULL_TEXT_VERIFIED | Dirichlet으로 만든 non-i.i.d. stream을 쓴다. IABN과 PBRS를 제안한다. |
| 8 | Robust Test-Time Adaptation in Dynamic Scenarios (RoTTA) | L. Yuan, B. Xie, S. Li | CVPR 2023 | arXiv 2303.13899 | FULL_TEXT_VERIFIED | correlated sampling과 continual shift를 결합한다. CIFAR10/100-C에서 10개 domain order 평균을 보고한다. |
| 9 | On Pitfalls of Test-Time Adaptation (TTAB) | H. Zhao, Y. Liu, A. Alahi, T. Lin | ICML 2023 | arXiv 2306.03536 | FULL_TEXT_VERIFIED | online batch 의존성 때문에 hyperparameter 선택이 어렵다. temporally correlated stream에서 실패한다. |
| 10 | Persistent Test-time Adaptation in Recurring Testing Scenarios (PeTTA) | T.-H. Hoang, D. M. Vo, M. N. Do | NeurIPS 2024 | arXiv 2311.18193 | FULL_TEXT_VERIFIED | recurring TTA와 collapse를 정의하고, reset 빈도를 정하기 어렵다고 논의한다. |
| 11 | On First-Order Meta-Learning Algorithms (Reptile) | A. Nichol, J. Achiam, J. Schulman | arXiv 2018 | arXiv 1803.02999 | FULL_TEXT_VERIFIED | 순차 SGD의 Taylor 전개 g_i = ḡ_i − αH̄_i Σ_{j<i} ḡ_j + O(α²)를 보이고, AvgGradInner 항을 정의한다. |
| 12 | On the Origin of Implicit Regularization in SGD | S. L. Smith, B. Dherin, D. G. T. Barrett, S. De | ICLR 2021 | arXiv 2101.12176 | FULL_TEXT_VERIFIED | ε²Σ_{k<j}∇∇C_j∇C_k 항이 mini-batch 순서에 의존한다고 밝힌다. 순서에 대해 평균하면 implicit regularizer가 된다. |
| 13 | Manipulating SGD with Data Ordering Attacks | I. Shumailov et al. | NeurIPS 2021 | arXiv 2104.09667 | FULL_TEXT_VERIFIED | 순서만 바꿔도 poisoning이나 backdoor가 가능하다. η² 항을 "data order dependent"라고 부른다. |
| 14 | Gradient Matching for Domain Generalization (Fish) | Y. Shi et al. | ICLR 2022 | arXiv 2104.09937 | FULL_TEXT_VERIFIED | 순차 inner-loop의 2차항이 도메인 간 gradient 내적과 정렬됨을 보인다. |
| 15 | The Geometry of Sequential Learning: Lie-Bracket Prediction of Transfer Order | J. Sweeney | ICML 2026 | arXiv 2606.24993 | 핵심 절은 sub-agent가 읽음(FULL_TEXT_VERIFIED 주장). 메인 에이전트는 초록만 확인 → 메인 기준 FULL_TEXT_UNVERIFIED | θ_AB − θ_BA = η²(H_B g_A − H_A g_B) + O(η³)를 명시한다. 이것을 **순서를 고르는 데** 쓰며, 부분공간을 고르는 데 쓰지는 않는다. **C1과 가장 가까운 선행연구.** |
| 16 | MT3: Meta Test-Time Training | A. Bartler et al. | AISTATS 2022 | arXiv 2103.16201 | FULL_TEXT_VERIFIED | MAML과 self-supervised loss를 결합하고, 샘플 단위로 적응한다. |
| 17 | Learning to Generate Gradients for TTA via TTT Layers (MGG) | Q. Deng et al. | AAAI 2025 | arXiv 2412.16901 | FULL_TEXT_UNVERIFIED | 학습된 optimizer 방식의 meta-TTA이다. |
| 18 | Test-time Adaptation for Regression by Subspace Alignment (SSA) | K. Adachi et al. | ICLR 2025 | arXiv 2410.03263 | FULL_TEXT_VERIFIED | **feature** 부분공간(source PCA)을 쓴다. 파라미터 부분공간이 아니다. |
| 19 | Subspace Optimization for Backprop-Free Continual TTA (PACE) | D. Sójka, S. Cygert, M. Masana | ECML PKDD 2026 | arXiv 2603.28678 | FULL_TEXT_UNVERIFIED | Fastfood random 파라미터 부분공간에서 CMA-ES를 돌린다. |
| 20 | The Golden Subspace (CVPR 2026) | G. Lai, D.-W. Zhou, Z. Li, H.-J. Ye | CVPR 2026 | arXiv 2603.21928 | FULL_TEXT_UNVERIFIED | classifier row space로 feature 업데이트를 제한한다. |
| 21 | Back to the Source: Diffusion-Driven Adaptation (DDA) | J. Gao et al. | CVPR 2023 | arXiv 2207.03442 | FULL_TEXT_VERIFIED | Tent가 batch 크기와 순서(shuffle 여부)에 "extremely sensitive"하다고 보고한다. episodic 방법은 영향을 받지 않는다. |
| 22 | Robust Mean Teacher (RMT) | M. Döbler, R. A. Marsden, B. Yang | CVPR 2023 | CVF open access | FULL_TEXT_VERIFIED | ImageNet-C에서 easy→hard와 hard→easy의 오차 차이가 12%이다. |
| 23 | Measuring the Intrinsic Dimension of Objective Landscapes | C. Li et al. | ICLR 2018 | arXiv 1804.08838 | FULL_TEXT_UNVERIFIED (메타데이터만) | 저차원 random 부분공간 학습. |
| 24 | Gradient Descent Happens in a Tiny Subspace | G. Gur-Ari, D. A. Roberts, E. Dyer | arXiv 2018 | arXiv 1812.04754 | FULL_TEXT_UNVERIFIED (메타데이터만) | gradient가 top-Hessian 부분공간에 모인다. |

sub-agent는 다음 문헌을 초록이나 요약 수준에서만 언급했다. 모두 FULL_TEXT_UNVERIFIED이며 결론에 쓰지 않는다.

- OATTA (arXiv 2601.21012)
- DPCore (arXiv 2406.10737)
- Jhawar & Wang (arXiv 2609.11235)
- Piontkovskaia & Nikolenko (arXiv 2607.16821)
- Rukhovich et al. "Commute Your Domains" (arXiv 2501.15556)
- DA-TTA

## 2. 후보 주장과 선행연구

| 주장 | 판정 | 근거 |
|---|---|---|
| C1. plain SGD 2-step: θ_ab − θ_ba = η²(H_b g_a − H_a g_b) + O(η³), quadratic이면 정확 | **KNOWN** | #11, #12, #13, #15. quadratic에서 정확하다는 것은 자명한 따름정리다(Hessian이 상수). 이 repo의 테스트는 **구현 검증**일 뿐 새 결과가 아니다. |
| C2. TTA 결과가 stream 순서, temporal correlation, non-i.i.d. batch에 의존함 | **KNOWN** | #1, #2, #4, #7, #8, #9, #21, #22 |
| C3. 주기적 또는 trigger 기반 reset이 장기/연속 TTA를 안정화함 | **KNOWN** | #2, #4, #5, #6, #10 |
| C4. 저차원 파라미터 부분공간으로 적응을 제한하는 것이 가능함 | **KNOWN** (일반론). 학습된 기준으로 고른 TTA 파라미터 부분공간은 PARTIALLY KNOWN | Tent의 BN affine 자체가 저차원이다. #19, #20, #23, #24 |
| C5. output 수준 2차 순서항을 줄이도록 고른 부분공간이, small-lr와 update-norm-matched 대조군보다 terminal holdout error의 순서 민감도를 더 줄이면서 gain을 유지함 | **NOT FOUND IN SEARCHED SET** (신규성 주장 아님) | 가장 가까운 것은 다음과 같다. #12는 순서 조작(forward-reverse)으로 순서항을 줄이고, #15는 교환자로 순서를 고른다. 순서 분산을 보고한 연구(#2, #3, #8)는 순서항 자체를 겨냥하지 않는다. small-lr와 norm-matched 대조군을 쓴 연구는 찾지 못했다. **검색은 완전하지 않으며**, 특히 2026년 최신 연구가 빠졌을 수 있다. |

## 3. 코드, 데이터, 가중치 사용권 (후속 실데이터 pilot용)

아래 항목은 sub-agent가 LICENSE 파일을 열어 확인했다고 보고한 것이다. **사람 검증 전까지 UNVERIFIED로 취급한다.**

| 대상 | 보고된 라이선스 | 비고 |
|---|---|---|
| Tent 코드 | MIT | |
| CoTTA 코드 | MIT | |
| EATA 코드 | MIT | |
| SAR 코드 | BSD-3 | |
| RDumb/CCC 코드 | MIT | |
| NOTE 코드 | MIT | |
| RoTTA 코드 | MIT | |
| TTAB 코드 | Apache-2.0 | |
| hendrycks/robustness | Apache-2.0 | |
| CIFAR-10-C, CIFAR-100-C, ImageNet-C (Zenodo) | CC-BY-4.0 | ImageNet 원 약관이 함께 적용되는지는 미검증 |
| RobustBench 코드와 대부분의 model zoo 가중치 | MIT | 예외 가중치가 있다(Apache-2.0, BSD-3, CC BY-NC-SA 4.0 등). **사용할 특정 가중치의 라이선스를 개별 확인해야 한다.** |
| PeTTA, ASR, MT3, Fish 코드 | 미확인 | |

이 단계에서는 어떤 코드, 데이터, 가중치, encoder도 내려받지 않았다.
