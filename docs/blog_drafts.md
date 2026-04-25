# 블로그 4편 초안

> 각 1500~2500자 권장. 벨로그/노션 마크다운 그대로 사용 가능.
> 수치는 실제 실험치 (`docs/*.html` 베이스).

---

## 블로그 1편 — "수학과 출신이 9주 만에 옵션 프라이싱 모노레포 만들기"

### 요지
- 퀀트 지식 0인 수학과 학생이 한화투자증권 디지털금융(Trading) 직무 지원하기 위해 5-Layer 모노레포 구축
- Classical 퀀트 + AI 4종 + 자동매매 + C++ 가속 동시 커버
- "같은 옵션을 5가지 방법으로 가격 매기고 비교"

### 구조 (8문단)

1. **왜 이걸 만들었나** — JD 4축 (자산운용·ELS·ETF LP·Trading 전산)을 한 저장소로
2. **Layer 지도** — A(이론), B1~B4(AI), C(비교), D(자동매매), E(C++) 각 1줄 설명
3. **Layer A: BSM 부터 ELS 까지** — 코드 라인 수보다 검증 매트릭스가 더 중요. Put-Call parity, Heston semi vs MC, ELS 공시 ±3% 검증
4. **Layer B 의 핵심 차별화** — 단순 NN 학습이 아니라 "왜 이 모델이 필요한가" 부터 (Hutchinson 1994 의 motivation: ELS 10만건 실시간 re-pricing)
5. **수치 결과 highlight** — 한화 ELS +2.17%, NN ATM IV 0.55 vp, Heston 14× 가속, C++ 14.8× 가속
6. **합성 vs 실데이터** — 학습은 합성, 검증은 실데이터. 면접 자리에서 이 구분 정직하게
7. **9주 압축의 비결** — Claude Code 사용. 각 레이어 30% 시간 절감, 디버깅과 평가는 본인 몫
8. **다음 단계** — KIS 모의투자 운영, 블로그 시리즈

### 핵심 1줄
> "추상적 이론을 구체적 수치로 옮기는 능력이 양적 트레이딩의 본질이다."

### 코드 인용 (1~2 블록)
```python
from pricing.bsm import BSMInputs, call_price
i = BSMInputs(S=100, K=100, T=1, r=0.03, q=0, sigma=0.2)
print(call_price(i))  # 9.4134
```

```bash
python scripts/reprice_hanwha_els.py
# base price: 10,217.4 ± 9.5  KI 14.1%  E[life] 1.45y  vs notional +2.17%
```

---

## 블로그 2편 — "Hutchinson 1994 NN Pricer 재현, log-loss 로 ATM IV 오차 2.9× 개선"

### 요지
- Hutchinson-Lo-Poggio 1994 원논문 재현
- 단순 MSE loss 가 deep OTM 에서 underfit 일으키는 진단
- log-space MSE 로 교체 → ATM IV 3.13 → 0.55 vol points (2.9×)

### 구조 (10문단)

1. **왜 NN Pricer 인가** — ELS 매일 재가격 시 BSM closed-form 못 쓰는 exotic 들의 MC 비용. NN 으로 ms 추론
2. **원논문 setup 그대로 재현** — MLP [5→128→128→64→1], BSM 합성 데이터 80k samples × 40 epochs CPU
3. **첫 결과의 함정** — val_mse 2.20e-4 (수렴) 이지만 mean rel err **838%** 라는 가짜 수치
4. **838% 의 정체** — Deep OTM 가격 1e-5 인데 분모로 들어가니 폭발. **dynamic range 5자리수 차이**
5. **MSE loss 의 진단** — 가격 큰 ITM 에 attention 쏠림. ATM/OTM 구조 학습 못 함. (SVG 첨부 가능)
6. **3가지 loss 후보 비교** — relative MSE (eps tuning hell), log-MSE, hybrid. log 가 명확한 winner
7. **결과 비교** — ATM IV 3.13 → 1.58 vp (80k), 500k 학습 시 0.55 vp 달성 (CPU 95분)
8. **평가 도구도 같이 바꿈** — IV-space + 5-bucket strata. 업계 표준
9. **20× CPU 가속** — BSM closed-form CPU loop 56µs vs NN 2µs/option
10. **한계** — 학습 도메인 박스 (S/K 0.6~1.4) 밖은 extrapolation 실패. 주기적 재학습 필요

### 키 takeaway
> "loss function 1줄 변경이 데이터 10배 늘리기보다 효과 컸다. 디버깅의 본질은 'metric 자체가 옳은지' 묻는 것."

### 그래프 (docs/nn_pricer_retrain_results.html 인용 가능)
- Before/After bar chart (5 buckets)
- 학습 곡선 overlay (MSE vs log)

---

## 블로그 3편 — "DART API 로 한화 ELS 받아서 재가격, +2.17% 정확도"

### 요지
- 합성 데이터 학습 후, 실데이터 검증 단계
- 한화스마트ELS 제8286호 (3자산, 3년, 연 11.31%)
- 공시 발행가 10,000원 vs 모델가 10,217원 = +2.17% (±3% 목표 달성)

### 구조 (8문단)

1. **합성 학습 한계** — 학습 검증과 실전 검증은 다름. 실시장 옵션 1건이라도 맞춰야
2. **DART 전자공시 Open API 발견** — 무료, 키 1줄, 한국 모든 공시 (실적/M&A/ELS 포함)
3. **API 함정** — `/pdf/download/main.do` 는 세션 기반이라 직접 못 받음. **`document.xml` API** 가 진짜 답
4. **131KB XML 파싱** — 표 구조 복잡. regex strip 후 키 필드 (issue_no, 만기, 쿠폰, 기초자산) 추출. KI/barrier 는 표 분해 실패 → 업계 표준 가정
5. **3자산 step-down ELS 의 구조** — KOSPI200 + S&P500 + SX5E worst-of, 6개월 자동조기상환, KI 50%, 쿠폰 11.31%/yr
6. **MC 50,000 paths × 3-asset Cholesky** — base 10,217 ± 9.5
7. **5 시나리오 민감도** — vol±3%, corr 0.2~0.9. corr 높을수록 worst-of 완화 → price ↑
8. **부수 발견** — KI hit prob 14.1%, **평균 듀레이션 1.45년** (만기 3년 아닌 1.45년 로 모델링해야)

### 면접 1줄
> "공시 발행가 10,000원이 fair value 이력 없는 상태에서, 표준 가정만으로 +2.17% 안에 맞춤. 실무 마진 1~3% 범위와 일치."

### 다음
- 더 많은 한화 공시 검증 표본 늘리기
- KI/barrier XML 파싱 자동화 (HTML parser 도입)
- B2 Deep Calibration 으로 실시장 IV → vol 자동 추정

---

## 블로그 4편 — "Deep Hedging 수렴 실패의 진짜 원인 — PPO 한계 진단"

### 요지
- Buehler 2019 Deep Hedging 재현
- PPO + dense reward shaping 의 BSM attractor 문제 진단
- 4가지 hyperparameter 튜닝 모두 실패 → 알고리즘 자체 한계로 결론
- 정직한 실패 보고가 면접 자리에서 강함

### 구조 (10문단)

1. **Deep Hedging 의 약속** — TC>0 환경에서 BSM Δ 보다 좋아야 함. 이론적으로 가능. 실제로는?
2. **gym env + PPO 첫 시도** — 50k step CPU. TC=0 에서도 BSM 보다 나쁨
3. **Sub-agent 의 v2** — Action 축소 [0,1] + Buehler eq.13 dense shaping + BC warm-start. TC=0 sanity PASS (1.02×), TC=0.3% **PARTIAL 7.8% 개선**
4. **추가 튜닝 3종 (v3/v3b/v3c)** — 더 강한 loss penalty / 더 긴 학습 / 더 약한 shaping. 모두 v2 보다 나쁨
5. **공통 패턴** — 학습 길어지면 std collapse 0.025, BSM 모방으로 수렴
6. **본질 진단** — Buehler shaping `−λ(V_hedge − V_opt)²` 의 minimum 이 정확히 BSM Δ. TC>0 의 진짜 optimum 은 BSM 보다 under-hedge. PPO 가 두 attractor 사이 못 이김
7. **v2 의 7.8% 정체** — budget 부족(25분)으로 BSM 에 끌려가기 전 멈춰서 우연히 나온 결과
8. **알고리즘 변경의 4가지 경로** — CVaR surrogate / SAC + quantile / **Imitation+Residual** / Randomized env
9. **Imitation+Residual 시도** — action 자체를 (BSM_Δ + δ) 로 정의, δ 만 학습. PPO 가 BSM 으로 끌려갈 수 없음
10. **결과** — (residual 학습 결과 수치 — 학습 끝나면 채워야 함)

### 학습 포인트
> "ML 에서 negative result 는 valuable. PPO 가 못 푸는 이유를 정확히 진단할 수 있으면 다음 알고리즘 선택이 명확해진다."

### 면접 모범 답안
"Buehler 2019 를 재현했지만 vanilla PPO+shaping 으로는 BSM attractor 한계 발견. 4번 시도 후 imitation+residual 로 우회 — 이 진단 자체가 회사 입장에서 가치 있는 지식. 공식 제품화하기 전에 SAC+quantile regression 으로 테스트해봐야 함."

---

## 발행 순서 권장

1. **블로그 3편 (한화 ELS 재가격)** 먼저 — 실데이터 + 한화 키워드 = SEO 강력 + 인사담당자에게 가장 직접적
2. **블로그 1편 (전체 모노레포 소개)** — 다른 글들의 hub 역할
3. **블로그 2편 (NN Pricer log-loss 진단)** — 디버깅 사고 깊이 보여주기
4. **블로그 4편 (Deep Hedging 실패)** — negative result + 진단 능력 어필

각 글 끝에 **GitHub 링크 + 다른 블로그 링크**. 블로그 4편이 1개 시리즈로 묶이면 SEO 누적 효과.

---

*업데이트: 2026-04-25 — 실제 발행 시 최신 수치/링크/스크린샷 보강 권장*
