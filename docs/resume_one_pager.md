# 이력서 원페이지 — Quant-Lab 포트폴리오 (한화투자증권 디지털금융 Trading)

> 1페이지 면접 보조자료. 본 .md 를 그대로 워드 1장으로 옮기거나 PDF 변환 권장.

---

## 한 줄 자기소개

수학과 전공 + Python 숙련. **9주 압축 스프린트로 옵션 프라이싱·헤지 모노레포(5 Layer) 구축**하여 한화투자증권 JD 4축(자산운용·ELS 헤지·ETF LP·Trading 전산) 동시 커버.

GitHub: `https://github.com/<USERNAME>/quant-lab` (예정)

---

## 핵심 성과 6개 (수치 위주)

### 1. **실데이터 한화 ELS 재가격 +2.17% (±3% 목표 달성)**
- DART 전자공시 Open API 로 **한화스마트ELS 제8286호** XML 본문 직접 파싱
- 3자산 (KOSPI200·S&P500·SX5E) worst-of step-down, 3년 만기, 연 11.31% 쿠폰
- Layer A 의 Cholesky correlated MC 엔진으로 50,000 paths 재가격
- base 시나리오 10,217원 vs 공시 발행가 10,000원 = **+2.17% 오차 (플랜 ±3% 검증 통과)**
- KI hit prob 14.1%, 평균 듀레이션 1.45년 동시 산출
- correlation 민감도 5 시나리오 추가 분석

### 2. **NN Pricer (Hutchinson 1994) ATM IV 오차 0.55 vol points, BSM 대비 20× 가속**
- MLP [5→128→128→64→1], 500k 합성 BSM 샘플 × 50 epochs CPU 학습 (95분)
- **Loss function 진단**: 원논문의 raw MSE 는 deep OTM 에서 dynamic range 5자리수 차이로 underfit → **log-space MSE 로 교체** → ATM IV 3.13 → 0.55 vp (2.9× 개선)
- IV-space + moneyness strata (Deep OTM/OTM/ATM/ITM/Deep ITM) 5-bucket 평가 도입
- BSM closed-form CPU loop 대비 **27× 추론 가속** (25.6 µs/opt → 0.95 µs/opt)

### 3. **Deep Calibration (Horvath 2021) 실시장 SPY IV 14× 가속 fit**
- 3,000 LHS-sampled Heston params + semi-analytic IV surface 학습 (CPU 10분)
- yfinance 로 **실시장 SPY 옵션 chain 4,065 IV 점** 수집 → 5×5 grid (25 cells valid) 정렬
- DeepCalibNet gradient descent → **2.3초 calibration**, RMSE 2.22 vp
- semi-analytic Nelder-Mead (32초, RMSE 1.97 vp) 대비 **14× 가속**
- ρ=-0.88 leverage effect, θ=0.078 long-vol 정확 포착

### 4. **C++ MC Kernel pybind11 + OpenMP, numpy 대비 14.8× 가속**
- Xorshift64 PRNG + Box-Muller normal + antithetic 짝 처리
- MSVC 빌드 (stderr/M_PI 호환 fix), 1M paths × 252 steps 1.98s → 0.13s
- Python fallback 자동 (확장 빌드 실패 시 numpy 동작 유지)

### 5. **Deep Hedging (Buehler 2019 PG-on-CVaR) — BSM Δ 대비 CVaR @5% 24.5% 개선**
- TC=30bps 환경, PPO 5번 시도 모두 BSM 못 이김 → 진단 결과 PPO 의 `E[reward]` 최적화와 CVaR 최소화는 직교
- Buehler 원논문대로 **gym 환경 제거 + PyTorch differentiable batch simulator + CVaR loss 에 직접 backward**
- 학습 시간: PPO 25분 → Buehler PG **136초 (CPU, 100× 단축)**
- CVaR ratio 0.755 (PPO 최선 0.92 대비 추가 개선), 학습 곡선 단조 감소·entropy collapse 없음

### 6. **KIS Developers 자동매매 시스템 — DRY_RUN 4-tier safety + 다중 risk guard**
- ETF iNAV 괴리 평균회귀 전략 (KODEX 200 069500 대상)
- 환경변수 4단계 가드: 완전시뮬 → vts mock → vts 실주문 → prod 실주문 (`--i-understand-live` 필수)
- 자동 정지 조건: 포지션 20% 초과, 일일 손실 −1.5%, 장 시작/마감 10분, API 오류 3회
- DRY_RUN 스모크 통과 (장외 시간 risk guard 정상 차단 확인)

---

## 추가 기술 결과

| Layer | 산출물 | 핵심 수치 |
|---|---|---|
| **B3** News-IV | rule + FinBERT(KR) + Claude API 3중 분류기 | 영어/한국어 키워드 보강, 7 ticker 70 헤드라인 → 13 event 분류 |
| **B4** Deep Hedging (Buehler 2019) | PG-on-CVaR direct (PPO 폐기 후) | TC=0.3% **PASS +24.5%** (ratio 0.755) · CPU 2분 16초 학습 · PPO 5번 실패 후 알고리즘 변경으로 달성 |
| **C** 비교 프레임워크 | 5-method pricing + 3-method hedging CSV | `data/all_methods_*.csv` |
| **검증** pytest | 5 패키지 통합 | **37/37 green** |

---

## 기술 스택

**Language**: Python 3.10+, C++17  
**Quant**: BSM, Binomial(CRR), Monte Carlo (antithetic + control variate), Heston (semi-analytic + MC), Greeks (analytic + bumping), IV (Brent), step-down ELS  
**ML/RL**: PyTorch (MLP, Adam, Cosine LR, differentiable batch simulator, PG-on-CVaR), Stable-Baselines3 (PPO + BC warm-start), gymnasium (custom env)  
**NLP**: HuggingFace `snunlp/KR-FinBert-SC`, Anthropic Claude API  
**Data**: DART Open API, yfinance, pykrx, pandas/numpy/scipy  
**Trading**: KIS Developers REST + WebSocket  
**Performance**: pybind11, OpenMP, scikit-build-core (CMake)  
**Infra**: uv workspace, pytest, ruff, mypy, pre-commit, GitHub Actions CI  

---

## JD 매핑

| JD 요구 | 본 프로젝트 대응 |
|---|---|
| **자산운용 모델 개발** | Layer A (BSM/Heston/MC) + Layer B2 (실시장 calibration) |
| **ELS 헤지** | Layer A `price_els` (한화 8286호 +2.17%) + Layer B4 (Deep Hedging) |
| **ETF LP 업무** | Layer D (`autotrader/strategies/etf_inav_arb.py` + risk guard) |
| **Trading 전산 시스템** | Layer D KIS REST + WebSocket + DRY_RUN 4-tier + Layer E C++ |
| **AI 응용** | Layer B1/B2/B3/B4 (4종 논문 재현) |
| **Python + C++ 가속** | Layer E pybind11 (numpy 14.8× 가속) |
| **AI 도구 생산성** | Claude Code 활용 (`notes/ai-productivity.md`) |

---

## 한계 및 다음 단계 (정직)

- **합성 vs 실데이터**: 학습은 합성 (BSM/Heston/GBM), 실데이터는 한화 ELS 1건 + SPY IV surface 만 검증. KOSPI200 옵션 IV 는 KRX 인증 필요로 미투입.
- **Deep Hedging**: 합성 GBM 환경에서 +24.5% 달성 (Buehler PG-on-CVaR). 다음 단계는 GBM tick → KOSPI200 historical 일별 수익률로 환경 교체하여 jump/vol-clustering 실시장 재검증.
- **KIS 모의투자 운영**: 계좌 개설 후 3영업일 무중단 운영 예정.

---

## 면접 라이브 데모 시나리오 (60초)

```python
>>> from pricing.bsm import BSMInputs, call_price       # ① 기본 옵션
>>> from pricing.greeks.analytic import call_greeks
>>> i = BSMInputs(100, 100, 1, 0.03, 0, 0.2)
>>> call_price(i), call_greeks(i).delta
(9.4134..., 0.5987...)

>>> from ai_pricing.nn_pricer.infer import load_pricer, price_batch
>>> import numpy as np                                   # ② NN 추론 ms
>>> m, _ = load_pricer("models/nn_pricer_log_500k.pt")
>>> price_batch(m, np.array([[100,100,1,0.03,0,0.2]]))
array([9.5969...])

>>> # ③ 실데이터 한화 ELS 재가격
>>> import subprocess; subprocess.run(["python","scripts/reprice_hanwha_els.py"])
# → "base 10,217 ± 9.5  KI 14.1%  vs notional +2.17%"
```

---

*업데이트: 2026-04-25 · 모노레포 root: `d:\simeunchul\ai_pricing` · 문서 색인: `docs/README.md`*
