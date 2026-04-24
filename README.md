# quant-lab

**Classical Pricing × AI Pricing(3종) × AI Hedging × Auto-Trading** 단일 모노레포.

## 5-Layer 네비게이션

| Layer | 내용 | 경로 |
|---|---|---|
| **A** Classical Pricing | BSM · Binomial · MC · Greeks · IV · Heston · ELS | [packages/pricing](packages/pricing) |
| **B1** NN Pricer | Hutchinson-Lo-Poggio (1994) 재현 | [packages/ai_pricing/src/ai_pricing/nn_pricer](packages/ai_pricing/src/ai_pricing/nn_pricer) |
| **B2** Deep Calibration | Horvath et al. (2021) Heston 역매핑 | [packages/ai_pricing/src/ai_pricing/deep_calib](packages/ai_pricing/src/ai_pricing/deep_calib) |
| **B3** News/LLM → IV | FinBERT / Claude 이벤트 기반 IV shift | [packages/ai_pricing/src/ai_pricing/news_iv](packages/ai_pricing/src/ai_pricing/news_iv) |
| **B4** Deep Hedging | Buehler et al. (2019) RL 헤지 정책 | [packages/ai_hedging](packages/ai_hedging) |
| **C** Comparison | 5가지 가격/헤지 동일 경로 비교 | [packages/experiments](packages/experiments) |
| **D** Auto-Trading | KIS Developers · ETF iNAV 괴리 | [packages/autotrader](packages/autotrader) |
| **E** C++ MC Core | pybind11 + OpenMP | [packages/fastmc](packages/fastmc) |

## 핵심 차별화

동일 옵션에 대해 **5가지 가격** 을 산출하고 비교:

1. BSM Closed-form (Layer A)
2. **NN Pricer** — 파라미터 → 가격 직접 매핑
3. **Deep Calibration** — IV surface → Heston 파라미터 역매핑
4. **News/LLM-adjusted Price** — 이벤트 기반 IV shift
5. **Deep Hedging implied cost** — 마찰 반영 가격

## 빠른 시작

```bash
# 개발 환경 (uv 권장, pip 도 가능)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e packages/pricing -e packages/ai_pricing -e packages/ai_hedging -e packages/experiments -e packages/autotrader

# Layer A 스모크 테스트
python -c "from pricing.bsm import call_price, BSMInputs; print(call_price(BSMInputs(100,100,1,0.03,0,0.2)))"

# 전체 테스트
pytest packages/
```

## 주차별 태그

- `v0.1-binomial` — Week 1
- `v0.2-classical-core` — Week 2
- `v0.3-els-heston` — Week 3
- `v0.4-nn-pricer` — Week 4
- `v0.5-deep-calib` — Week 5
- `v0.6-news-llm` — Week 6
- `v0.7-deep-hedging-compare` — Week 7
- `v0.8-autotrader` — Week 8
- `v1.0-portfolio-complete` — Week 9

## 참고 문헌

- Hull, *Options, Futures, and Other Derivatives*
- Shreve, *Stochastic Calculus for Finance Vol II*
- Buehler et al. (2019), *Deep Hedging*, QF 19(8)
- Hutchinson, Lo, Poggio (1994), *A Nonparametric Approach to Pricing*
- Horvath, Muguruza, Tomas (2021), *Deep Learning Volatility*
- Araci (2019), *FinBERT*

## License

Educational / Portfolio use.
