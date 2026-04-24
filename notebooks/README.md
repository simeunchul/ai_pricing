# Notebooks

플랜의 8개 노트북 배치. 각 노트북은 해당 주차 산출물.

| # | Notebook | 주차 |
|---|---|---|
| 01 | binomial_intuition.ipynb | W1 |
| 02 | bsm_greeks_surface.ipynb | W2 |
| 03 | els_reprice_hanwha.ipynb | W3 |
| 04 | nn_pricer_vs_bsm.ipynb | W4 |
| 05 | deep_calibration.ipynb | W5 |
| 06 | news_iv_shift.ipynb | W6 |
| 07 | all_methods_comparison.ipynb | W7 |
| 08 | etf_inav_live.ipynb | W8 |

## 실행 순서

각 노트북은 상위 패키지(`pricing`, `ai_pricing` 등)가 editable install 되어 있다고 가정.

```bash
pip install -e packages/pricing -e packages/ai_pricing -e packages/ai_hedging \
            -e packages/experiments -e packages/autotrader
jupyter lab
```

데이터가 필요한 노트북은 먼저 해당 스크립트를 실행해서 `data/` 아래 자료를 채워둘 것.
