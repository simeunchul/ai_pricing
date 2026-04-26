"""Cross-layer integration wrappers.

각 Layer 가 isolated 검증된 후, production 통합:
  - els_daily_nav: B2 calibrated vol → Layer A ELS 매일 재가격
  - news_aware_hedger: B3 news classifier → B4 hedge wrapper (별도 패키지)
  - market_aware_hedge_env: B2 → B4 환경 σ 동적 update
"""
