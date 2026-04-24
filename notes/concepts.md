# Concept Ladder (개념 사다리)

> 플랜 Part 1 요약. 면접 질문의 60%는 여기서 결정됨.

## 1.1~1.3 No-Arbitrage · Risk-Neutral · Replicating

- 가격 = 복제 비용. 수요공급 아님.
- Girsanov: drift μ → r 로 바꾼 Q 하에서 Price = E^Q[e^(-rT) Payoff].
- 옵션 = Δ주 + β채권. Δ = 헤지 비율.

## 1.4 Binomial Tree

q = (e^(rΔt) − d) / (u − d). American = Bellman.

## 1.5~1.7 GBM · Ito · BSM PDE · Feynman-Kac

dS = μS dt + σS dW.
PDE: ∂V/∂t + ½σ²S²∂²V/∂S² + rS∂V/∂S − rV = 0.
PDE 해 = Q-기대값.

## 1.8~1.9 MC · Greeks

MC 수렴 O(1/√N). Antithetic / Control / Importance.
Δ, Γ, Vega, Θ, ρ : analytic vs bumping vs pathwise.

## 1.10 ELS Step-down Auto-call

2~3자산, 6M 단위 자동상환. KI 미발생 시 full coupon. 발행사는 dynamic hedge로 마진.

## 1.11 Volatility 모델

IV surface · Dupire local vol · Heston · SABR.

## 1.12 ETF LP / iNAV

|ETF − iNAV|/iNAV 가 매매 시그널 (Avellaneda-Stoikov 간이).

## 1.13 Deep Hedging (Buehler 2019)

min_θ E[Loss(PnL)], Loss ∈ {CVaR, mean-var, exp-utility}. 이산시점·TC·불완전시장을 objective 에 직접 반영.

## 1.14 NN Pricer (Hutchinson-Lo-Poggio 1994)

(S,K,T,r,σ) → Price 매핑. Exotic/ELS re-pricing 실시간(ms). 학습 도메인 밖 extrapolation 실패가 한계.

## 1.15 Deep Calibration (Horvath 2021)

Heston params → IV surface NN 사전 학습. 시장 IV 관측 시 param-space grad desc 로 1초 내 calibration (기존 대비 100×). ELS 매일 가격 고시 실무에 직결.

## 1.16 News/LLM → IV

FinBERT / Claude 로 event 분류 → ΔIV lookup → adjusted price. IV는 역사적, 뉴스는 선행 정보.
