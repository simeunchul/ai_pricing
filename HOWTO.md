# HOWTO — quant-lab 동작 설명서

> 이 저장소가 **실제로 무엇을 하는지**, **어떻게 쓰는지**, **각 조각이 어떻게 맞물리는지** 를 정리한 문서. 플랜 전체는 `notes/concepts.md`, 구조 네비는 `README.md`.

---

## 1. 한 줄 요약

**같은 옵션에 대해 5가지 방법으로 가격을 매기고 서로 비교하는 실험실**. 더불어 한화투자증권 JD 의 ELS 헤지·ETF LP·Trading 전산을 **1개 저장소** 로 커버.

## 2. 데이터 흐름 (End-to-End)

```
┌────────────────────┐        ┌──────────────────────────┐        ┌──────────────────────┐
│ 시장/뉴스 입력     │        │  Layer A (Classical)     │        │ Layer C (Comparison) │
│                    │        │                          │        │                      │
│ • KOSPI200 옵션    │──────▶│  BSM · Binomial · MC     │──────▶│ 5가지 가격 동일 경로 │
│ • 한화 ELS 공시    │        │  Greeks · IV · Heston    │        │ PnL · Sharpe · CVaR  │
│ • 네이버/DART 뉴스 │        │  ELS step-down           │        │ MaxDD · 추론속도     │
└────────────────────┘        └───────────┬──────────────┘        └──────────┬───────────┘
                                          │                                  │
                              학습 라벨 생성 / 파라미터 공간 샘플링           │
                                          ▼                                  │
                              ┌──────────────────────────────────┐           │
                              │ Layer B (AI Pricing & Hedging)   │           │
                              │                                  │           │
                              │  B1 NN Pricer (Hutchinson 1994)  │───────────┤
                              │  B2 Deep Calib (Horvath 2021)    │───────────┤
                              │  B3 News/LLM → IV shift          │───────────┤
                              │  B4 Deep Hedging (Buehler 2019)  │───────────┘
                              └──────────────────────────────────┘
                                          │
                                          ▼
                              ┌──────────────────────────────────┐
                              │ Layer D (Auto-Trading, KIS)      │
                              │   ETF iNAV 괴리 매매 · 모의투자   │
                              └──────────────────────────────────┘
                                          │
                                          ▼
                              ┌──────────────────────────────────┐
                              │ Layer E (C++ MC Performance)     │
                              │   numpy → C++ OpenMP 20~30×      │
                              └──────────────────────────────────┘
```

## 3. 설치

### 3.1 최소 설치 (Layer A + 기본 테스트만)

```bash
cd d:/simeunchul/ai_pricing
python -m pip install -e packages/pricing
python -m pytest packages/pricing/tests -v
```

### 3.2 풀 설치 (Layer A~D)

```bash
python -m pip install -e packages/pricing -e packages/ai_pricing -e packages/ai_hedging -e packages/experiments -e packages/autotrader
python -m pip install torch stable-baselines3 gymnasium transformers feedparser requests pandas matplotlib
python -m pytest packages/ -v --ignore=packages/fastmc
```

### 3.3 Layer E (C++ MC) - 선택

Windows: **"x64 Native Tools Command Prompt for VS"** 에서:
```cmd
pip install packages/fastmc
```

빌드 실패해도 `fastmc.mc_euro_call` 은 numpy fallback 으로 동작.

## 4. 각 Layer 사용법

### 4.1 Layer A — Classical Pricing

**핵심 API**:

```python
from pricing.bsm import BSMInputs, call_price, put_price
from pricing.binomial import binomial_european, binomial_american
from pricing.mc.engine import mc_price
from pricing.payoffs import call_payoff, asian_arith_call, barrier_up_and_out_call
from pricing.greeks.analytic import call_greeks
from pricing.iv import implied_vol
from pricing.heston import HestonParams, heston_call_semi, heston_mc
from pricing.els.step_down import StepDownELS, price_els
import numpy as np

# 1) BSM
i = BSMInputs(S=100, K=100, T=1, r=0.03, q=0, sigma=0.2)
print(call_price(i))                    # 9.41...

# 2) Binomial
print(binomial_european(i, N=2000, opt="call"))
print(binomial_american(i, N=500, opt="put"))

# 3) MC — 임의 payoff
mc = mc_price(i, call_payoff(100), n_paths=200_000, seed=42)
print(mc.price, mc.stderr, mc.ci95())

# 4) Greeks
g = call_greeks(i)
print(g.delta, g.gamma, g.vega)

# 5) IV roundtrip
price = call_price(i)
iv = implied_vol(price, i, opt="call")
print(iv)                               # 0.200000

# 6) Heston (semi-analytic)
p = HestonParams(kappa=2, theta=0.04, xi=0.3, rho=-0.5, v0=0.04)
print(heston_call_semi(100, 100, 1, 0.03, p))

# 7) ELS (2-asset step-down)
product = StepDownELS(
    S0=np.array([100., 100.]),
    barriers=[0.90, 0.85, 0.85, 0.80, 0.80, 0.75],
    ki_barrier=0.50, coupon_rate=0.03, maturity_years=3.0,
)
res = price_els(product, r=0.03,
                q=np.zeros(2), sigma=np.array([0.25, 0.30]),
                corr=np.array([[1,0.5],[0.5,1]]),
                n_paths=20_000, seed=0)
print(res.price, res.ki_hit_prob, res.expected_life)
```

**검증 기준** (플랜 Part 7): BSM↔Binomial↔MC 상대오차 < 1e-3, IV roundtrip < 1e-6, Heston semi↔MC < 0.3%.

### 4.2 Layer B1 — NN Pricer (Hutchinson 1994)

**목적**: `(S, K, T, r, q, σ) → Price` 매핑을 MLP 로 근사. Exotic/ELS 의 MC 비용(초 단위) 을 추론(ms) 로 낮춤.

**학습** — `--loss` 옵션 중요:

| Loss | 언제 쓰나 | ATM IV 오차 (80k/40ep) |
|---|---|---|
| `mse` | Hutchinson 1994 원본 재현 용도 | 3.13 vp (underfit) |
| `log` <span style="color:#0b8043">★권장</span> | 실전 사용, Deep OTM 편향 해결 | 1.58 vp |
| `hybrid` | log + mse 균형. Deep ITM trade-off 완화 | 1.55 vp |
| `rel` | 실험용 — eps 튜닝 민감 | 불안정, 권장 X |

```bash
# 권장 CPU 소량 (12분): 80k/40ep log
python -m ai_pricing.nn_pricer.train --n 80000 --epochs 40 --loss log \
    --out models/nn_pricer_log.pt --device cpu

# 풀 CPU 학습 (25분, 플랜 목표 도달): ATM IV 0.55 vp
python -m ai_pricing.nn_pricer.train --n 500000 --epochs 50 --batch 4096 \
    --loss log --out models/nn_pricer_log_500k.pt --device cpu

# Hybrid (Deep ITM 완화)
python -m ai_pricing.nn_pricer.train --n 80000 --epochs 40 --loss hybrid \
    --hybrid-alpha 0.5 --out models/nn_pricer_hybrid.pt --device cpu

# 플랜 원본 (비교용)
python -m ai_pricing.nn_pricer.train --n 100000 --epochs 30 --loss mse \
    --out models/nn_pricer_mse.pt --device cpu
```

**평가** (IV-space + moneyness strata):
```bash
python scripts/bench_nn_pricer.py --model models/nn_pricer_log.pt --n 5000 --label "log-MSE"
```

평가 이론 배경 및 비교 결과: [docs/nn_pricer_underfit_analysis.html](docs/nn_pricer_underfit_analysis.html), [docs/nn_pricer_retrain_results.html](docs/nn_pricer_retrain_results.html).

**추론**:
```python
import numpy as np
from ai_pricing.nn_pricer.infer import load_pricer, price_batch

model, dev = load_pricer("models/nn_pricer.pt")
X = np.array([[100, 100, 1.0, 0.03, 0.0, 0.20]])   # (1, 6)
print(price_batch(model, X, device=dev))           # [9.41...]
```

### 4.3 Layer B2 — Deep Calibration (Horvath 2021)

**목적**: 시장 IV surface → Heston 5개 파라미터 역매핑을 **1초 내**. 기존 semi-analytic 최적화 대비 100× 이상.

**학습** (파라미터 공간 → IV surface 순방향 매핑을 NN 으로 사전학습):
```bash
python -m ai_pricing.deep_calib.train --n 5000 --epochs 40 \
    --out models/deep_calib.pt --data data/deep_calib_cache.npz
```

**실시간 캘리브레이션** (시장 IV 관측 → params):
```python
import torch
from pricing.heston import HestonParams
from ai_pricing.deep_calib.surface import iv_surface
from ai_pricing.deep_calib.model import DeepCalibNet, DeepCalibConfig
from ai_pricing.deep_calib.calibrate import calibrate

ckpt = torch.load("models/deep_calib.pt", map_location="cpu")
model = DeepCalibNet(DeepCalibConfig(**ckpt["cfg"]))
model.load_state_dict(ckpt["state_dict"]); model.eval()

# 가상의 시장 IV
true_p = HestonParams(kappa=1.5, theta=0.04, xi=0.4, rho=-0.6, v0=0.05)
iv_market = iv_surface(true_p)

p_fit, rmse = calibrate(iv_market, model,
                        ckpt["x_mean"], ckpt["x_std"],
                        ckpt["y_mean"], ckpt["y_std"])
print(p_fit, f"RMSE={rmse*100:.2f} vol pts")
```

### 4.4 Layer B3 — News/LLM → IV Shift

**목적**: 뉴스/공시에서 이벤트를 분류하고 IV 에 shift 를 반영해 재가격. 시장 IV 가 반응하기 전 선행 조정.

**3가지 classifier 선택 가능**:
| Backend | 비용 | 정확도 | 환경 |
|---|---|---|---|
| `rule` | 0원 | 낮음 | 바로 동작 |
| `finbert` | 0원 | 중 | `pip install transformers torch` (첫 실행 ~400MB 다운) |
| `claude` | 유료(Haiku) | 높음 | `ANTHROPIC_API_KEY` + `pip install anthropic` |

**사용**:
```python
from datetime import datetime
from pricing.bsm import BSMInputs
from ai_pricing.news_iv.fetch import NewsItem
from ai_pricing.news_iv.pipeline import price_with_news

# 실제 뉴스 fetch 대신 수동 주입도 가능
news = [
    NewsItem("test", "005930", "삼성전자 어닝쇼크 실적 부진", "반도체 수요 감소...", "", datetime.now().isoformat())
]
opt = BSMInputs(S=100, K=100, T=0.5, r=0.03, q=0, sigma=0.22)
out = price_with_news(opt, ticker="005930", news=news, classifier="rule")
print(out)
# {'base_price': ..., 'adjusted_price': ..., 'iv_shift': +0.03, 'dominant_event': 'earnings_miss', ...}
```

**IV shift 규칙표** (`ai_pricing/news_iv/iv_shift.py::IV_SHIFT_RULES`):
| event | ΔIV (vol points) |
|---|---|
| earnings_miss | +0.03 |
| earnings_beat | −0.01 |
| regulatory | +0.02 |
| macro_shock | +0.05 |
| mna | +0.04 |
| rating_change | +0.01 |
| neutral | 0.00 |

### 4.5 Layer B4 — Deep Hedging (Buehler 2019)

**목적**: 이산시점·거래비용·불완전시장 조건에서 **BSM Δ 헤지보다 CVaR 를 개선** 하는 헤지 정책을 RL로 학습. 수렴값 = 마찰 포함 실무 가격.

**학습**:
```bash
# TC=0 (sanity check) — BSM Δ 에 수렴해야 함
python -m ai_hedging.agents.ppo_hedger --tc 0 --steps 50000 --out models/ppo_tc0.zip

# TC=0.3% 실전 세팅
python -m ai_hedging.agents.ppo_hedger --tc 0.003 --steps 200000 --out models/ppo_tc03.zip
```

**평가**:
```python
from ai_hedging.env import HedgingEnvConfig
from ai_hedging.agents.ppo_hedger import evaluate_ppo
from ai_hedging.baselines.bsm_delta import BSMDeltaHedger, rebalance_interval

cfg = HedgingEnvConfig(tc_rate=0.003)

# BSM Δ baseline
bsm_res = rebalance_interval(cfg, every=1, hedger=BSMDeltaHedger("call"), n_paths=2000)
ppo_res = evaluate_ppo("models/ppo_tc03.zip", cfg, n_paths=2000)

print(f"BSM Δ   CVaR p05={bsm_res['p05']:+.3f}")
print(f"PPO     CVaR p05={ppo_res['p05']:+.3f}")
```

**Success target**: TC=0.3% 에서 Deep Hedging CVaR@5% 가 BSM Δ 대비 **20% 이상 개선**.

### 4.6 Layer C — Comparison Framework

**5가지 방법을 동일 옵션 패널에서 비교**:
```bash
# BSM + NN Pricer 비교
python -m experiments.compare_pricers --nn-model models/nn_pricer.pt --out data/pricing_comparison.csv

# 5가지 전체 (NN + PPO 학습 후)
python -m experiments.compare_all \
    --nn-model models/nn_pricer.pt \
    --ppo-model models/ppo_tc03.zip \
    --tc 0.003 --n-panel 2000 --n-hedge 1000 \
    --out data
```

**산출 파일**:
- `data/all_methods_pricing.csv` — method별 mean_err, p95_rel_err, inference_ms
- `data/all_methods_hedging.csv` — method별 mean_pnl, std, cvar_5, sharpe, max_dd

**이력서 문장 템플릿**:
> "5가지 프라이싱·헤지 기법을 동일 경로에서 비교, TC 0.3% 환경에서 Deep Hedging 이 BSM Δ 대비 CVaR@5% XX% 개선. 500k 데이터 학습 NN Pricer 는 BSM 대비 상대오차 XX%, 추론 XX ms."

### 4.7 Layer D — Auto-Trader (KIS Developers)

**세이프티 모델 (3단계)**:

| 상태 | 환경변수 | 결과 |
|---|---|---|
| 완전 시뮬 | (없음) 또는 `--dry` | KIS API 호출 X, mock 가격 사용 |
| 모의투자 + DRY | `KIS_ENV=vts` `KIS_DRY_RUN=true` | 시세 조회 O, 주문은 mock |
| 모의투자 + 실주문 | `KIS_ENV=vts` `KIS_DRY_RUN=false` | 모의 계좌에 실제 주문 (가상머니) |
| 실계좌 실주문 | `KIS_ENV=prod` `KIS_DRY_RUN=false` + `--i-understand-live` | **실계좌 실주문** |

**사전 준비**:
1. `docs.koreainvestment.com` 가입 → 앱키/시크릿 발급
2. KIS 모의투자 계좌 개설 (승인 1~2 영업일)
3. `cp .env.example .env` 후 값 입력

**실행**:
```bash
# 1단계: 완전 시뮬 (가상 가격 tick)
python -m autotrader.runner --dry --seconds 60 --log data/runner_log.json

# 2단계: 모의투자 시세 조회 + mock 주문
KIS_APP_KEY=... KIS_APP_SECRET=... KIS_ACCOUNT=... KIS_ENV=vts \
  python -m autotrader.runner --symbol 069500 --seconds 300

# 3단계: 모의투자 실주문 (가상머니)
KIS_DRY_RUN=false python -m autotrader.runner --symbol 069500 --seconds 300
```

**전략**: KODEX 200 (069500) 같은 ETF 의 가격이 구성종목 바스켓에서 계산한 iNAV 와 **30bps 이상 괴리** 될 때 매매. 개인계정은 단방향이지만 "LP 의사결정 모사" 포지셔닝.

**리스크 가드** (자동 적용):
- 포지션 상한 20%
- 일일 손실 −1.5% 초과 시 정지
- 장 시작/마감 10분 매매 금지
- API 오류 3회 연속 → 정지

### 4.8 Layer E — fastmc C++ Kernel

```python
from fastmc import mc_euro_call, _HAS_NATIVE

r = mc_euro_call(S0=100, K=100, T=1.0, r=0.03, q=0.0, sigma=0.2,
                 n_paths=1_000_000, n_steps=252, seed=42, n_threads=8)
print(r.price, r.stderr)
print("Native C++:", _HAS_NATIVE)
```

**벤치마크**:
```bash
python packages/fastmc/benchmarks/bench_mc.py
```

목표 (1M × 252 steps):
- numpy: ~12s (1×)
- numba: ~4s (3×)
- C++ OpenMP 8코어: ~0.45s (~27×)

## 5. 전형적인 워크플로우

### 5.1 플랜 Week-by-Week 실행 순서

```bash
# Week 1
python -m pytest packages/pricing/tests/test_binomial.py -v
git commit -m "feat(pricing): CRR binomial + American/European" && git tag v0.1-binomial

# Week 2
python -m pytest packages/pricing/tests -v
git commit -am "feat(pricing): BSM + Greeks + MC + IV" && git tag v0.2-classical-core

# Week 3
python notebooks/03_els_reprice_hanwha.py
git tag v0.3-els-heston

# Week 4
python -m ai_pricing.nn_pricer.train --n 100000 --epochs 30
python -m experiments.compare_pricers --out data/pricing_w4.csv
git tag v0.4-nn-pricer

# Week 5
python -m ai_pricing.deep_calib.train --n 5000 --epochs 40
git tag v0.5-deep-calib

# Week 6
python notebooks/06_news_iv_shift.py
git tag v0.6-news-llm

# Week 7
python -m ai_hedging.agents.ppo_hedger --tc 0.003 --steps 200000
python -m experiments.compare_all --out data
git tag v0.7-deep-hedging-compare

# Week 8
python -m autotrader.runner --dry --seconds 3600
git tag v0.8-autotrader

# Week 9 (선택)
pip install packages/fastmc
python packages/fastmc/benchmarks/bench_mc.py
git tag v1.0-portfolio-complete
```

### 5.2 일일 워크플로우

```bash
# 코드 수정 → 테스트 → 커밋
python -m pytest packages/ --ignore=packages/fastmc -q
git add -A && git commit -m "..."

# AI 생산성 로그 append
echo "## $(date +%Y-%m-%d)" >> notes/ai-productivity.md
```

## 6. 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| `ModuleNotFoundError: pricing` | `python` 과 `pip` 이 서로 다른 환경 | `python -m pip install -e ...` 사용 |
| pytest `ModuleNotFoundError: tests.test_xxx` | 다중 패키지의 `tests/__init__.py` 충돌 | 이미 해결됨 (root `conftest.py` + `--import-mode=importlib`) |
| fastmc 빌드 실패 | MSVC/CMake 부재 | numpy fallback 자동 사용 (`_HAS_NATIVE=False`) 또는 x64 Native Tools prompt 에서 재시도 |
| `IV 솔버 no sign change` | target price 가 [intrinsic, S] 범위 밖 | 입력 재확인 (arbitrage 위반 가격) |
| Heston semi vs MC 괴리 크게 남 | `n_paths` 부족 / `n_steps` 너무 성김 | `n_paths=100_000`, `n_steps=200` 이상 |
| KIS `EGW00002` 등 인증 에러 | 앱키/시크릿 또는 계좌번호 오류 | `.env` 값 재확인, 모의/실계좌 환경 분리 |

## 7. 파일 구조 빠른 참조

```
ai_pricing/
├── README.md             # 5 Layer 네비
├── HOWTO.md              # 이 문서
├── pyproject.toml        # uv workspace root
├── conftest.py           # pytest multi-package import fix
├── .env.example          # KIS/Claude API 키 템플릿
│
├── packages/
│   ├── pricing/          # Layer A (검증 완료, 16 tests)
│   ├── ai_pricing/       # Layer B1/B2/B3 (7 tests)
│   ├── ai_hedging/       # Layer B4 (4 tests)
│   ├── experiments/      # Layer C (2 tests)
│   ├── autotrader/       # Layer D (6 tests)
│   └── fastmc/           # Layer E (C++ + fallback)
│
├── notebooks/            # 01~08 jupytext format (.py)
├── notes/
│   ├── concepts.md       # 개념 사다리
│   └── ai-productivity.md # 매일 3~5줄
├── data/                 # 입력/중간 산출물 (gitignored)
├── models/               # 학습 체크포인트 (gitignored)
└── .github/workflows/    # CI
```

## 8. 면접 대비 매핑 (Part 6 요약)

각 기술 질문 → 이 저장소의 어떤 폴더/파일을 가리키면 되는지:

| 질문 | 가리킬 파일/폴더 |
|---|---|
| BSM 유도 | `notes/concepts.md` §1.6, `packages/pricing/src/pricing/bsm.py` |
| ELS 헤지 | `packages/pricing/src/pricing/els/step_down.py` + `packages/ai_hedging/` |
| NN 으로 옵션 가격 | `packages/ai_pricing/src/ai_pricing/nn_pricer/` |
| IV surface 캘리브레이션 빠르게 | `packages/ai_pricing/src/ai_pricing/deep_calib/` |
| 뉴스 반응 AI | `packages/ai_pricing/src/ai_pricing/news_iv/` |
| Deep Hedging 이 BSM 보다 낫나 | `packages/ai_hedging/` + `data/all_methods_hedging.csv` |
| ETF LP 경험 | `packages/autotrader/src/autotrader/strategies/etf_inav_arb.py` |
| Trading 전산 안정성 | `packages/autotrader/src/autotrader/risk/limits.py` |
| Python 느린데 어떻게 | `packages/fastmc/` |
| AI 도구 생산성 | `notes/ai-productivity.md` |

## 9. 다음 단계 체크리스트

### 반드시 (Week 1 권장)
- [ ] KIS Developers 가입 + 모의투자 계좌 개설
- [ ] DART 에서 한화 ELS 공시 1~2건 → `data/els_samples/`
- [ ] (선택) Anthropic API 키 발급 → `.env`

### 실제 학습 (CPU 에서 1회 소량)
- [ ] `python -m ai_pricing.nn_pricer.train --n 50000 --epochs 20`
- [ ] `python -m ai_pricing.deep_calib.train --n 2000 --epochs 30`
- [ ] `python -m ai_hedging.agents.ppo_hedger --tc 0 --steps 30000`

### 비교 실험 (Week 7 산출)
- [ ] `python -m experiments.compare_all --out data`
- [ ] 결과 CSV 를 이력서 bullet 으로 변환

### 운영
- [ ] 모의투자 3영업일 무중단 `python -m autotrader.runner`
- [ ] C++ MC 빌드 + 벤치마크 27× 달성
- [ ] 블로그 4편 작성
