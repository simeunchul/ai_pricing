# 자동매매 시스템 — 배포 패키지

주식 (KIS 모의투자) + 코인 (Binance Testnet) 자동매매 봇 + 대시보드.

## 1. 시스템 구성

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  주식 (KIS 모의투자)                                          │
│  ──────────────────                                         │
│   1. Dual 자동매매 Runner   (외국인+기관 동방향 신호)         │
│      → start_kis_runner.bat                                 │
│   2. Dashboard              (실시간 잔고/매도임박도/신호)      │
│      → start_kis_dashboard.bat                              │
│                                                             │
│  코인 (Binance Futures Testnet)                              │
│  ─────────────────────────────                              │
│   3. Crypto Per-Symbol Runner (BTC/ETH/SOL/AVAX/BNB         │
│                                + 종목별 strategy + Trailing) │
│      → start_crypto_runner.bat                              │
│                                                             │
│  공용                                                        │
│  ────                                                       │
│   4. 모두 시작:  start_all.bat                              │
│   5. 모두 정지:  stop_all.bat                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 2. 설치 (한 번만)

### 필수 사전 조건
- Python 3.10+ (Anaconda 권장)
- Windows 10/11

### 설치 단계

```powershell
# 1. 코드 폴더로 이동
cd d:\simeunchul\ai_pricing

# 2. 의존성 설치
.\release\install.bat

# 3. API 키 설정
.\release\setup_env.bat
# 또는 .env.example 파일을 .env 로 복사 후 직접 편집
```

### API 키 발급

| 서비스 | URL | 비용 |
|---|---|---|
| KIS 모의투자 | https://apiportal.koreainvestment.com/ | 무료 |
| Binance Futures Testnet | https://testnet.binancefuture.com | 무료 (가짜 USDT 10,000$) |
| Naver 검색 API (선택) | https://developers.naver.com | 무료 25,000 calls/day |

## 3. 실행 — 한 번에 시작

```powershell
# 모두 한 번에
.\release\start_all.bat

→ KIS dual runner    : 5분마다 외국인+기관 신호 polling
→ KIS dashboard      : http://localhost:8501
→ Crypto runner      : 30분마다 종목별 strategy 폴링
```

## 4. 실행 — 개별 시작

### A. 주식 자동매매 (KIS Dual)

```powershell
.\release\start_kis_runner.bat
```

**전략**:
- 매 5분마다 KIS 외국인+기관 가집계 호출
- 둘 다 ±5% 이상 → 매수/매도 후보
- max_concurrent 7종 / MDD cap 30% / cooldown 10일

**로그**: `data/dual_trading.log` + `data/dual_state.json`

### B. 주식 대시보드

```powershell
.\release\start_kis_dashboard.bat
```

→ 브라우저: http://localhost:8501

표시 내용:
- KIS 실제 잔고 (truth source)
- 보유 종목 + 외인/기관 매도 임박도
- 실시간 dual confirmation 신호 (매수 후보 / 매도 후보)
- Portfolio 시계열 + Drawdown 차트

### C. 코인 자동매매 (Binance Crypto)

```powershell
.\release\start_crypto_runner.bat
```

**전략** (종목별 차별화):
| 종목 | Strategy | Trailing |
|---|---|---|
| BTCUSDT | B_trend (long-short MA crossover) | Trail2% |
| ETHUSDT | C_trend_long_only | Trail2% |
| SOLUSDT | D_trend+funding (펀딩 필터) | Trail2% |
| AVAXUSDT | B_trend | Trail3% |
| BNBUSDT | D_trend+funding | Trail2% |

**자본**: 5종목 균등 분할 + Multi-asset margin (USDT+USDC pool)

**로그**: `data/crypto_per_symbol_log_<TS>.json`

## 5. 모니터링

### 실시간 상태 확인

```powershell
# KIS 잔고 직접 조회
python scripts\verify_kis.py

# Binance 잔고 확인
python -c "from autotrader.broker.binance_testnet_client import *; ..."

# 로그 tailing
Get-Content data\dual_trading.log -Wait -Tail 20
```

### Dashboard

http://localhost:8501 에서 실시간 자동 새로고침 (5초)

## 6. 정지

```powershell
# 모두 정지
.\release\stop_all.bat

# 개별 정지 (Ctrl+C)
# 각 콘솔 창에서
```

## 7. 폴더 구조

```
ai_pricing/
├── .env                          ← API 키 (gitignored)
├── .env.example                  ← 템플릿
├── packages/                     ← 코드 모듈
│   └── autotrader/
│       └── src/autotrader/
│           ├── broker/           (KIS, Binance client)
│           ├── backtest/         (전략 + backtest engine)
│           ├── data/             (시세 fetch + 캐시)
│           └── paper/            (paper trading state)
├── scripts/
│   ├── run_dual_paper_trading.py     ← 주식 runner
│   ├── dual_dashboard.py              ← 대시보드
│   └── run_crypto_per_symbol.py      ← 코인 runner
├── data/                         ← 로그/상태 (런타임 생성)
├── release/                      ← 본 배포 패키지
│   ├── README.md
│   ├── install.bat
│   ├── setup_env.bat
│   ├── start_kis_runner.bat
│   ├── start_kis_dashboard.bat
│   ├── start_crypto_runner.bat
│   ├── start_all.bat
│   ├── stop_all.bat
│   └── requirements.txt
└── docs/                         ← 보고서
```

## 8. 자주 묻는 질문

### Q. 실거래 (실제 돈) 하려면?

**기본은 모의투자**. 실거래로 전환:

1. KIS: `.env` 의 `KIS_ENV=vts` → `KIS_ENV=prod` + `KIS_DRY_RUN=false`
2. Binance: 별도 계정 (mainnet) 키 발급 + 코드 수정 필요

⚠ **실거래 전 충분한 backtest + paper trading 검증 필수**

### Q. 매일 자동 시작 (Windows 스케줄러)?

```powershell
# 관리자 PowerShell 에서
schtasks /Create /SC DAILY /ST 09:00 /TN "KIS_Dual_Runner" `
  /TR "d:\simeunchul\ai_pricing\release\start_kis_runner.bat"
```

### Q. 매도 신호 떴는데 매도 안 됨

KIS API 일시 장애 (500 error) 시 다음 5분 polling 에 자동 재시도. state.positions 와 KIS holdings 가 sync 되도록 [scripts/run_dual_paper_trading.py](scripts/run_dual_paper_trading.py) 의 매도 로직이 주문 성공 시만 state 업데이트.

### Q. Drift 발생 (state vs KIS 차이)

```powershell
# state 강제 sync (KIS holdings 기준 재구성)
python scripts\sync_state_with_kis.py
```

## 9. 위험 + 면책

- 본 시스템은 **학습/포트폴리오용**
- KIS 모의투자 / Binance Testnet = **가짜 자금**
- 실거래 시 책임은 사용자 본인
- backtest 결과 ≠ 미래 수익률
- Sample size, market regime, overfitting 한계 인정

## 10. 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| `ModuleNotFoundError` | 의존성 미설치 | `install.bat` 재실행 |
| `KIS_APP_KEY 미설정` | .env 누락 | `.env.example` 복사 + 키 입력 |
| `500 Server Error` (KIS) | vts 일시 장애 | 5분 후 자동 재시도 |
| `EGW00201` (KIS) | rate limit 초과 | poll interval 늘림 |
| Streamlit 안 열림 | 8501 포트 사용 중 | `stop_all.bat` 후 재시작 |
| Binance auth 실패 | testnet 키 ≠ mainnet | testnet.binancefuture.com 에서 발급 |

## 11. 관련 보고서

- [docs/2026-04-30/crypto_3period_comparison.html](docs/2026-04-30/crypto_3period_comparison.html) — 3-period crypto backtest
- [docs/2026-05-01/crypto_final_summary.html](docs/2026-05-01/crypto_final_summary.html) — Trailing stop 검증
- [docs/2026-04-28/외국인_기관_dual_realtime.html](docs/2026-04-28/외국인_기관_dual_realtime.html) — Dual confirmation 전략
