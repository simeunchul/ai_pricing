# data/

| 폴더 | 내용 | 생성 방법 |
|---|---|---|
| `els_samples/` | 한화 ELS 공시 PDF | DART 수동 다운 |
| `market_snapshots/` | KOSPI200 옵션 일간 스냅샷 | `python -m scripts.snapshot_kospi200` |
| `news_cache/` | 네이버/DART 뉴스 캐시 | `ai_pricing.news_iv.fetch` 자동 |
| `hedging_paths/` | Deep Hedging 평가용 path | `ai_hedging.agents.ppo_hedger.evaluate_ppo` |

.gitignore 에서 바이너리/큰 파일은 제외.
