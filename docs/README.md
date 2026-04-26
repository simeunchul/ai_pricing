# docs/

프로젝트 진단·분석 리포트 모음. 브라우저에서 바로 열리는 HTML 위주.

| 파일 | 내용 | 작성일 |
|---|---|---|
| [e2e_flow.html](e2e_flow.html) | **E2E 흐름 설명서.** 처음 보는 사람 대상. 입력 → Layer → 지표 → 산출물 전체 flow. 핵심 개념 5개, 합성 vs 실제 데이터 구분, 4개 AI 모델 입출력, 지표 카탈로그, 실행 절차 Phase 1~5 | 2026-04-24 |
| [hanwha_els_reprice.html](hanwha_els_reprice.html) | **DART 실데이터 기반 Layer A 검증.** 한화스마트ELS 제8286호 (3자산 KOSPI200/S&P500/SX5E, 3Y, 연 11.31%) 공시 → XML 파싱 → MC 재가격. base 10,217원 vs 공시 10,000원 **+2.17%** (±3% 목표 달성) | 2026-04-24 |
| [full_stack_status.html](full_stack_status.html) | **5 Layer 전체 상태 스냅샷.** 각 레이어별 완료 여부 + 핵심 지표 + 플랜 목표 대비. Layer A/B1/B2/B4/E 목표 달성, B3 실데이터 진단 완료, D DRY_RUN 통과 | 2026-04-26 |
| [nn_pricer_underfit_analysis.html](nn_pricer_underfit_analysis.html) | Layer B1 NN Pricer 의 underfit + Deep OTM 838% 문제 원인 분석, loss function 교체 방안, IV-space 평가 전환, 코드 위치 상세 맵 | 2026-04-24 |
| [nn_pricer_retrain_results.html](nn_pricer_retrain_results.html) | 위 분석의 해결방안 실제 적용 결과. MSE → log-space MSE 로 교체 후 Before/After 비교 + 500k CPU 학습 결과 (플랜 목표 달성) | 2026-04-24 |
| [deep_hedging_tuning.html](deep_hedging_tuning.html) | Layer B4 Deep Hedging PPO 미수렴 진단 및 v2 수정 결과. Action space 축소 + Buehler 2019 dense reward shaping + BSM behavior-cloning warm-start. TC=0 sanity PASS (1.02x), TC=30bps 7.8% 개선 (플랜 20% 목표 PARTIAL) | 2026-04-24 |
| [resume_one_pager.md](resume_one_pager.md) | **이력서 원페이지** — 핵심 성과 7개 (수치 위주), JD 매핑, 라이브 데모 60초 시나리오 | 2026-04-26 |
| [blog_drafts.md](blog_drafts.md) | **블로그 5편 초안** — 모노레포 소개 / NN Pricer 진단 / 한화 ELS 재가격 / Deep Hedging Buehler 성공 / News-IV 가설 실패 진단. 발행 순서 권장 | 2026-04-26 |
| [why_dh_failed.html](why_dh_failed.html) | **Deep Hedging 5번 모두 실패한 이유** 시각 분석. 목적함수 mismatch (PPO=E[reward] vs CVaR), BSM attractor 다이어그램, entropy collapse 곡선, 4가지 알고리즘 변경 경로, 면접 talking points 3개 | 2026-04-25 |
| [daily_report_20260425.html](daily_report_20260425.html) | **2026-04-25 일일 작업 보고서** — 3 commits 의 변경사항, B2 SPY 실데이터 milestone, B4 5번째 실패, 면접 자료 작성 | 2026-04-25 |
| [2026-04-25/B4_buehler_implementation.html](2026-04-25/B4_buehler_implementation.html) | **🏆 Layer B4 Buehler 2019 직접 구현으로 플랜 목표 달성.** PPO 5번 실패 후 PG-on-CVaR 직접 구현. CPU 2분 16초 학습으로 ratio 0.755 = **+24.5% 개선** (PASS ≥20%) | 2026-04-25 |
| [2026-04-25/portfolio_update.html](2026-04-25/portfolio_update.html) | **포트폴리오 일괄 갱신 (B4 PASS 반영) 작업 보고서** — resume / blog / status / csv 4파일 narrative 동기화 + 끊김 시점 진단 + 재개 절차 | 2026-04-25 |
| [2026-04-26/B3_news_iv_backtest.html](2026-04-26/B3_news_iv_backtest.html) | **Layer B3 News-IV 매크로 가설 실데이터 검증.** FOMC 18 + CPI 28 (N=46) × VIX 부호 일치율 **34.8% (FAIL)** → vol-crush 진단 → top-20% magnitude 60% PASS. 케이스 스터디 5건 첨부 | 2026-04-26 |
| [2026-04-26/production_gap_closure.html](2026-04-26/production_gap_closure.html) | **실데이터·실무 격차 좁히기 5 항목 일괄.** 매크로 캘린더 46→208건, B3 v2 surprise-aware 54.8%, **B4 SPY historical +31.3% PASS**, ELS 표본 1→4건, fastmc N=10k 옵션 9.8초 | 2026-04-26 |
| [2026-04-26/desk_day_workflow.html](2026-04-26/desk_day_workflow.html) | **Phase 1 — 한화 ELS 데스크의 하루.** desk_day.py orchestration 7 stage + 시각 3 메타박스 (PPO 5번 실패) + compliance 모듈 (audit log·편차 모니터·HITL) 신설 | 2026-04-26 |
| [2026-04-26/lp_upgrade.html](2026-04-26/lp_upgrade.html) | **Phase 2 — ETF LP Avellaneda-Stoikov 업그레이드.** 단방향 iNAV → 양방향 LP + 재고 페널티. 6개월 KODEX 200 일봉 백테스트 + 3 config sweep. trending market 한계 진단 + Phase 1 compliance hook cross-cutting 통합 검증 | 2026-04-26 |
| [2026-04-26/project_overview.html](2026-04-26/project_overview.html) | **🧭 프로젝트 종합 설명서 (start here).** "이 저장소가 뭘 하는지, 어떻게 쓰는지, 진짜로 동작하는지" 한 문서. pytest 37/37, desk_day 7-stage 풀파이프라인 검증, 5 Layer 점수판, 22개 용어 정리집 | 2026-04-26 |
| [2026-04-26/jump_stress_test.html](2026-04-26/jump_stress_test.html) | **B4 Buehler — Merton jump 옵션 추가 + 철학 보존 검증.** HedgingEnv 에 jump 4-필드 (default 0=호환), 41/41 pytest, 4 시나리오 stress test: GBM +21.28% / 일상 +4.41% / 어닝쇼크 +8.69% / 코로나급 −1.53%. 정책 입력에 jump 신호 0개 → "예측 안 함" 철학 보존 | 2026-04-26 |
| [2026-04-26/jump_aware_training.html](2026-04-26/jump_aware_training.html) | **B4 Jump-Aware Retraining — specialization vs generalization 실증.** 정책 input 5-dim 그대로, 환경만 비정형화하며 retrain. 3 모델 × 4 시나리오 매트릭스: GBM 학습 (전문가) vs Jump-fixed (편식, Medium +30.47%/GBM −101.47%) vs Jump-random (제너럴리스트). NFL 실증 + regime 별 ensemble 권장 | 2026-04-26 |
| [2026-04-26/B3_kr_naver_backtest.html](2026-04-26/B3_kr_naver_backtest.html) | **B3 한국 시장 Naver 풀 파이프 — Negative Result + 진단.** Naver 검색 API 9,291 헤드라인 → 825 분류 → 612 정렬, 3가지 가설 모두 FAIL (event 일 ratio 0.39×, baseline 2.96% 비정상). 4가지 개선 경로 제시 (event dedup + magnitude scoring 우선) | 2026-04-26 |

## 열어보는 방법

```bash
# Windows
start docs/nn_pricer_underfit_analysis.html

# macOS / Linux
open docs/nn_pricer_underfit_analysis.html
xdg-open docs/nn_pricer_underfit_analysis.html

# VSCode
code docs/nn_pricer_underfit_analysis.html   # 이후 우측 상단 Preview 버튼
```

모든 HTML 은 **외부 의존성 없이** 오프라인에서 완전 동작 (inline SVG, 임베디드 CSS).
