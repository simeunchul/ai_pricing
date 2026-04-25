# docs/

프로젝트 진단·분석 리포트 모음. 브라우저에서 바로 열리는 HTML 위주.

| 파일 | 내용 | 작성일 |
|---|---|---|
| [e2e_flow.html](e2e_flow.html) | **E2E 흐름 설명서.** 처음 보는 사람 대상. 입력 → Layer → 지표 → 산출물 전체 flow. 핵심 개념 5개, 합성 vs 실제 데이터 구분, 4개 AI 모델 입출력, 지표 카탈로그, 실행 절차 Phase 1~5 | 2026-04-24 |
| [hanwha_els_reprice.html](hanwha_els_reprice.html) | **DART 실데이터 기반 Layer A 검증.** 한화스마트ELS 제8286호 (3자산 KOSPI200/S&P500/SX5E, 3Y, 연 11.31%) 공시 → XML 파싱 → MC 재가격. base 10,217원 vs 공시 10,000원 **+2.17%** (±3% 목표 달성) | 2026-04-24 |
| [full_stack_status.html](full_stack_status.html) | **5 Layer 전체 상태 스냅샷.** 각 레이어별 완료 여부 + 핵심 지표 + 플랜 목표 대비. Layer A/B1/B2/E 목표 달성, B4 수렴 미완, B3/D 부분 완료 | 2026-04-24 |
| [nn_pricer_underfit_analysis.html](nn_pricer_underfit_analysis.html) | Layer B1 NN Pricer 의 underfit + Deep OTM 838% 문제 원인 분석, loss function 교체 방안, IV-space 평가 전환, 코드 위치 상세 맵 | 2026-04-24 |
| [nn_pricer_retrain_results.html](nn_pricer_retrain_results.html) | 위 분석의 해결방안 실제 적용 결과. MSE → log-space MSE 로 교체 후 Before/After 비교 + 500k CPU 학습 결과 (플랜 목표 달성) | 2026-04-24 |
| [deep_hedging_tuning.html](deep_hedging_tuning.html) | Layer B4 Deep Hedging PPO 미수렴 진단 및 v2 수정 결과. Action space 축소 + Buehler 2019 dense reward shaping + BSM behavior-cloning warm-start. TC=0 sanity PASS (1.02x), TC=30bps 7.8% 개선 (플랜 20% 목표 PARTIAL) | 2026-04-24 |
| [resume_one_pager.md](resume_one_pager.md) | **이력서 원페이지** — 핵심 성과 5개 (수치 위주), JD 매핑, 라이브 데모 60초 시나리오 | 2026-04-25 |
| [blog_drafts.md](blog_drafts.md) | **블로그 4편 초안** — 모노레포 소개 / NN Pricer 진단 / 한화 ELS 재가격 / Deep Hedging 실패. 발행 순서 권장 | 2026-04-25 |
| [why_dh_failed.html](why_dh_failed.html) | **Deep Hedging 5번 모두 실패한 이유** 시각 분석. 목적함수 mismatch (PPO=E[reward] vs CVaR), BSM attractor 다이어그램, entropy collapse 곡선, 4가지 알고리즘 변경 경로, 면접 talking points 3개 | 2026-04-25 |
| [daily_report_20260425.html](daily_report_20260425.html) | **2026-04-25 일일 작업 보고서** — 3 commits 의 변경사항, B2 SPY 실데이터 milestone, B4 5번째 실패, 면접 자료 작성 | 2026-04-25 |

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
