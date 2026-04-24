# docs/

프로젝트 진단·분석 리포트 모음. 브라우저에서 바로 열리는 HTML 위주.

| 파일 | 내용 | 작성일 |
|---|---|---|
| [full_stack_status.html](full_stack_status.html) | **5 Layer 전체 상태 스냅샷.** 각 레이어별 완료 여부 + 핵심 지표 + 플랜 목표 대비. Layer A/B1/B2/E 목표 달성, B4 수렴 미완, B3/D 부분 완료 | 2026-04-24 |
| [nn_pricer_underfit_analysis.html](nn_pricer_underfit_analysis.html) | Layer B1 NN Pricer 의 underfit + Deep OTM 838% 문제 원인 분석, loss function 교체 방안, IV-space 평가 전환, 코드 위치 상세 맵 | 2026-04-24 |
| [nn_pricer_retrain_results.html](nn_pricer_retrain_results.html) | 위 분석의 해결방안 실제 적용 결과. MSE → log-space MSE 로 교체 후 Before/After 비교 + 500k CPU 학습 결과 (플랜 목표 달성) | 2026-04-24 |

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
