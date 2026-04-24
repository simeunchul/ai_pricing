# docs/

프로젝트 진단·분석 리포트 모음. 브라우저에서 바로 열리는 HTML 위주.

| 파일 | 내용 | 작성일 |
|---|---|---|
| [nn_pricer_underfit_analysis.html](nn_pricer_underfit_analysis.html) | Layer B1 NN Pricer 의 underfit + Deep OTM 838% 문제 원인 분석, loss function 교체 방안, IV-space 평가 전환, 코드 위치 상세 맵 | 2026-04-24 |
| [nn_pricer_retrain_results.html](nn_pricer_retrain_results.html) | 위 분석의 해결방안 실제 적용 결과. MSE → log-space MSE 로 교체 후 Before/After 비교. ATM IV error 2× 개선, ATM point 3.7× 개선 | 2026-04-24 |

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
