# Portable 빌드 가이드 — 환경설정 없이 실행 가능한 .zip 만들기

이 문서는 **개발자용** — 사용자에게 배포할 portable .zip 만드는 방법.

## 목표

```
사용자 PC 에서:
  1. ai_pricing_portable.zip 다운로드 (~1GB)
  2. 압축 풀기
  3. start.exe 더블클릭
  → Python 설치 / pip / .env 편집 등 일체 불필요
```

## 빌드 단계 (개발자 PC 에서, 한 번만)

### 1. Python Embeddable 다운로드

```powershell
# Windows x64 embeddable zip
$url = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip"
Invoke-WebRequest -Uri $url -OutFile python-embed.zip
Expand-Archive python-embed.zip -DestinationPath release\python_portable
```

### 2. pip 활성화

```powershell
# python311._pth 편집 → "import site" 주석 해제
$pthFile = "release\python_portable\python311._pth"
(Get-Content $pthFile) -replace '#import site', 'import site' | Set-Content $pthFile

# get-pip.py 다운로드 + 실행
Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile get-pip.py
release\python_portable\python.exe get-pip.py
del get-pip.py
```

### 3. 의존성 사전 설치

```powershell
release\python_portable\python.exe -m pip install -r release\requirements.txt
```

이 단계가 ~500MB 다운로드 (pandas/streamlit/binance 등). 약 5~10분.

### 4. start.exe 만들기 (PyInstaller)

```powershell
# launcher 코드 (release\launcher.py) 작성 후
pip install pyinstaller
pyinstaller --onefile --windowed --icon=icon.ico release\launcher.py -n start

# 결과: dist\start.exe → release\start.exe 복사
```

### 5. .zip 패키징

```powershell
Compress-Archive -Path release, scripts, packages, config, .env.example `
                 -DestinationPath ai_pricing_portable.zip
```

`ai_pricing_portable.zip` 생성 (~1GB). 사용자에게 배포.

## 사용자 측 흐름 (받는 쪽)

```
1. ai_pricing_portable.zip 받음
2. 우클릭 → 압축 풀기 → ai_pricing\
3. ai_pricing\start.exe 더블클릭
   ↓
   첫 실행:
     ┌─────────────────────────────────────────┐
     │ API 키 입력 wizard (GUI)                │
     │   KIS_APP_KEY:    [____________]        │
     │   KIS_APP_SECRET: [____________]        │
     │   KIS_ACCOUNT:    [____________]        │
     │   Binance Key:    [____________]        │
     │   Binance Secret: [____________]        │
     │            [저장]                       │
     └─────────────────────────────────────────┘
   ↓
   2회 이후 실행:
     ┌─────────────────────────────────────────┐
     │ 자동매매 시스템                         │
     │                                         │
     │  [▶ 모두 시작]                         │
     │  [■ 모두 정지]                         │
     │  [⚙ 종목 설정]                         │
     │  [📊 대시보드 열기]                    │
     │                                         │
     │  상태: ✓ KIS runner 실행 중             │
     │       ✓ Crypto runner 실행 중           │
     │       ✓ Dashboard http://localhost:8501│
     └─────────────────────────────────────────┘
```

## 현재 상태 (Quick Path — 개발자가 한 번 빌드 필요)

지금까지 완성된 것:
- ✓ 종목 설정 시스템 ([config/symbols.json](config/symbols.json))
- ✓ Dashboard 사이드바 종목 편집 GUI
- ✓ Runner 가 config 자동 읽음 (whitelist/blacklist 적용)
- ✓ 기존 .bat 스크립트 (Python 설치된 PC 에서 작동)

미완성 (full portable 위해):
- [ ] launcher.py (Tkinter GUI for start/stop/config)
- [ ] First-run wizard (API 키 GUI)
- [ ] PyInstaller 빌드 (start.exe)
- [ ] Python embeddable 번들

## 임시 해결책 — 부분 portable

지금 받는 사람이 Python 만 설치되어 있으면:

```
1. 코드 zip
2. install.bat 실행 (pip install)
3. setup_env.bat 실행 (.env 편집)
4. start_all.bat 실행
```

→ Python 설치 + .env 편집 두 번 손이 가지만 코드 변경 X.

## Full portable 빌드 시간 견적

| 단계 | 시간 |
|---|---|
| Python embeddable + pip 활성화 | 10분 |
| 의존성 설치 (~500MB) | 10분 |
| launcher.py 작성 (Tkinter GUI) | 1~2시간 |
| First-run wizard | 1시간 |
| PyInstaller 빌드 + 테스트 | 30분 |
| .zip 패키징 + 검증 | 30분 |
| **합계** | **~4시간** |

다음 세션에서 진행 가능. 지금은 부분 portable + 종목 설정 GUI 완성.
