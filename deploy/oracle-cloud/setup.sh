#!/usr/bin/env bash
# Oracle Cloud Ubuntu 22.04 ARM 한 줄 deployment 스크립트.
#
# 실행 방법 (VM 에 SSH 접속 후):
#   wget https://raw.githubusercontent.com/simeunchul/ai_pricing/master/deploy/oracle-cloud/setup.sh
#   chmod +x setup.sh
#   ./setup.sh
#
# 이 스크립트가 하는 일:
#   1. 시스템 패키지 업데이트 + python3-venv 설치
#   2. 한국시간(KST) 으로 timezone 설정
#   3. /home/ubuntu/ai_pricing 에 repo clone
#   4. Python venv 생성 + 의존성 설치
#   5. .env 템플릿 안내 (사용자가 KIS 키 입력 필요)
#   6. systemd 서비스 등록 (자동 시작 + 자동 재시작)
#   7. 방화벽 8501 (dashboard) 열기

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/simeunchul/ai_pricing.git}"
INSTALL_DIR="${INSTALL_DIR:-/home/ubuntu/ai_pricing}"

log() {
    echo "[$(date '+%H:%M:%S')] $*"
}

# ===== 1. 시스템 준비 =====
log "1/7 시스템 업데이트 + 필수 패키지 설치"
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3 python3-venv python3-pip \
    git curl wget \
    build-essential \
    libssl-dev libffi-dev

# ===== 2. Timezone KST =====
log "2/7 timezone 을 Asia/Seoul 로 설정"
sudo timedatectl set-timezone Asia/Seoul

# ===== 3. Repo clone =====
if [ ! -d "$INSTALL_DIR" ]; then
    log "3/7 repo clone → $INSTALL_DIR"
    git clone "$REPO_URL" "$INSTALL_DIR"
else
    log "3/7 repo 이미 존재 — git pull"
    cd "$INSTALL_DIR" && git pull
fi

cd "$INSTALL_DIR"

# ===== 4. Python venv + 의존성 =====
log "4/7 Python venv 생성"
if [ ! -d "$INSTALL_DIR/venv" ]; then
    python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate

log "    pip + 의존성 설치"
pip install --quiet --upgrade pip wheel
# 핵심 의존성만 (full requirements 는 너무 큼)
pip install --quiet \
    requests \
    pandas \
    numpy \
    pyarrow \
    streamlit \
    altair

# ===== 5. data 디렉토리 + .env 템플릿 =====
log "5/7 data 디렉토리 생성"
mkdir -p data/krx_cache data/news_cache data/market_snapshots data/hedging_paths

if [ ! -f "$INSTALL_DIR/.env" ]; then
    log "    .env 템플릿 생성 — KIS 키 직접 입력 필요"
    cat > "$INSTALL_DIR/.env" <<'EOF'
# KIS 모의투자(vts) 키 — Korea Investment Securities Open API 신청
KIS_ENV=vts
KIS_APP_KEY=your_app_key_here
KIS_APP_SECRET=your_app_secret_here
KIS_ACCOUNT_NO=your_8digit_account
KIS_ACCOUNT_PROD=01
KIS_DRY_RUN=false
EOF
    chmod 600 "$INSTALL_DIR/.env"
    echo ""
    echo "    ⚠ /home/ubuntu/ai_pricing/.env 를 vi 등으로 편집해서 KIS 키 입력해주세요."
    echo ""
fi

# ===== 6. systemd 서비스 등록 =====
log "6/7 systemd 서비스 설치"
sudo cp "$INSTALL_DIR/deploy/oracle-cloud/systemd/dual-bot.service" /etc/systemd/system/
sudo cp "$INSTALL_DIR/deploy/oracle-cloud/systemd/dual-dashboard.service" /etc/systemd/system/
sudo cp "$INSTALL_DIR/deploy/oracle-cloud/systemd/eod-reconcile.service" /etc/systemd/system/
sudo cp "$INSTALL_DIR/deploy/oracle-cloud/systemd/eod-reconcile.timer" /etc/systemd/system/

sudo systemctl daemon-reload

# 자동 시작 등록 (실제 start 는 .env 채운 뒤)
sudo systemctl enable dual-bot.service
sudo systemctl enable dual-dashboard.service
sudo systemctl enable eod-reconcile.timer

# ===== 7. 방화벽 + 보안 그룹 =====
log "7/7 방화벽 8501 포트 (dashboard) 허용"
# Oracle Cloud 의 iptables 가 기본 INPUT DROP 이라 명시적 ACCEPT 필요
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 8501 -j ACCEPT || true
sudo iptables-save | sudo tee /etc/iptables/rules.v4 > /dev/null || true

echo ""
echo "================================================================"
echo "  ✓ 설치 완료"
echo "================================================================"
echo ""
echo "  다음 단계:"
echo "    1. .env 편집 + KIS 키 입력:"
echo "         vi $INSTALL_DIR/.env"
echo ""
echo "    2. 봇 + dashboard 시작:"
echo "         sudo systemctl start dual-bot dual-dashboard eod-reconcile.timer"
echo ""
echo "    3. 상태 확인:"
echo "         systemctl status dual-bot"
echo "         journalctl -u dual-bot -f"
echo ""
echo "    4. Dashboard 접속:"
echo "         http://<VM 공인 IP>:8501"
echo "         (Oracle Cloud Console 의 Security List 에도 8501 포트 추가 필요)"
echo ""
echo "    5. 운영 명령어:"
echo "         systemctl restart dual-bot   # 코드 업데이트 후 재시작"
echo "         systemctl stop dual-bot      # 정지"
echo "         journalctl -u dual-bot -n 50 # 최근 50줄 로그"
echo ""
echo "================================================================"
