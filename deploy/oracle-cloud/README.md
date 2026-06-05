# Oracle Cloud Always Free 배포

ARM Ampere A1.Flex VM (영구 무료) 에 자동매매 봇 deploy.

## 빠른 시작 (VM 안에서)

```bash
wget https://raw.githubusercontent.com/simeunchul/ai_pricing/master/deploy/oracle-cloud/setup.sh
chmod +x setup.sh
./setup.sh

# .env 편집 후
vi /home/ubuntu/ai_pricing/.env

# 시작
sudo systemctl start dual-bot dual-dashboard eod-reconcile.timer
```

## 사전 준비

상세 가이드: [docs/2026-06-05/cloud_migration_guide.html](../../docs/2026-06-05/cloud_migration_guide.html)

1. Oracle Cloud Always Free 계정 (Seoul region 권장)
2. SSH 키페어 생성
3. VM.Standard.A1.Flex Ubuntu 22.04 인스턴스 생성
4. Security List 에 8501 포트 (dashboard) 허용

## 운영 cheat sheet

```bash
# 봇 재시작
sudo systemctl restart dual-bot

# 로그 실시간
journalctl -u dual-bot -f

# 코드 업데이트 (git pull + 재시작)
cd ~/ai_pricing && git pull && sudo systemctl restart dual-bot

# EOD reconcile 즉시 실행
sudo systemctl start eod-reconcile.service

# Dashboard 접속
# http://<VM 공인 IP>:8501
```

## 구조

| 서비스 | 역할 | 자동복구 |
|---|---|---|
| `dual-bot.service` | KIS 자동매매 봇 (run_dual_paper_trading.py --watch) | Restart=always |
| `dual-dashboard.service` | Streamlit dashboard (port 8501) | Restart=always |
| `eod-reconcile.timer` | 매일 15:35 KST 자동 정합 검증 | Persistent=true |

systemd 가 Windows watchdog 을 완전히 대체 — 별도 watchdog 스크립트 불필요.
