#!/usr/bin/env bash
# Run once with sudo:  sudo bash deploy/install.sh <username> <password>
# Installs the systemd service, the nginx site on port 8090 with basic auth, and opens the firewall.
set -euo pipefail
USER_NAME="${1:?usage: sudo bash deploy/install.sh <username> <password>}"
PASSWORD="${2:?usage: sudo bash deploy/install.sh <username> <password>}"
HERE="$(cd "$(dirname "$0")" && pwd)"

# 1. app as a systemd service (auto-start on boot, auto-restart on crash)
install -m 644 "$HERE/btcpred.service" /etc/systemd/system/btcpred.service
systemctl daemon-reload
systemctl enable --now btcpred
sleep 3
systemctl is-active btcpred >/dev/null || { journalctl -u btcpred -n 20 --no-pager; exit 1; }

# 2. nginx reverse proxy with basic auth on port 8090
printf '%s:%s\n' "$USER_NAME" "$(openssl passwd -apr1 "$PASSWORD")" > /etc/nginx/btcpred.htpasswd
chmod 640 /etc/nginx/btcpred.htpasswd; chown root:www-data /etc/nginx/btcpred.htpasswd
install -m 644 "$HERE/nginx-btcpred.conf" /etc/nginx/sites-available/btcpred
ln -sf /etc/nginx/sites-available/btcpred /etc/nginx/sites-enabled/btcpred
nginx -t && systemctl reload nginx

# 3. firewall (ufw, if present and active)
if command -v ufw >/dev/null && ufw status | grep -q "Status: active"; then ufw allow 8090/tcp; fi

IP="$(curl -s -m 5 https://api.ipify.org || hostname -I | awk '{print $1}')"
echo
echo "Done. Open:  http://$IP:8090   (user: $USER_NAME)"
echo "If it does not load, also open TCP 8090 in your VPS provider's cloud firewall."
