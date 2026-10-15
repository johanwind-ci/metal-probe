#!/bin/bash
# Runs ON the CI Mac (GitHub Actions or CircleCI): key-only sshd exposed through a bore.pub TCP tunnel,
# then waits until the deadline in /tmp/session_deadline (extendable) or until /tmp/stop exists.
# Env: PUBKEY (the one key allowed in), PORT (bore.pub remote port), MINUTES, SESSION_ID (CI run id, lets the
# client check it reached its own runner and not something else on that bore.pub port).
set -eu

mkdir -p ~/.ssh && chmod 700 ~/.ssh
printf '%s\n' "$PUBKEY" >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
printf 'PasswordAuthentication no\nKbdInteractiveAuthentication no\nPermitRootLogin no\n' \
  | sudo tee /etc/ssh/sshd_config.d/000-session.conf > /dev/null
sudo launchctl load -w /System/Library/LaunchDaemons/ssh.plist 2>/dev/null || true
sudo launchctl kickstart -k system/com.openssh.sshd 2>/dev/null || true
for _ in 1 2 3 4 5; do nc -z localhost 22 && break; sleep 1; done
nc -z localhost 22 || { echo "sshd is not listening"; exit 1; }

echo "$SESSION_ID" > ~/.session_id
echo $(( $(date +%s) + MINUTES * 60 )) > /tmp/session_deadline

curl -fsSL https://github.com/ekzhang/bore/releases/download/v0.6.0/bore-v0.6.0-aarch64-apple-darwin.tar.gz | tar xz -C /tmp
/tmp/bore local 22 --to bore.pub --port "$PORT" > /tmp/bore.log 2>&1 &
BPID=$!
sleep 5
cat /tmp/bore.log
kill -0 "$BPID" 2>/dev/null || { echo "bore failed to start (port taken?)"; exit 1; }
echo "ready: ssh -p $PORT $(whoami)@bore.pub"

# Prints every 30 s: CircleCI kills steps that are silent for no_output_timeout.
while [ "$(date +%s)" -lt "$(cat /tmp/session_deadline)" ] && [ ! -f /tmp/stop ]; do
  kill -0 "$BPID" 2>/dev/null || { echo "bore exited:"; cat /tmp/bore.log; exit 1; }
  echo "alive $(date +%T), $(( ($(cat /tmp/session_deadline) - $(date +%s)) / 60 )) min left"
  sleep 30
done
echo "session over"
