#!/usr/bin/env bash
# Disable Minos automatic updates on this machine.

set -euo pipefail

UPDATER_NAME="minos-auto-update"
CONFIG_FILE="${HOME}/.minos/auto_update.env"

if ! command -v pm2 >/dev/null 2>&1; then
  echo "PM2 is not installed, so no PM2 updater process can be stopped."
else
  if pm2 describe "$UPDATER_NAME" >/dev/null 2>&1; then
    pm2 delete "$UPDATER_NAME"
    pm2 save
    echo "Stopped PM2 updater: ${UPDATER_NAME}"
  else
    echo "No PM2 updater process named ${UPDATER_NAME} was running."
  fi
fi

if [[ -f "$CONFIG_FILE" ]]; then
  echo "Keeping config file for reference: ${CONFIG_FILE}"
  echo "Remove it manually if you do not want to keep the selected process/branch."
fi

echo "Auto-update is disabled."
