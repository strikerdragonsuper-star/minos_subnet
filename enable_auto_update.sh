#!/usr/bin/env bash
# Enable safe automatic updates for this Minos miner or validator.
#
# What this does:
#   1. Finds the PM2 process that runs your miner or validator.
#   2. Saves the selected process name in ~/.minos/auto_update.env.
#   3. Starts a small PM2 updater named "minos-auto-update".
#
# The updater is intentionally cautious:
#   - It skips updates when your git checkout has local changes.
#   - It only pulls fast-forward updates from the current branch.
#   - It restarts only the PM2 process you confirm here.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${HOME}/.minos"
CONFIG_FILE="${CONFIG_DIR}/auto_update.env"
UPDATER_NAME="minos-auto-update"
UPDATER_SCRIPT="${ROOT_DIR}/scripts/auto_update.sh"

green() { printf '\033[0;32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[1;33m%s\033[0m\n' "$*"; }
red() { printf '\033[0;31m%s\033[0m\n' "$*"; }
info() { printf '%s\n' "$*"; }

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    red "Missing required command: $1"
    if [[ "$1" == "pm2" ]]; then
      info "Install PM2 with: npm install -g pm2"
    fi
    exit 1
  fi
}

quote_env() {
  # Print a value safely for a simple shell env file.
  printf "%q" "$1"
}

detect_role() {
  local text
  text="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  if [[ "$text" == *validator* ]]; then
    printf "validator"
  elif [[ "$text" == *miner* ]]; then
    printf "miner"
  else
    printf "unknown"
  fi
}

candidate_rows() {
  local repo="$1"
  pm2 jlist | node -e '
const fs = require("fs");
const repo = process.argv[1];
const input = fs.readFileSync(0, "utf8");
let rows = [];
try {
  const processes = JSON.parse(input);
  rows = processes
    .map((proc) => {
      const env = proc.pm2_env || {};
      const name = proc.name || "";
      const cwd = env.pm_cwd || "";
      const script = env.pm_exec_path || env.script || "";
      const args = Array.isArray(env.args) ? env.args.join(" ") : String(env.args || "");
      const haystack = `${name} ${cwd} ${script} ${args}`.toLowerCase();
      const role = haystack.includes("validator")
        ? "validator"
        : haystack.includes("miner")
          ? "miner"
          : "unknown";
      let score = 0;
      if (cwd === repo) score += 5;
      if (cwd.startsWith(repo)) score += 3;
      if (name.toLowerCase().includes("minos")) score += 2;
      if (role !== "unknown") score += 4;
      if (script.toLowerCase().includes("start-validator") || script.toLowerCase().includes("start-miner")) score += 3;
      return { name, role, cwd, script, score };
    })
    .filter((row) => row.score > 0)
    .sort((a, b) => b.score - a.score || a.name.localeCompare(b.name));
} catch (error) {
  process.exit(2);
}
for (const row of rows) {
  console.log([row.name, row.role, row.cwd, row.script].join("\t"));
}
' "$repo"
}

choose_process() {
  local rows_file="$1"
  local count
  count="$(wc -l < "$rows_file" | tr -d ' ')"

  if [[ "$count" == "0" ]]; then
    yellow "No likely Minos PM2 process was detected."
    info "Run 'pm2 list' and enter the exact PM2 process name to update."
    read -r -p "PM2 process name: " selected_name
    selected_role="$(detect_role "$selected_name")"
    return
  fi

  info ""
  info "Detected PM2 process candidates:"
  local index=1
  while IFS=$'\t' read -r name role cwd script; do
    printf "  %s) %s  [%s]\n" "$index" "$name" "$role"
    printf "     cwd: %s\n" "${cwd:-unknown}"
    printf "     script: %s\n" "${script:-unknown}"
    index=$((index + 1))
  done < "$rows_file"

  if [[ "$count" == "1" ]]; then
    local name role cwd script
    IFS=$'\t' read -r name role cwd script < "$rows_file"
    info ""
    read -r -p "Use PM2 process '${name}' for auto-update? [Y/n]: " answer
    answer="${answer:-Y}"
    if [[ ! "$answer" =~ ^[Yy]$ ]]; then
      read -r -p "PM2 process name: " selected_name
      selected_role="$(detect_role "$selected_name")"
      return
    fi
    selected_name="$name"
    selected_role="$role"
    return
  fi

  info ""
  read -r -p "Select PM2 process number to update: " choice
  if ! [[ "$choice" =~ ^[0-9]+$ ]] || (( choice < 1 || choice > count )); then
    red "Invalid selection."
    exit 1
  fi

  local line
  line="$(sed -n "${choice}p" "$rows_file")"
  IFS=$'\t' read -r selected_name selected_role _ <<< "$line"
}

require_command git
require_command pm2
require_command node

if [[ ! -f "${UPDATER_SCRIPT}" ]]; then
  red "Missing updater script: ${UPDATER_SCRIPT}"
  exit 1
fi

cd "$ROOT_DIR"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  red "This folder is not a git repository: ${ROOT_DIR}"
  exit 1
fi

BRANCH="$(git branch --show-current)"
if [[ -z "$BRANCH" ]]; then
  red "Could not detect the current git branch."
  exit 1
fi

REMOTE="${MINOS_AUTO_UPDATE_REMOTE:-origin}"
if ! git remote get-url "$REMOTE" >/dev/null 2>&1; then
  red "Git remote '${REMOTE}' was not found."
  exit 1
fi

tmp_rows="$(mktemp)"
trap 'rm -f "$tmp_rows"' EXIT
candidate_rows "$ROOT_DIR" > "$tmp_rows"

selected_name=""
selected_role=""
choose_process "$tmp_rows"

if [[ -z "$selected_name" ]]; then
  red "No PM2 process selected."
  exit 1
fi

mkdir -p "$CONFIG_DIR"
cat > "$CONFIG_FILE" <<EOF
# Minos auto-update settings.
# Edit this file only if you rename your PM2 process or move the repo.
MINOS_AUTO_UPDATE_REPO=$(quote_env "$ROOT_DIR")
MINOS_AUTO_UPDATE_REMOTE=$(quote_env "$REMOTE")
MINOS_AUTO_UPDATE_BRANCH=$(quote_env "$BRANCH")
MINOS_AUTO_UPDATE_PM2_PROCESS=$(quote_env "$selected_name")
MINOS_AUTO_UPDATE_ROLE=$(quote_env "${selected_role:-unknown}")
MINOS_AUTO_UPDATE_INTERVAL_SECONDS=${MINOS_AUTO_UPDATE_INTERVAL_SECONDS:-300}
MINOS_AUTO_UPDATE_ALLOW_DIRTY=0
MINOS_AUTO_UPDATE_LOG=$(quote_env "${CONFIG_DIR}/auto_update.log")
EOF

chmod 600 "$CONFIG_FILE"
chmod +x "$UPDATER_SCRIPT"

info ""
green "Saved auto-update config:"
info "  ${CONFIG_FILE}"
info "  repo: ${ROOT_DIR}"
info "  branch: ${BRANCH}"
info "  process: ${selected_name}"

if pm2 describe "$UPDATER_NAME" >/dev/null 2>&1; then
  pm2 restart "$UPDATER_NAME" --update-env
else
  pm2 start "$UPDATER_SCRIPT" --name "$UPDATER_NAME" --interpreter bash
fi

pm2 save

info ""
green "Auto-update is enabled."
info "View logs with: pm2 logs ${UPDATER_NAME}"
info "The updater will skip pulls if this repo has local git changes."
