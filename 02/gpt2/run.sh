#!/usr/bin/env bash
# Uruchamia gpt2.ipynb jako niezależną usługę systemd użytkownika (przeżywa zamknięcie terminala),
# z blokadą uśpienia i zamknięcia klapy na czas obliczeń.
#
#   ./run.sh pilot     -> gpt2_pilot.ipynb (~30 min)
#   ./run.sh full      -> gpt2_full.ipynb  (~3-3,5 h)
#
# Podgląd:   tail -f logs/<tryb>.log | grep -E '^\[ *[0-9.]+ min\]'
# Status:    systemctl --user status gpt2-<tryb>
# Przerwij:  systemctl --user stop gpt2-<tryb>   (ponowne ./run.sh wznawia od zapisanych wyników)
set -euo pipefail

MODE="${1:?użycie: ./run.sh pilot|full}"
case "$MODE" in
  pilot) FLAG=True ;;
  full) FLAG=False ;;
  *) echo "nieznany tryb: $MODE (pilot|full)" >&2; exit 1 ;;
esac

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
mkdir -p "$HERE/logs"

systemd-run --user --unit="gpt2-$MODE" \
  --working-directory="$HERE" \
  --setenv=HF_HUB_OFFLINE=1 --setenv=PYTHONUNBUFFERED=1 \
  -p StandardOutput="append:$HERE/logs/$MODE.log" -p StandardError=inherit \
  systemd-inhibit --what=sleep:idle:handle-lid-switch --who="gpt2-$MODE" --why="Lab02 GPT-2 $MODE" \
  "$ROOT/.venv/bin/papermill" gpt2.ipynb "gpt2_$MODE.ipynb" -p PILOT "$FLAG" --log-output -k python3

echo "Start: gpt2-$MODE. Log: $HERE/logs/$MODE.log"
