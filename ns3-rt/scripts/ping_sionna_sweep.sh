#!/usr/bin/env bash
set -euo pipefail

# Sweep distances and record ping metrics to a CSV.
#
# Usage examples:
#   DISTANCES="10 20 30 40" SIM_TIME=120 PING_SECS=20 ./scripts/ping_sionna_sweep.sh
#   TAP_NAME=thetap OUT=results/ping_sweep.csv ./scripts/ping_sionna_sweep.sh
#   SIONNA_MODE=0 ./scripts/ping_sionna_sweep.sh   # baseline ns-3 (no Sionna server)

DISTANCES="${DISTANCES:-50 100 200 300 400 500}"
RUNS_PER_DIST="${RUNS_PER_DIST:-3}"
SIONNA_MODE="${SIONNA_MODE:-1}"
SIM_TIME="${SIM_TIME:-60}"
PING_SECS="${PING_SECS:-10}"
TAP_NAME="${TAP_NAME:-thetap}"
TAP_NAME_TEMPLATE="${TAP_NAME_TEMPLATE:-}"
TAP_MODE="${TAP_MODE:-ConfigureLocal}"
TAP_ADDR="${TAP_ADDR:-1.0.0.2/8}"
CLEAN_TAP="${CLEAN_TAP:-1}"
DEBUG="${DEBUG:-1}"
OUT_DIR="${OUT_DIR:-results}"
DEBUG_DIR="${DEBUG_DIR:-${OUT_DIR}/ping_sweep_debug}"
OUT="${OUT:-${OUT_DIR}/ping_sweep.csv}"
PING_DELAY="${PING_DELAY:-5}"
EXTRA_NS3_ARGS="${EXTRA_NS3_ARGS:-}"
OUT_PREFIX_BASE="${OUT_PREFIX_BASE:-${OUT_DIR}/summary/ping_sweep}"
NS3_WAIT="${NS3_WAIT:-}"

UE_IP="${UE_IP:-7.0.0.2}"
PGW_IP="${PGW_IP:-1.0.0.1}"

# Optional fixed coordinates + walking behavior
# If ENB_X/Y/Z and UE_X/Y/Z are set, the script will compute UE positions per distance.
# WALK_MODE=from_tx (default): UE position = ENB + unit(UE_START-ENB) * distance
# WALK_MODE=from_start: UE position = UE_START + unit(UE_START-ENB) * distance
# Optional WALK_DIR_X/Y/Z overrides the direction vector.
ENB_X="${ENB_X:-}"
ENB_Y="${ENB_Y:-}"
ENB_Z="${ENB_Z:-}"
UE_X="${UE_X:-}"
UE_Y="${UE_Y:-}"
UE_Z="${UE_Z:-}"
WALK_MODE="${WALK_MODE:-}"
WALK_DIR_X="${WALK_DIR_X:-}"
WALK_DIR_Y="${WALK_DIR_Y:-}"
WALK_DIR_Z="${WALK_DIR_Z:-}"

if [ -z "$NS3_WAIT" ]; then
  NS3_WAIT=$((SIM_TIME + 20))
fi

mkdir -p "$(dirname "$OUT")"
mkdir -p "$(dirname "$OUT_PREFIX_BASE")"
MODE_LABEL="baseline"
if [ "$SIONNA_MODE" = "1" ] || [ "$SIONNA_MODE" = "true" ]; then
  MODE_LABEL="sionna"
fi

echo "distance_param_m,run,tx,rx,loss_pct,avg_ms,distance_actual_m,sionna_pathgain_db,sionna_delay_s,sionna_los,mode" > "$OUT"

if [ "$DEBUG" = "1" ]; then
  mkdir -p "$DEBUG_DIR"
fi

terminate_ns3() {
  local pid="$1"
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    for _ in {1..20}; do
      if ! kill -0 "$pid" 2>/dev/null; then
        break
      fi
      sleep 0.2
    done
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
  fi
  wait "$pid" 2>/dev/null || true
}

wait_ns3() {
  local pid="$1"
  local timeout_s="$2"
  local waited=0
  while kill -0 "$pid" 2>/dev/null; do
    if [ "$waited" -ge "$timeout_s" ]; then
      return 1
    fi
    sleep 1
    waited=$((waited + 1))
  done
  return 0
}

for d in $DISTANCES; do
  for r in $(seq 1 "$RUNS_PER_DIST"); do
    if [ -n "$TAP_NAME_TEMPLATE" ]; then
      if printf "$TAP_NAME_TEMPLATE" "$d" >/dev/null 2>&1; then
        TAP_NAME_RUN="$(printf "$TAP_NAME_TEMPLATE" "$d")"
      else
        TAP_NAME_RUN="${TAP_NAME_TEMPLATE//\{d\}/$d}"
      fi
    else
      TAP_NAME_RUN="$TAP_NAME"
    fi

    echo "=== distance=${d}m run=${r}/${RUNS_PER_DIST} ==="

    # Always delete any stale tap before (re)creating
    sudo ip link delete "$TAP_NAME_RUN" 2>/dev/null || true
    sleep 0.5

    if [ "$TAP_MODE" != "ConfigureLocal" ]; then
      sudo ip tuntap add mode tap "$TAP_NAME_RUN"
      sudo ip addr replace "$TAP_ADDR" dev "$TAP_NAME_RUN"
      sudo ip link set "$TAP_NAME_RUN" up
    fi

    RUN_EXTRA_ARGS="${EXTRA_NS3_ARGS}"
    if [ -n "$ENB_X" ] && [ -n "$ENB_Y" ] && [ -n "$ENB_Z" ] && [ -n "$UE_X" ] && [ -n "$UE_Y" ] && [ -n "$UE_Z" ]; then
      if [ -z "$WALK_MODE" ]; then
        WALK_MODE="from_tx"
      fi
      UE_POS="$(python3 - "$ENB_X" "$ENB_Y" "$ENB_Z" "$UE_X" "$UE_Y" "$UE_Z" "$d" "$WALK_MODE" "$WALK_DIR_X" "$WALK_DIR_Y" "$WALK_DIR_Z" <<'PY'
import math
import sys

def f(x):
    try:
        return float(x)
    except Exception:
        return None

enb_x, enb_y, enb_z = map(f, sys.argv[1:4])
ue_x, ue_y, ue_z = map(f, sys.argv[4:7])
dist = f(sys.argv[7])
mode = sys.argv[8] if len(sys.argv) > 8 else "from_tx"
dir_x = f(sys.argv[9]) if len(sys.argv) > 9 and sys.argv[9] != "" else None
dir_y = f(sys.argv[10]) if len(sys.argv) > 10 and sys.argv[10] != "" else None
dir_z = f(sys.argv[11]) if len(sys.argv) > 11 and sys.argv[11] != "" else None

if None in (enb_x, enb_y, enb_z, ue_x, ue_y, ue_z, dist):
    sys.exit(2)

if dir_x is None or dir_y is None or dir_z is None:
    vx, vy, vz = ue_x - enb_x, ue_y - enb_y, ue_z - enb_z
else:
    vx, vy, vz = dir_x, dir_y, dir_z

norm = math.sqrt(vx * vx + vy * vy + vz * vz)
if norm == 0:
    sys.exit(3)

ux, uy, uz = vx / norm, vy / norm, vz / norm

if mode == "from_start":
    base_x, base_y, base_z = ue_x, ue_y, ue_z
else:
    base_x, base_y, base_z = enb_x, enb_y, enb_z

out_x = base_x + ux * dist
out_y = base_y + uy * dist
out_z = base_z + uz * dist

print(f"{out_x:.6f} {out_y:.6f} {out_z:.6f}")
PY
)"
      read -r UE_RUN_X UE_RUN_Y UE_RUN_Z <<< "$UE_POS"
      RUN_EXTRA_ARGS="--enbX=${ENB_X} --enbY=${ENB_Y} --enbZ=${ENB_Z} --ueX=${UE_RUN_X} --ueY=${UE_RUN_Y} --ueZ=${UE_RUN_Z} ${RUN_EXTRA_ARGS}"
    fi

    OUT_PREFIX="${OUT_PREFIX_BASE}_d${d}_r${r}"
    ./ns3 run "simple-sionna-example --sionna=${SIONNA_MODE} --flowmon=0 --local_machine=1 --tap=1 --tapName=${TAP_NAME_RUN} --tapMode=${TAP_MODE} --simTime=${SIM_TIME} --shutdownSionna=0 --distance=${d} --outPrefix=${OUT_PREFIX} ${RUN_EXTRA_ARGS}" &
    ns3pid=$!

  # Wait for tap to appear
  for _ in {1..20}; do
    if ip link show "$TAP_NAME_RUN" >/dev/null 2>&1; then
      break
    fi
    sleep 0.5
  done

    if ! ip link show "$TAP_NAME_RUN" >/dev/null 2>&1; then
      echo "Tap ${TAP_NAME_RUN} not found; marking loss and continuing."
      echo "${d},${r},0,0,100,,,,${MODE_LABEL}" >> "$OUT"
      if [ "$DEBUG" = "1" ]; then
        DEBUG_FILE="${DEBUG_DIR}/d${d}_r${r}.log"
        {
          echo "=== distance=${d} run=${r} (tap missing) ==="
          date
          echo "TAP_NAME=${TAP_NAME_RUN} TAP_MODE=${TAP_MODE}"
          ip link show "$TAP_NAME_RUN" 2>&1 || true
          ip -o -4 addr show dev "$TAP_NAME_RUN" 2>&1 || true
          ip route show 2>&1 | grep -E "7.0.0.0/8|${TAP_NAME_RUN}" || true
          ss -lunp 2>&1 | grep 8103 || true
        } >> "$DEBUG_FILE"
      fi
      terminate_ns3 "$ns3pid"
      if [ "$TAP_MODE" = "ConfigureLocal" ]; then
        sudo ip link delete "$TAP_NAME_RUN" 2>/dev/null || true
      fi
      sleep 1
      continue
    fi

  # Wait for IPv4 address to be configured (ConfigureLocal)
  for _ in {1..20}; do
    if ip -o -4 addr show dev "$TAP_NAME_RUN" | grep -q "${TAP_ADDR%%/*}"; then
      break
    fi
    sleep 0.5
  done

  sudo ip route replace 7.0.0.0/8 via "$PGW_IP" dev "$TAP_NAME_RUN"

  if [ "$PING_DELAY" != "0" ]; then
    sleep "$PING_DELAY"
  fi

    if [ "$DEBUG" = "1" ]; then
      DEBUG_FILE="${DEBUG_DIR}/d${d}_r${r}.log"
      {
        echo "=== distance=${d} run=${r} (pre-ping) ==="
        date
        echo "TAP_NAME=${TAP_NAME_RUN} TAP_MODE=${TAP_MODE}"
        ip link show "$TAP_NAME_RUN" 2>&1 || true
        ip -o -4 addr show dev "$TAP_NAME_RUN" 2>&1 || true
        ip route show 2>&1 | grep -E "7.0.0.0/8|${TAP_NAME_RUN}" || true
        ss -lunp 2>&1 | grep 8103 || true
      } >> "$DEBUG_FILE"
    fi

  ping_out="$(ping -I "$TAP_NAME_RUN" -w "$PING_SECS" "$UE_IP" || true)"

    if [ "$DEBUG" = "1" ]; then
      DEBUG_FILE="${DEBUG_DIR}/d${d}_r${r}.log"
      {
        echo "--- ping output ---"
        echo "$ping_out"
        echo ""
      } >> "$DEBUG_FILE"
    fi

  tx="$(echo "$ping_out" | grep -Eo "[0-9]+ packets transmitted" | awk '{print $1}')"
  rx="$(echo "$ping_out" | grep -Eo "[0-9]+ received" | awk '{print $1}')"
  loss="$(echo "$ping_out" | grep -Eo "[0-9.]+% packet loss" | awk -F'%' '{print $1}')"
  rtt_line="$(echo "$ping_out" | grep -E "rtt min/avg/max" || true)"
  if [ -n "$rtt_line" ]; then
    avg="$(echo "$rtt_line" | awk -F' = ' '{print $2}' | awk -F'/' '{print $2}')"
  else
    avg=""
  fi

  # Try to let ns-3 finish naturally so summary files are written
    if ! wait_ns3 "$ns3pid" "$NS3_WAIT"; then
      terminate_ns3 "$ns3pid"
    fi

    # Extract Sionna metrics from summary CSV (if present)
    summary_file="${OUT_PREFIX}_summary.csv"
    if [ -f "$summary_file" ]; then
      summary_line="$(tail -n 1 "$summary_file")"
      distance_actual="$(echo "$summary_line" | awk -F',' '{print $2}')"
      sionna_pathgain="$(echo "$summary_line" | awk -F',' '{print $8}')"
      sionna_delay="$(echo "$summary_line" | awk -F',' '{print $9}')"
      sionna_los="$(echo "$summary_line" | awk -F',' '{print $10}')"
    else
      distance_actual=""
      sionna_pathgain=""
      sionna_delay=""
      sionna_los=""
    fi

    echo "${d},${r},${tx:-0},${rx:-0},${loss:-100},${avg},${distance_actual},${sionna_pathgain},${sionna_delay},${sionna_los},${MODE_LABEL}" >> "$OUT"

    if [ "$CLEAN_TAP" = "1" ]; then
      sudo ip link delete "$TAP_NAME_RUN" 2>/dev/null || true
    fi

    sleep 1
  done
done

echo "Wrote ${OUT}"
