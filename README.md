# Sionna RT + ns-3 (ns3-rt) — Build & Run

  

This repository contains a fork of ns-3 (`ns3-rt`) integrated with NVIDIA Sionna RT.

Use the steps below to build ns-3 and run the `simple-sionna-example`.

  

## Dependencies

  

### ns3-rt build prerequisites (from ns-3 docs)

- C++ compiler: `g++` (>= 9) or `clang++`

- CMake: >= 3.10

- Build system: `make` (or `ninja`)

- Python 3: 3.10-3.12 (used by the `./ns3` helper even when Python bindings are disabled)

- Git, tar, bunzip2 (common tools for source-based workflows)

  

### Sionna RT runtime prerequisites

- Python 3 + `pip`

- Sionna RT v1.0.1 or newer

- TensorFlow:

- CPU: `tensorflow`

- GPU: `tensorflow[and-cuda]` + appropriate GPU drivers

- LLVM (required by Sionna RT / Mitsuba 3)

  

## Initial setup (first time on a new machine)

  

1. Clone the repository and enter it:

  

```bash

git clone <YOUR_REPO_URL>

cd Sionna-ns3-rt

```

  

2. Use Python 3.12.2 for the ns-3 helper script (recommended):

  

```bash

cd ns3-rt

pyenv install 3.12.2 # if not already installed

pyenv local 3.12.2

```

  

3. Build ns3-rt from scratch:

  

```bash

cd ns3-rt

./ns3 configure --disable-python --enable-examples

./ns3 build

```

  

## Build ns3-rt (after code changes)

  

Run from the `ns3-rt` directory:

  

```bash

cd ns3-rt

./ns3 configure --disable-python --enable-examples

./ns3 build

```

  

## End-to-end run guide (RT + legacy + report)

  

Use two terminals.

  

**Terminal 1 = Sionna server**

**Terminal 2 = ns-3 runs**

  

### 0) Clean stale taps (Terminal 2, optional)

  

```bash

cd ns3-rt

sudo ip link delete thetap 2>/dev/null || true

for d in 50 100 200 300 400 500; do sudo ip link delete thetap${d} 2>/dev/null || true; done

```

  

### 1) Start Sionna server (Terminal 1)

  

```bash

cd ns3-rt/src/sionna

python sionna_v1_server_script_static_scene.py \
--local-machine \
--frequency=2.1e9 \
--path-to-xml-scenario=scenarios/Cnam_Scenario/cnam_scene.xml \
--rt-profile=high \
--gpu=0

```

  

**Server flags used**

- `--path-to-xml-scenario`: Scene XML file.

- `--frequency`: Carrier frequency in Hz.

- `--local-machine`: Bind on `127.0.0.1` (local).

- `--rt-profile`: Ray tracing preset (`medium`, `high`, `ultra`).

- `--gpu`: GPU count (`0` = CPU only).

  

### 2) Build ns-3 example (Terminal 2)

  

```bash

cd ns3-rt

./ns3 build simple-sionna-example

```

  

### 3) Run Sionna RT sweep (Terminal 2)

  

Uses the provided coordinates and walks away from the transmitter along the TX→RX line.

  

```bash

cd ns3-rt

SIONNA_MODE=1 OUT_DIR=results/rt \
DISTANCES="50 100 200 300 400 500" RUNS_PER_DIST=3 \
SIM_TIME=60 PING_SECS=10 \
ENB_X=49 ENB_Y=-17 ENB_Z=12 \
UE_X=32 UE_Y=-9 UE_Z=0 \
WALK_MODE=from_tx \
TAP_MODE=UseLocal TAP_NAME_TEMPLATE="thetap%d" \
./scripts/ping_sionna_sweep.sh

```

  

**Sweep variables used**

- `SIONNA_MODE`: `1` for Sionna RT, `0` for legacy baseline

- `OUT_DIR`: Output root directory

- `DISTANCES`: Space‑separated list of distances (meters)

- `RUNS_PER_DIST`: Number of runs per distance

- `SIM_TIME`: ns-3 simulation time (seconds)

- `PING_SECS`: Ping duration per run (seconds)

- `TAP_MODE`: `UseLocal` in this guide

- `TAP_NAME_TEMPLATE`: Tap name template per distance

- `ENB_X`, `ENB_Y`, `ENB_Z`: Transmitter (eNB) start position

- `UE_X`, `UE_Y`, `UE_Z`: Receiver (UE) start position

- `WALK_MODE`: `from_tx` (distance from transmitter)

  

### 4) Stop Sionna server (Terminal 1)

  

Press `Ctrl+C`.

  

### 5) Run legacy (baseline) sweep (Terminal 2)

  

Same walk path, no Sionna server required.

  

```bash

cd ns3-rt

SIONNA_MODE=0 OUT_DIR=results/legacy \
DISTANCES="50 100 200 300 400 500" RUNS_PER_DIST=3 \
SIM_TIME=60 PING_SECS=10 \
ENB_X=49 ENB_Y=-17 ENB_Z=12 \
UE_X=32 UE_Y=-9 UE_Z=0 \
WALK_MODE=from_tx \
TAP_MODE=UseLocal TAP_NAME_TEMPLATE="thetap%d" \
./scripts/ping_sionna_sweep.sh

```

  

### 6) Generate combined report + plots (Terminal 2)

  

Outputs:

- `results/rt/…` for RT

- `results/legacy/…` for legacy

- `results/combined/…` for combined

  

```bash

cd ns3-rt

python3 scripts/ping_sweep_report.py \
--csv results/rt/ping_sweep.csv \
--csv results/legacy/ping_sweep.csv \
--out-dir results/combined \
--use-actual-distance \
--per-distance-plots

```

  

**Report flags used**

- `--csv`: Input CSV file(s). Repeated twice (RT + legacy).

- `--out-dir`: Output folder for combined report/plots.

- `--use-actual-distance`: Use `distance_actual_m`.

- `--per-distance-plots`: Generate per‑distance plots.

  

### Notes / knobs

- To shorten runs further: `SIM_TIME=40 PING_SECS=8 RUNS_PER_DIST=2`

- To log Sionna pathgain/delay/LOS: `EXTRA_NS3_ARGS="--logSionnaMetrics=1"` (Sionna server must be running)