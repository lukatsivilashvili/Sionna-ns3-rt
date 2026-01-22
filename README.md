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

## Build ns3-rt

Run from the `ns3-rt` directory:

```bash
cd ns3-rt
./ns3 configure --disable-python --enable-examples
./ns3 build
```

## Run the example

1) Start the Sionna RT server (from `ns3-rt/src/sionna`, so the scenario path resolves):

```bash
cd ns3-rt/src/sionna
python sionna_v1_server_script_static_scene.py --local-machine --frequency=2.1e9 --path-to-xml-scenario=scenarios/Cnam_Scenario/cnam_scene.xml
```

2) In another terminal, run the ns-3 example (from `ns3-rt`):

```bash
cd ns3-rt
./ns3 run simple-sionna-example
```

## Example source location

The runner file for the example is located at:

`ns3-rt/src/sionna/examples/simple-sionna-example.cc`
