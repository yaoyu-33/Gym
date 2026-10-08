# Two-Way Swaps, One Run Command

Choose a benchmark and a harness. Run a task. Change either choice and run again.

The notebook runs the **actual Gym CLI**, with ordinary example YAML configs.
There is no custom run alias in the demo. The branch includes the Pi/TB2.1 draft stack at
`0e912e822804f09ccfb6feb131b25ea7b4bf7936`; these examples are not all on `main` yet.
See [validation and known limitations](VALIDATION.md) for what was actually run.

## Before recording

Use a dedicated Linux checkout with Docker access, Python 3.13.14+, `uv`, and a
Chat Completions model endpoint reachable from this host. Sandboxes need network
access to Gym and runtime package registries. No GPU is required on the demo host
when inference is hosted elsewhere.

```bash
git clone --branch codex/harness-benchmark-swap-demo https://github.com/yaoyu-33/Gym.git
cd Gym
export UV_CACHE_DIR="$PWD/cache/uv"
uv sync --extra dev --extra sandbox
uv pip install --python .venv/bin/python jupyterlab
source .venv/bin/activate
```

Privately set `DEMO_MODEL_URL` (ending in `/v1`), `DEMO_MODEL_NAME`,
`DEMO_MODEL_KEY`, and `DEMO_GYM_HOST` (this host's address reachable from Docker).
The notebook can prompt for missing values; never paste a real key into a cell.
Do not expose the notebook server or Gym ports to untrusted networks.

Prepare data once, outside the recording:

```bash
python benchmarks/swebench/pro/prepare.py
git clone https://github.com/harbor-framework/terminal-bench-2-1 \
  benchmarks/terminal_bench_2_1/terminal-bench-2-1
git -C benchmarks/terminal_bench_2_1/terminal-bench-2-1 checkout --detach \
  7131e4375048a0e408a8fb404b5f499d726b695b
python benchmarks/terminal_bench_2_1/prepare.py
python examples/harness-swaps/prepare.py
```

Preparation can download large datasets/images. If you already have prepared
inputs, pass `--swe-source FILE --tb-source FILE` to the demo preparation command.
Use `--swe-task ID` and `--tb-task ID` to choose other tasks (repeat for two).
The selector refuses to overwrite earlier inputs. Keep the TB task directory alongside its prepared rows: the verifier
needs its assets. Select 1–2 unchanged rows before recording; a first row is not
guaranteed to run quickly. The rehearsal uses the same Ansible SWE-Pro task for
both harnesses and `terminal-bench/regex-log` for the benchmark swap. Do not
use this small, selected demo to compare accuracy.

```bash
jupyter lab examples/harness-swaps/demo.ipynb --ip 127.0.0.1 --no-browser
```

Connect through SSH forwarding if running remotely. Leave Jupyter's token
authentication enabled. Run each setup cell, its Gym command, and its cleanup cell. First
startup installs per-server dependencies and each sandbox installs its harness;
rehearse before recording. Every invocation creates new artifacts and closes
its own services/containers. It does not stop your model endpoint or other runs.

## The demo

The visible notebook cells execute these commands directly:

```bash
gym eval run --no-serve \
  --config examples/harness-swaps/swe-pro.yaml \
  --agent hermes_agent --limit 1

gym eval run --no-serve \
  --config examples/harness-swaps/swe-pro.yaml \
  --agent pi_agent --limit 1

gym eval run --no-serve \
  --config examples/harness-swaps/tb21.yaml \
  --agent pi_agent --limit 1
```

Change `--agent` to swap the harness. Change `--config` to swap the benchmark.
The SWE config makes both harnesses available against the same Resources server;
the TB config includes Pi. The selected input file is declared in each config.
These are example configs, not new benchmark catalog aliases.

`--no-serve` requires running services. The notebook's setup cells start them
with `gym env start --config ...`; they do not execute or wrap the eval command.
Each setup uses a new results directory and an available Head port. Run the
matching cleanup cell before switching benchmark services or repeating a run.

Without Jupyter, after privately setting the model variables and activating the
virtual environment, prepare one run in a terminal:

```bash
export DEMO_RUN_ID="swe-pro-$(date +%Y%m%d-%H%M%S)"
export DEMO_RUN_DIR="$PWD/results/harness-swaps/$DEMO_RUN_ID"
export DEMO_HEAD_PORT=48977  # Choose an unused port.
mkdir -p results/harness-swaps
mkdir -m 700 "$DEMO_RUN_DIR"
gym env start --config examples/harness-swaps/swe-pro.yaml \
  > "$DEMO_RUN_DIR/services.log" 2>&1 &
DEMO_SERVICES_PID=$!
./scripts/wait_for_servers.sh "$DEMO_SERVICES_PID" "$DEMO_HEAD_PORT" 900
```

Run the matching `gym eval run` command above in that same terminal. Then stop
only this service process with `kill -INT "$DEMO_SERVICES_PID"`, then
`wait "$DEMO_SERVICES_PID"` for shutdown. Check for leftover containers with
`docker ps -a --filter "label=gym-swap-demo=$DEMO_RUN_ID"`. Do not prune other runs.
For the next run, create a fresh ID/directory and start the matching config again.
Do not reuse an output path: collection overwrites it by default.

Use `--input FILE` to override the prepared task file or `--limit 2` for two tasks.
Outputs go into `DEMO_RUN_DIR`. Keep raw logs, model captures, rollouts, health reports,
`summary.json` and `cleanup.json` private. A valid reward of zero is different
from incomplete verification. Read the health verdict, not just its file count.
Unexpected missing rows, incomplete verification or leftover containers need
investigation before presenting the run as functional.

## How it works — separate from the user demo

The example YAML composes shared model/provider settings, the benchmark's
Resources config and each harness's Agent config. Native sessions bind the
independent parts; no harness or verifier Python is rewritten.
The flat prepared rows enter the native EnvironmentServer through
`single_agent_turn_legacy`, which adapts row format, not execution to a host CLI.
Both harnesses run inside the Resources-owned task sandbox.

Setup starts `gym env start`, then the visible cell calls `gym eval run --no-serve`.
`--no-serve` is intentional:
this source's default E2E route prepares its declared dataset instead of simply
collecting the selected input. No custom HTTP collector or fake reward is used.
The hosted-model adapter applies a bounded 60-second retry delay for transient
provider errors. The rehearsal's earlier vLLM-adapter attempt exhausted its short
retries on NVIDIA HTTP 429. This config does not add an upstream Responses API
requirement: both harnesses call Chat Completions.
Per-call output limits belong to each harness: Hermes uses 8,192 and Pi uses
32,768. A shared model override must not silently replace those values. The first
Pi rehearsal hit an 8K limit mid-response; the subsequent real Pi runs use the
larger budget. These runs are not a matched-budget accuracy comparison.

`run.py` remains an optional rehearsal helper and provides shared result/cleanup
utilities for the notebook. Its old `demo` alias is not the presented interface.

For these compatible pairs: **zero Python adapter edits per swap**. A new
harness still needs its Agent adapter; a new benchmark needs data preparation
and Resources setup/verification/cleanup. Reusing other compatible counterparts
then needs configuration and a real per-pair smoke, not another bespoke adapter.
Existing older pairs could already be config-only; do not claim universal past
LOC or time savings. Service-producing TB tasks have a known lifecycle caveat;
this regex-log demo does not certify all 89 TB2.1 tasks.
