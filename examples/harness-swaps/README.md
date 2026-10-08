# Two-Way Swaps, One Run Command

Choose a benchmark and a harness. Run a task. Change either choice and run again.

This is a **demo notebook/helper**, not a new built-in Gym command. It uses the
normal Gym CLI underneath. The branch includes the Pi/TB2.1 draft stack at
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
authentication enabled. Run the setup cells, then the three demo cells. First
startup installs per-server dependencies and each sandbox installs its harness;
rehearse before recording. Every invocation creates new artifacts and closes
its own services/containers. It does not stop your model endpoint or other runs.

## The demo

The notebook exposes just two selectors:

```python
run(harness="hermes", benchmark="swe-pro")
run(harness="pi", benchmark="swe-pro")  # Same task, model and verifier.
run(harness="pi", benchmark="tb21")    # Same harness and model.
```

The identical command is also usable without Jupyter:

```bash
alias demo='python examples/harness-swaps/run.py'
demo --harness hermes --benchmark swe-pro
demo --harness pi --benchmark swe-pro
demo --harness pi --benchmark tb21
```

`demo` is a shell alias for this example helper, not a built-in Gym command.
It defaults to the files created by `prepare.py`; use `--input FILE` to override.
Add `--limit 2` for two tasks. `--task-id ID` selects an exact prepared task;
repeat it for two IDs. Outputs go into a fresh directory under
`results/harness-swaps/`. Keep raw logs, model captures, rollouts, health reports,
`summary.json` and `cleanup.json` private. A valid reward of zero is different
from incomplete verification. Read the health verdict, not just its file count.
Unexpected missing rows, incomplete verification or leftover containers need
investigation before presenting the run as functional.

## How it works — separate from the user demo

`run.py` composes three ordinary YAML files: shared model/provider settings,
the benchmark's Resources config and the harness's Agent config. A generated
run-local config binds the two names; no harness or verifier Python is rewritten.
The flat prepared rows enter the native EnvironmentServer through
`single_agent_turn_legacy`, which adapts row format, not execution to a host CLI.
Both harnesses run inside the Resources-owned task sandbox.

The helper starts `gym env start --config RUN_CONFIG`, waits for `/readyz`, then
uses `gym eval run --no-serve --config RUN_CONFIG --agent AGENT --input INPUT ...`.
The exact commands are saved in `manifest.json`. `--no-serve` is intentional:
this source's default E2E route prepares its declared dataset instead of simply
collecting the selected input. No custom HTTP collector or fake reward is used.
The hosted-model adapter applies a bounded 60-second retry delay for transient
provider errors. The rehearsal's earlier vLLM-adapter attempt exhausted its short
retries on NVIDIA HTTP 429. This config does not add an upstream Responses API
requirement: both harnesses call Chat Completions.

For these compatible pairs: **zero Python adapter edits per swap**. A new
harness still needs its Agent adapter; a new benchmark needs data preparation
and Resources setup/verification/cleanup. Reusing other compatible counterparts
then needs configuration and a real per-pair smoke, not another bespoke adapter.
Existing older pairs could already be config-only; do not claim universal past
LOC or time savings. Service-producing TB tasks have a known lifecycle caveat;
this regex-log demo does not certify all 89 TB2.1 tasks.
