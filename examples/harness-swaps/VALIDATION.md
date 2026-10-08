# Demo validation — October 8, 2026

Base: Pi/TB2.1 preview stack `0e912e822804f09ccfb6feb131b25ea7b4bf7936`.
Initial helper `57aa37f8ea7ad951b7caa6dc116266c1651578b4`; budget correction
`355c668da62cb194b47a36be69e14c9da777708c`. Linux Docker host, Python 3.13.14,
NVIDIA-hosted `nvidia/nemotron-3-super-120b-a12b` through Chat Completions.
This is one selected task per pair, not an accuracy comparison.

## Completed helper-driven runs

These runs invoked the real Gym CLI under a helper. The user subsequently asked
to show the literal CLI instead; fresh direct-command captures are in progress.
Do not describe the older alias-based video as the final requested recording.

| Pair | Run | Reward | Verifier | Matched tool calls/results | Elapsed |
|---|---|---:|---|---:|---:|
| Hermes × SWE-Pro | `20261008-104900-hermes-swe-pro-2d4963` | 1.0 | 8/8 tests passed | 81 | 1,850 s |
| Pi × SWE-Pro | `20261008-114445-pi-swe-pro-0c28ea` | 0.0 | 3/8 tests passed | 82 | 1,864 s |
| Pi × TB2.1 | `20261008-121605-pi-tb21-9480eb` | 1.0 | Verification complete | 7 | 223 s |

All three collected one rollout, completed stock verification with CLI exit 0,
remained unmasked, had no verifier error, and cleaned up with no leftover owned
containers. The Pi/SWE zero is a real solution failure, not an infrastructure
failure. It is retained. Both SWE runs used the identical Ansible task and input
hash `350f66195db074bc7345d1458abf4e44d730a4c702c515604ccf0164707b4b99`.
TB used `terminal-bench/regex-log` from dataset commit
`7131e4375048a0e408a8fb404b5f499d726b695b`.

**Agent completion is separate from verification.** Both SWE agent observations
are `incomplete` at the configured 30-minute execution budget; their saved
patches were still graded. Pi/TB's agent observation is `completed`.

**Health limits:** Hermes reports zero usage despite nonzero captured model
usage, causing `rollout_token_count_mismatch`. A recovered hosted 429 also leaves
an observation gap. The two corrected-budget Pi runs have healthy verdicts for
the available checks and no captured model errors. Pi still records gaps for
unavailable reasoning/cache details and subagent hierarchy. Healthy available
checks do not imply complete observability or a correct model solution.

## Actual CLI recording

The notebook now executes literal `gym eval run --no-serve` cells. Ordinary YAML
configs replace the custom demo selectors; setup and cleanup remain separate.
The SWE config makes both harnesses available: `--agent` selects their native
session routes. `tb21.yaml` selects the other benchmark with Pi unchanged.
No agent, model, verifier or core CLI code is changed by these examples.

Native-command config/routing tests pass on macOS and Linux. Fresh Hermes run
`20261008-123605-native-hermes-swe-pro` completed verification and clean shutdown
in 312 seconds: reward 0.0, eight matched tool calls/results, no mask or verifier
error. The agent status is failed: its last model response hit the configured
8,192-token limit with no usable text or tools. Health flags both that truncated
response and the known zero-usage mismatch. The stock test report is empty;
do not call this a passing harness/accuracy check. This result is not replaced
with the earlier successful helper run. The two direct Pi captures are pending.

## Checks and earlier attempts

- 15 focused example tests pass on macOS and Linux, including actual config
  parsing, Environment config validation and per-agent routing assertions.
- Notebook schema validates and all cells compile, including Bash magics, in
  the workstation's Jupyter environment. All-file pre-commit passes, including
  the staged native configs.
- Earlier Linux Environment/Model validation and all three dependency prefetch
  commands passed.
- An actual 12-second Terminal help-command MP4 verified ScreenCaptureKit.
  It is recording-system validation, not benchmark evidence.
- Initial startup lacked the concurrency queue timeout; the example now sets
  60 seconds and tests the config contract.
- The initial model run exhausted short retries on HTTP 429, collecting no rows.
  The existing hosted-model adapter with bounded 60-second backoff resolved
  that setup issue. The failed run remains in the evidence history.
- Shared uv-cache permissions failed; a task-owned cache succeeded.
- The first Pi/SWE run hit a shared 8K per-call ceiling and received an empty
  length-limited response. It scored zero (6/8 tests). The correction removed
  the model-level override and gives Pi 32K; Hermes stays at 8K. Both corrected
  Pi runs above were recorded anew. These are not matched-budget comparisons.
- Bash noclobber initially blocked the recorder's status marker after a genuine
  completed run. Only the task-owned marker now uses `>|`; benchmark output was
  not changed. Full raw footage and the diagnosis are retained privately.

Raw model captures, tasks, credentials, private logs and large recordings are not
committed. Each run retains its manifest, summaries, stock rollout and cleanup
evidence. ScreenCaptureKit captures only the dedicated real Terminal window.
The short video cuts waits and labels edits; no animated/replayed execution is
used. The earlier helper runs took minutes to half an hour, not three minutes.
The selected regex-log task does not certify service-producing TB tasks or all
89 tasks in this preview dataset.
