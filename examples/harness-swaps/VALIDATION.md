# Demo validation — October 8, 2026

Runtime source: `57aa37f8ea7ad951b7caa6dc116266c1651578b4`, on the Pi/TB2.1
preview stack at `0e912e822804f09ccfb6feb131b25ea7b4bf7936`.
Linux Docker host, Python 3.13.14, NVIDIA-hosted
`nvidia/nemotron-3-super-120b-a12b` through Chat Completions. Each run uses one
unchanged selected task; this is a workflow check, not an accuracy comparison.

## Rehearsal

| Pair | Run | Result |
|---|---|---|
| Hermes × SWE-Pro | `20261008-101746-hermes-swe-pro-744c53` | Reward 1.0; all 8 verifier tests passed; 78 matched tool calls/results; no leftover sandbox |
| Pi × SWE-Pro | Pending | Not yet verified in this demo checkout |
| Pi × TB2.1 | Pending | Not yet verified in this demo checkout |

Hermes collected one rollout with `evaluation_completed=true`,
`mask_sample=false`, `patch_applied=true`, `resolved=true`, no verifier error,
and CLI exit 0. It took approximately 31 minutes, including hosted-model rate
limit waits. The three-minute video will shorten waits, not imply a three-minute
benchmark runtime.
The agent observation is `incomplete` with a configured 30-minute execution
budget; the saved patch nevertheless passes all eight stock verifier tests.
Completed verification is not the same as a naturally finished agent conversation.

**Known health warning:** Hermes's response usage is still hard-coded to zero
in this source. Model captures contain token usage, so the health report flags
`rollout_token_count_mismatch`. The task's reward is valid, but this is not an
all-green trajectory/usage validation. This demo does not fix that harness issue.

## Checks and failed attempts

- 11 focused example tests passed on macOS and Linux; all-file pre-commit passed.
- Notebook code cells compile and `nbformat.validate` passes.
- Actual Linux Environment/Model config validation and dependency prefetch passed.
- A real Terminal help-command MP4 verified the ScreenCaptureKit recording path.
  This capture test is not benchmark evidence.
- Initial startup failed because the concurrency config lacked a queue timeout;
  the example now includes it and has a regression test.
- The first model run exhausted short retries on HTTP 429 and collected no rows.
  The example now uses the existing hosted-model adapter with bounded 60-second
  backoff. Failed logs were retained; they are not counted as successful runs.
- The shared uv cache had a permission failure; a new task-owned cache worked.

Raw model captures, benchmark outputs, credentials, and large recordings are not
committed. Each completed run retains its own `manifest.json`, `summary.json`,
`quality_summary.json`, `cleanup.json`, stock rollout, and private logs.

## Recording

Fresh actual-window benchmark captures are in progress. No final video is
published yet. The recording uses AppleScript to submit the real command to a
dedicated Terminal window and ScreenCaptureKit to capture that window to MP4.
No animations or replayed execution output are used.
