#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Prepare once on Linux x86-64 glibc; mount read-only at /opt/gym-cli.
set -euo pipefail
: "${DEPS_DIR:?Set DEPS_DIR to a dedicated absolute runtime directory}"
[[ "$DEPS_DIR" = /* && "$DEPS_DIR" != / ]] || { echo "DEPS_DIR must be a dedicated absolute directory" >&2; exit 1; }
[[ "$(uname -s)" = Linux && "$(uname -m)" = x86_64 ]] || { echo "Build on Linux x86-64" >&2; exit 1; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../responses_api_agents/anyswe_agent/setup_scripts/_portable_python.sh"
[[ "$ARCH" = x86_64-unknown-linux-gnu ]] || { echo "This builder is glibc-only; musl artifacts need separate validation" >&2; exit 1; }
mkdir -p "$DEPS_DIR"
exec 9>"$DEPS_DIR/.prepare.lock"
flock 9
install_portable_python
NODE_VERSION=22.19.0
if [[ "$("$DEPS_DIR/bin/node" --version 2>/dev/null || true)" != "v$NODE_VERSION" ]]; then
    node_tmp=$(mktemp -d "$DEPS_DIR/.node.XXXXXX")
    archive="node-v${NODE_VERSION}-linux-x64.tar.xz"
    curl -fsSL --retry 3 "https://nodejs.org/dist/v${NODE_VERSION}/$archive" -o "$node_tmp/$archive"
    curl -fsSL --retry 3 "https://nodejs.org/dist/v${NODE_VERSION}/SHASUMS256.txt" -o "$node_tmp/SHASUMS256.txt"
    (cd "$node_tmp" && sha256sum --check --ignore-missing SHASUMS256.txt)
    tar xJf "$node_tmp/$archive" -C "$DEPS_DIR" --strip-components=1
fi
export PATH="$DEPS_DIR/bin:$PATH"
npm install -g --prefix "$DEPS_DIR" \
    "@earendil-works/pi-coding-agent@0.80.2" "openclaw@2026.6.11" "@openai/codex@0.144.4"
# Separate launchers call this runtime's Node without changing the PATH that
# benchmark terminal tools inherit. Do not replace the npm-created symlinks.
"$DEPS_DIR/bin/python3" -I - "$DEPS_DIR" <<'PY'
import pathlib
import shlex
import sys
root = pathlib.Path(sys.argv[1]).resolve()
launchers = root / 'launchers'
launchers.mkdir(exist_ok=True)
for name in ('pi', 'openclaw', 'codex'):
    target = (root / 'bin' / name).resolve().relative_to(root)
    script = '#!/bin/sh\nset -eu\n'
    script += 'root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)\n'
    script += 'exec "$root/bin/node" "$root"/' + shlex.quote(str(target)) + ' "$@"\n'
    path = launchers / name
    path.write_text(script)
    path.chmod(0o755)
PY
"$DEPS_DIR/launchers/pi" --version
"$DEPS_DIR/launchers/openclaw" --version
"$DEPS_DIR/launchers/codex" --version
