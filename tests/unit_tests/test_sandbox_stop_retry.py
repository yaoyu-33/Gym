# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock

import pytest

from nemo_gym.sandbox import AsyncSandbox, SandboxHandle
from tests.unit_tests.test_sandbox_connect import FakeConnectableProvider


@pytest.mark.asyncio
async def test_owner_stop_retries_failed_remote_close_without_discarding_provider():
    provider = FakeConnectableProvider()
    provider.close = AsyncMock(side_effect=[RuntimeError("still running"), None])
    provider.aclose = AsyncMock()
    box = AsyncSandbox(provider, owns_provider=True)
    box._handle = SandboxHandle(sandbox_id="owned", provider_name=provider.name, raw="owned")
    box._stopped = False
    with pytest.raises(RuntimeError, match="still running"):
        await box.stop()
    assert not box._closed and not box._stopped
    provider.aclose.assert_not_awaited()
    await box.stop()
    assert provider.close.await_count == 2
    provider.aclose.assert_awaited_once()
    await box.stop()
    assert provider.close.await_count == 2
