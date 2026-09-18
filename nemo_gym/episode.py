# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire contracts for environment servers."""

import re
from typing import Annotated, Generic, Literal, TypeVar

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator
from typing_extensions import Self

from nemo_gym.base_resources_server import (
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    MCPServerMetadata,
)
from nemo_gym.rollout_observability import AgentObservationBundle


class EpisodeId(BaseModel):
    """Identify one physical attempt of a logical rollout."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rollout_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    attempt: int = Field(default=0, ge=0)

    @field_validator("rollout_id")
    @classmethod
    def reserve_attempt_suffix(cls, rollout_id: str) -> str:
        """Keep the derived capture key injective without changing existing keys."""
        if re.search(r"-a[1-9][0-9]*$", rollout_id):
            raise ValueError("rollout_id must not end with the reserved attempt suffix '-a<N>'")
        return rollout_id

    @property
    def capture_key(self) -> str:
        """Return the attempt-qualified key used by capture routes."""
        return self.rollout_id if self.attempt == 0 else f"{self.rollout_id}-a{self.attempt}"


class TaskId(BaseModel):
    """Identify durable task content."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    taskset: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    revision: str | None = None


class EpisodeFailure(BaseModel):
    """Describe a handled episode failure."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(max_length=2000)
    terminal: bool = Field(description="Whether rollout collection must not attempt this episode again.")


TaskInputT = TypeVar("TaskInputT", bound=BaseModel)
EpisodeResultT = TypeVar("EpisodeResultT")


class MaterializedTask(BaseModel, Generic[TaskInputT]):
    """Carry durable task identity and protocol-shaped task input."""

    model_config = ConfigDict(extra="forbid")

    task_id: TaskId
    task_input: TaskInputT


class BaseEpisodeRequest(BaseModel, Generic[TaskInputT]):
    """Carry environment-neutral identity and typed task input."""

    model_config = ConfigDict(extra="forbid")

    episode_id: EpisodeId
    task: MaterializedTask[TaskInputT]


class BaseEpisodeResponse(BaseModel, Generic[EpisodeResultT]):
    """Return either a typed result or a handled failure."""

    model_config = ConfigDict(extra="forbid")

    episode_id: EpisodeId
    task_id: TaskId
    result: EpisodeResultT | None = None
    failure: EpisodeFailure | None = None

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if (self.result is None) == (self.failure is None):
            raise ValueError("exactly one of result or failure is required")
        return self


class DirectHTTPToolAccess(BaseModel):
    """Connect a trusted Python agent to typed tool routes."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["direct_http"] = "direct_http"
    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    required: bool
    base_url: AnyHttpUrl
    cookies: dict[str, str] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)


class MCPStreamableHTTPConnection(BaseModel):
    """Connect to an MCP server over an absolute HTTP endpoint."""

    model_config = ConfigDict(extra="forbid")

    transport: Literal["streamable_http"] = "streamable_http"
    url: AnyHttpUrl
    headers: dict[str, str] = Field(default_factory=dict)


class MCPStdioConnection(BaseModel):
    """Start an MCP server as a child process without invoking a shell."""

    model_config = ConfigDict(extra="forbid")

    transport: Literal["stdio"] = "stdio"
    command: str = Field(min_length=1)
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None


MCPConnection = Annotated[
    MCPStreamableHTTPConnection | MCPStdioConnection,
    Field(discriminator="transport"),
]


class MCPToolAccess(BaseModel):
    """Describe one named MCP server that an agent adapter must configure.

    Names are session-wide identifiers. Agent session requests reject duplicate names,
    and adapters must also reject collisions with their static tool configuration.
    Failure to establish a required server fails session seeding; an optional server may
    be skipped. On agent-session close, adapters disconnect HTTP clients without stopping
    the remote server and terminate stdio processes they started.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["mcp"] = "mcp"
    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    required: bool
    connection: MCPConnection


ToolAccess = Annotated[
    DirectHTTPToolAccess | MCPToolAccess,
    Field(discriminator="kind"),
]


class DirectSandboxConnection(BaseModel):
    """Reconnect through a named top-level sandbox-provider configuration."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["direct"] = "direct"
    provider_config_ref: str
    descriptor: dict[str, JsonValue]


# TODO: Add a sandbox-server connection after that server and its lease contract exist.


class SandboxAccess(BaseModel):
    """Describe borrower access to an owner-managed sandbox."""

    model_config = ConfigDict(extra="forbid")

    connection: DirectSandboxConnection
    workdir: str


class ResourcesSeedSessionRequest(BaseSeedSessionRequest):
    """Initialize resources-server state for one episode."""

    model_config = ConfigDict(extra="forbid")

    episode_id: EpisodeId
    task_id: TaskId
    task_data: dict[str, JsonValue]


class ResourcesSeedSessionResponse(BaseSeedSessionResponse):
    """Return resources state and optional agent access."""

    model_config = ConfigDict(extra="forbid")

    resources_session_id: str
    resources_tools: MCPServerMetadata | None = None
    sandbox_access: SandboxAccess | None = None


VerificationInputT = TypeVar("VerificationInputT")


class BaseEpisodeResourcesVerifyRequest(BaseModel, Generic[VerificationInputT]):
    """Carry typed environment output to a resources server."""

    model_config = ConfigDict(extra="forbid")

    episode_id: EpisodeId
    task_id: TaskId
    verification_input: VerificationInputT


class ResourcesCloseSessionRequest(BaseModel):
    """Close resources-server state."""

    model_config = ConfigDict(extra="forbid")

    resources_session_id: str
    episode_id: EpisodeId


class ResourcesCloseSessionResponse(BaseModel):
    """Confirm resources-server state was closed."""

    model_config = ConfigDict(extra="forbid")

    resources_session_id: str


class AgentSeedSessionRequest(BaseModel):
    """Initialize agent-server state for one episode."""

    model_config = ConfigDict(extra="forbid")

    episode_id: EpisodeId
    task_id: TaskId
    tool_accesses: list[ToolAccess] = Field(default_factory=list)
    sandbox_access: SandboxAccess | None = None

    @field_validator("tool_accesses")
    @classmethod
    def require_unique_tool_names(cls, tool_accesses: list[ToolAccess]) -> list[ToolAccess]:
        names = [access.name for access in tool_accesses]
        if len(names) != len(set(names)):
            raise ValueError("tool access names must be unique within an agent session")
        return tool_accesses


class AgentSeedSessionResponse(BaseModel):
    """Return the worker-local agent session identifier."""

    model_config = ConfigDict(extra="forbid")

    agent_session_id: str


class AgentCloseSessionRequest(BaseModel):
    """Close agent-server state."""

    model_config = ConfigDict(extra="forbid")

    agent_session_id: str
    episode_id: EpisodeId


class AgentCloseSessionResponse(BaseModel):
    """Confirm closure and return captured observations."""

    model_config = ConfigDict(extra="forbid")

    agent_session_id: str
    agent_observations: AgentObservationBundle | None = None
    resources_cookies: dict[str, str] | None = None
