"""Environment-side primitives for agent-native BEHAVIOR rollouts."""

from omnigibson.learning.agentic.controller_manifest import ControllerManifest
from omnigibson.learning.agentic.controller_manifest import ControllerManifestError
from omnigibson.learning.agentic.controller_manifest import ControllerSegment
from omnigibson.learning.agentic.controller_manifest import build_controller_manifest
from omnigibson.learning.agentic.controller_manifest import validate_runtime_action
from omnigibson.learning.agentic.controller_manifest import validate_r1pro_manifest
from omnigibson.learning.agentic.environment_session import AgenticEnvironmentSession
from omnigibson.learning.agentic.environment_session import EnvironmentStepResult
from omnigibson.learning.agentic.environment_session import SessionState
from omnigibson.learning.agentic.evaluator_state import EvaluatorSnapshotComponent
from omnigibson.learning.agentic.snapshot import AttributeSnapshotComponent
from omnigibson.learning.agentic.snapshot import CompositeSnapshot
from omnigibson.learning.agentic.snapshot import CompositeSnapshotManager
from omnigibson.learning.agentic.snapshot import RestoreReport
from omnigibson.learning.agentic.snapshot import SnapshotCompatibilityError
from omnigibson.learning.agentic.snapshot import SnapshotMetadata

__all__ = [
    "AttributeSnapshotComponent",
    "AgenticEnvironmentSession",
    "CompositeSnapshot",
    "CompositeSnapshotManager",
    "ControllerManifest",
    "ControllerManifestError",
    "ControllerSegment",
    "EnvironmentStepResult",
    "EvaluatorSnapshotComponent",
    "RestoreReport",
    "SnapshotCompatibilityError",
    "SnapshotMetadata",
    "SessionState",
    "build_controller_manifest",
    "validate_runtime_action",
    "validate_r1pro_manifest",
]
