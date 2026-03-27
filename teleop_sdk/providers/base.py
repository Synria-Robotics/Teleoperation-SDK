from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseTeleopProvider(ABC):
    """Interface for human/base-action providers."""

    action_dim: int = 7

    def reset(self) -> None:
        """Reset episode-local state."""

    @abstractmethod
    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """Return one normalized teleop action in ``[-1, 1]``."""

    def peek_next_action(self) -> np.ndarray:
        """Return the next teleop action without advancing state."""
        return np.zeros(self.action_dim, dtype=np.float32)

    def is_active(self) -> bool:
        """Whether teleoperation is active this step."""
        return True

    def close(self) -> None:
        """Release provider resources."""


class BaseResidualPolicy(ABC):
    """Interface for residual-action policies."""

    action_dim: int = 7

    def reset(self) -> None:
        """Reset policy state before a new episode."""

    @abstractmethod
    def get_action(self, obs: dict[str, np.ndarray], base_action: np.ndarray) -> np.ndarray:
        """Return one residual action in ``[-1, 1]``."""

    def get_base_scale(self, obs: dict[str, np.ndarray], base_action: np.ndarray) -> np.ndarray:
        """Optional multiplicative gating applied to the human/base action.

        Policies that want stronger shared-control can override this to
        selectively attenuate parts of the human action near task-critical
        regions. The default behavior leaves the operator unchanged.
        """
        return np.ones(self.action_dim, dtype=np.float32)
