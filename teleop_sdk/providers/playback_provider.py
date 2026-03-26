from __future__ import annotations

import numpy as np

from teleop_sdk.providers.base import BaseTeleopProvider


class PlaybackTeleopProvider(BaseTeleopProvider):
    """Replay human teleop actions from a prerecorded array."""

    def __init__(self, actions: np.ndarray, *, loop: bool = False):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != self.action_dim:
            raise ValueError(f"Expected actions with shape [T, {self.action_dim}], got {actions.shape}")
        self.actions = np.clip(actions, -1.0, 1.0)
        self.loop = loop
        self._index = 0

    def reset(self) -> None:
        self._index = 0

    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        del obs
        action = self.peek_next_action()
        if self._index < len(self.actions) - 1:
            self._index += 1
        elif self.loop and len(self.actions) > 0:
            self._index = 0
        return action

    def peek_next_action(self) -> np.ndarray:
        if len(self.actions) == 0:
            return np.zeros(self.action_dim, dtype=np.float32)
        idx = min(self._index, len(self.actions) - 1)
        return np.asarray(self.actions[idx], dtype=np.float32)

    def is_active(self) -> bool:
        return len(self.actions) > 0
