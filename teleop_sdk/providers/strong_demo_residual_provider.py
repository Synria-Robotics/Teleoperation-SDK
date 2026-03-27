from __future__ import annotations

import numpy as np

from teleop_sdk.providers.base import BaseResidualPolicy


class StrongDemoResidualPolicy(BaseResidualPolicy):
    """Phase-aware scripted copilot for a visibly strong shared-control demo."""

    def __init__(
        self,
        *,
        residual_limit: float = 0.40,
        align_gain: float = 1.35,
        near_align_gain: float = 2.40,
        near_align_floor: float = 0.38,
        descend_gain: float = 0.72,
        close_gain: float = 1.20,
        lift_gain: float = 0.85,
        place_gain: float = 1.35,
        near_place_gain: float = 1.80,
        near_place_floor: float = 0.30,
        place_descend_gain: float = 0.60,
        grasp_xy_threshold: float = 0.09,
        descend_xy_threshold: float = 0.020,
        descend_z_threshold: float = 0.055,
        grasp_xyz_threshold: float = 0.040,
        close_xy_threshold: float = 0.020,
        close_z_threshold: float = 0.022,
        place_xy_threshold: float = 0.12,
        place_descend_xy_threshold: float = 0.055,
        lifted_height: float = 0.09,
        idle_deadband: float = 0.02,
    ):
        self.residual_limit = float(residual_limit)
        self.align_gain = float(align_gain)
        self.near_align_gain = float(near_align_gain)
        self.near_align_floor = float(near_align_floor)
        self.descend_gain = float(descend_gain)
        self.close_gain = float(close_gain)
        self.lift_gain = float(lift_gain)
        self.place_gain = float(place_gain)
        self.near_place_gain = float(near_place_gain)
        self.near_place_floor = float(near_place_floor)
        self.place_descend_gain = float(place_descend_gain)
        self.grasp_xy_threshold = float(grasp_xy_threshold)
        self.descend_xy_threshold = float(descend_xy_threshold)
        self.descend_z_threshold = float(descend_z_threshold)
        self.grasp_xyz_threshold = float(grasp_xyz_threshold)
        self.close_xy_threshold = float(close_xy_threshold)
        self.close_z_threshold = float(close_z_threshold)
        self.place_xy_threshold = float(place_xy_threshold)
        self.place_descend_xy_threshold = float(place_descend_xy_threshold)
        self.lifted_height = float(lifted_height)
        self.idle_deadband = float(idle_deadband)

    def reset(self) -> None:
        pass

    @staticmethod
    def _safe_norm(vec: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(vec, dtype=np.float32)))

    @staticmethod
    def _floor_vector(vec: np.ndarray, floor_mag: float) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            return np.zeros_like(vec)
        return vec / norm * max(norm, float(floor_mag))

    def get_action(self, obs: dict[str, np.ndarray], base_action: np.ndarray) -> np.ndarray:
        base_action = np.asarray(base_action, dtype=np.float32)
        if self._safe_norm(base_action) < self.idle_deadband:
            return np.zeros(self.action_dim, dtype=np.float32)

        box_to_ee = np.asarray(obs["box_to_ee"], dtype=np.float32)
        box_to_basket = np.asarray(obs["box_to_basket"], dtype=np.float32)
        box_pos = np.asarray(obs["box_pos"], dtype=np.float32)
        gripper_pos = float(np.asarray(obs["gripper_pos"], dtype=np.float32).reshape(-1)[0])

        residual = np.zeros(self.action_dim, dtype=np.float32)

        box_xy = box_to_ee[:2]
        box_xy_dist = self._safe_norm(box_xy)
        box_xyz_dist = self._safe_norm(box_to_ee)
        lifted = float(box_pos[2]) >= self.lifted_height

        if not lifted:
            align_xy = np.clip(box_xy, -1.0, 1.0) * self.align_gain
            if box_xy_dist < self.grasp_xy_threshold:
                align_xy = self._floor_vector(np.clip(box_xy, -1.0, 1.0) * self.near_align_gain, self.near_align_floor)
            residual[:2] += align_xy

            box_z_err = abs(float(box_to_ee[2]))
            if box_xy_dist < self.descend_xy_threshold and box_z_err < self.descend_z_threshold:
                residual[2] += float(np.clip(box_to_ee[2], -1.0, 1.0)) * self.descend_gain

            if (
                box_xy_dist < self.close_xy_threshold
                and box_z_err < self.close_z_threshold
                and box_xyz_dist < self.grasp_xyz_threshold
            ):
                residual[6] -= self.close_gain
                residual[2] += self.lift_gain
            elif (
                box_xy_dist < self.close_xy_threshold * 1.25
                and box_z_err < self.close_z_threshold
                and box_xyz_dist < self.grasp_xyz_threshold * 1.25
            ):
                residual[6] -= 0.5 * self.close_gain
        else:
            place_xy = box_to_basket[:2]
            place_xy_dist = self._safe_norm(place_xy)
            place_xy_cmd = np.clip(place_xy, -1.0, 1.0) * self.place_gain
            if place_xy_dist < self.place_xy_threshold:
                place_xy_cmd = self._floor_vector(
                    np.clip(place_xy, -1.0, 1.0) * self.near_place_gain,
                    self.near_place_floor,
                )
            residual[:2] += place_xy_cmd

            if place_xy_dist < self.place_descend_xy_threshold:
                residual[2] += float(np.clip(box_to_basket[2], -1.0, 1.0)) * self.place_descend_gain

            if place_xy_dist < self.place_descend_xy_threshold and abs(float(box_to_basket[2])) < 0.05:
                residual[6] += np.clip((0.012 - gripper_pos) / 0.012, 0.0, 1.0) * self.close_gain

        return np.clip(residual, -self.residual_limit, self.residual_limit).astype(np.float32)


class DemoSnapPhasePolicy(BaseResidualPolicy):
    """Explicit phase-based demo copilot for obvious snap/descend/grasp/lift/place behavior."""

    def __init__(
        self,
        *,
        residual_limit: float = 0.9,
        align_floor: float = 0.42,
        place_floor: float = 0.32,
        descend_speed: float = 0.85,
        lift_speed: float = 0.85,
        carry_lift: float = 0.04,
        carry_xy_damp: float = 0.0,
        close_speed: float = 1.0,
        release_speed: float = 1.0,
        grasp_align_xy: float = 0.12,
        grasp_descend_xy: float = 0.05,
        grasp_close_xy: float = 0.020,
        grasp_close_z: float = 0.030,
        lift_height: float = 0.11,
        carry_safe_height: float = 0.10,
        place_align_xy: float = 0.16,
        place_descend_xy: float = 0.06,
        place_release_xy: float = 0.035,
        place_release_z: float = 0.035,
        human_xy_scale: float = 0.10,
        human_rot_scale: float = 0.20,
        carry_human_xy_scale: float = 1.0,
        min_human_xy_norm: float = 0.08,
        min_human_z_norm: float = 0.10,
        intent_cos_threshold: float = 0.35,
        close_steps: int = 12,
        lift_steps: int = 18,
        release_steps: int = 10,
    ):
        self.residual_limit = float(residual_limit)
        self.align_floor = float(align_floor)
        self.place_floor = float(place_floor)
        self.descend_speed = float(descend_speed)
        self.lift_speed = float(lift_speed)
        self.carry_lift = float(carry_lift)
        self.carry_xy_damp = float(carry_xy_damp)
        self.close_speed = float(close_speed)
        self.release_speed = float(release_speed)
        self.grasp_align_xy = float(grasp_align_xy)
        self.grasp_descend_xy = float(grasp_descend_xy)
        self.grasp_close_xy = float(grasp_close_xy)
        self.grasp_close_z = float(grasp_close_z)
        self.lift_height = float(lift_height)
        self.carry_safe_height = float(carry_safe_height)
        self.place_align_xy = float(place_align_xy)
        self.place_descend_xy = float(place_descend_xy)
        self.place_release_xy = float(place_release_xy)
        self.place_release_z = float(place_release_z)
        self.human_xy_scale = float(human_xy_scale)
        self.human_rot_scale = float(human_rot_scale)
        self.carry_human_xy_scale = float(carry_human_xy_scale)
        self.min_human_xy_norm = float(min_human_xy_norm)
        self.min_human_z_norm = float(min_human_z_norm)
        self.intent_cos_threshold = float(intent_cos_threshold)
        self.close_steps = int(close_steps)
        self.lift_steps = int(lift_steps)
        self.release_steps = int(release_steps)
        self.reset()

    def reset(self) -> None:
        self.phase = "align_grasp"
        self.phase_step = 0

    @staticmethod
    def _norm(vec: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(vec, dtype=np.float32)))

    @staticmethod
    def _floor_dir(vec: np.ndarray, floor_mag: float) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            return np.zeros_like(vec)
        return vec / norm * float(floor_mag)

    def _advance(self, phase: str) -> None:
        if self.phase != phase:
            self.phase = phase
            self.phase_step = 0
        else:
            self.phase_step += 1

    def _xy_intent_ok(self, base_action: np.ndarray, target_xy: np.ndarray) -> bool:
        human_xy = np.asarray(base_action[:2], dtype=np.float32)
        target_xy = np.asarray(target_xy, dtype=np.float32)
        human_norm = self._norm(human_xy)
        target_norm = self._norm(target_xy)
        if human_norm < self.min_human_xy_norm or target_norm < 1e-6:
            return False
        cosine = float(np.dot(human_xy, target_xy) / max(human_norm * target_norm, 1e-6))
        return cosine >= self.intent_cos_threshold

    def _downward_intent_ok(self, base_action: np.ndarray) -> bool:
        return float(base_action[2]) < -self.min_human_z_norm

    def get_base_scale(self, obs: dict[str, np.ndarray], base_action: np.ndarray) -> np.ndarray:
        scale = np.ones(self.action_dim, dtype=np.float32)
        box_xy = self._norm(np.asarray(obs["box_to_ee"], dtype=np.float32)[:2])
        place_xy = self._norm(np.asarray(obs["box_to_basket"], dtype=np.float32)[:2])
        box_z = float(np.asarray(obs["box_pos"], dtype=np.float32)[2])
        lifted = box_z >= self.lift_height
        engage_grasp = self._xy_intent_ok(base_action, np.asarray(obs["box_to_ee"], dtype=np.float32)[:2])
        engage_place = self._xy_intent_ok(base_action, np.asarray(obs["box_to_basket"], dtype=np.float32)[:2])

        if ((not lifted and box_xy < self.grasp_align_xy and engage_grasp) or (lifted and place_xy < self.place_align_xy and engage_place)):
            scale[:2] *= self.human_xy_scale
            scale[3:6] *= self.human_rot_scale
        if self.phase in {"descend_grasp", "close_grasp", "lift_grasp", "descend_place", "release_place"}:
            scale[:2] *= self.human_xy_scale
            scale[2] *= 0.25
            scale[3:6] *= self.human_rot_scale
        return scale

    def get_action(self, obs: dict[str, np.ndarray], base_action: np.ndarray) -> np.ndarray:
        base_action = np.asarray(base_action, dtype=np.float32)
        box_to_ee = np.asarray(obs["box_to_ee"], dtype=np.float32)
        box_to_basket = np.asarray(obs["box_to_basket"], dtype=np.float32)
        box_pos = np.asarray(obs["box_pos"], dtype=np.float32)
        box_xy = self._norm(box_to_ee[:2])
        box_z_err = abs(float(box_to_ee[2]))
        place_xy = self._norm(box_to_basket[:2])
        place_z_err = abs(float(box_to_basket[2]))
        lifted = float(box_pos[2]) >= self.lift_height

        residual = np.zeros(self.action_dim, dtype=np.float32)
        engage_grasp = self._xy_intent_ok(base_action, box_to_ee[:2])
        engage_place = self._xy_intent_ok(base_action, box_to_basket[:2])
        descend_intent = self._downward_intent_ok(base_action)

        if not lifted and self.phase not in {"close_grasp", "lift_grasp", "carry_hold"}:
            self.phase = "align_grasp"
            self.phase_step = 0
        if lifted and self.phase in {"align_grasp", "descend_grasp", "close_grasp", "lift_grasp"}:
            self.phase = "carry_hold"
            self.phase_step = 0

        if self.phase == "align_grasp":
            if not engage_grasp:
                return residual
            residual[:2] = self._floor_dir(box_to_ee[:2], self.align_floor)
            if box_xy < self.grasp_descend_xy and descend_intent:
                self._advance("descend_grasp")
        elif self.phase == "descend_grasp":
            if not engage_grasp:
                self.phase = "align_grasp"
                self.phase_step = 0
                return residual
            residual[:2] = self._floor_dir(box_to_ee[:2], self.align_floor)
            residual[2] = float(np.clip(box_to_ee[2], -1.0, 1.0)) * self.descend_speed
            if box_xy < self.grasp_close_xy and box_z_err < self.grasp_close_z:
                self._advance("close_grasp")
        elif self.phase == "close_grasp":
            residual[:2] = self._floor_dir(box_to_ee[:2], self.align_floor)
            residual[2] = 0.05
            if self.phase_step >= self.close_steps:
                self._advance("lift_grasp")
        elif self.phase == "lift_grasp":
            residual[2] = min(self.lift_speed, 0.45)
            if float(box_pos[2]) >= self.carry_safe_height or self.phase_step >= min(self.lift_steps, 10):
                self._advance("carry_hold")
        elif self.phase == "carry_hold":
            residual[2] = self.carry_lift
            if engage_place and place_xy < self.place_align_xy:
                self._advance("align_place")
        elif self.phase == "align_place":
            if not engage_place:
                residual[2] = 0.02
                return residual
            residual[:2] = self._floor_dir(box_to_basket[:2], self.place_floor)
            residual[2] = 0.02
            if place_xy < self.place_descend_xy and descend_intent:
                self._advance("descend_place")
        elif self.phase == "descend_place":
            if not engage_place:
                self.phase = "align_place"
                self.phase_step = 0
                return residual
            residual[:2] = self._floor_dir(box_to_basket[:2], self.place_floor)
            residual[2] = float(np.clip(box_to_basket[2], -1.0, 1.0)) * self.descend_speed
            if place_xy < self.place_release_xy and place_z_err < self.place_release_z:
                self._advance("release_place")
        elif self.phase == "release_place":
            residual[2] = 0.20
            if self.phase_step >= self.release_steps:
                self._advance("align_grasp")
        else:
            self.phase = "align_grasp"
            self.phase_step = 0

        self.phase_step += 1
        return np.clip(residual, -self.residual_limit, self.residual_limit).astype(np.float32)
