# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
import numpy as np
import os
os.environ["HYDRA_FULL_ERROR"] = "1"

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import Camera, FrameTransformer
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform, combine_frame_transforms, subtract_frame_transforms
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


from .so_100_reach_env_cfg import So100ReachEnvCfg


class So100ReachEnv(DirectRLEnv):
    cfg: So100ReachEnvCfg

    def __init__(self, cfg: So100ReachEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Get joint indices for action mapping
        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)
        self.last_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

        
        self.action_scale_robot = self.cfg.action_scale_robot

        self.target_poses = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_poses[:, 0] = self.cfg.target_pos_x + self.scene.env_origins[:, 0]
        self.target_poses[:, 1] = self.cfg.target_pos_y + self.scene.env_origins[:, 1]
        self.target_poses[:, 2] = self.cfg.target_pos_z + self.scene.env_origins[:, 2]

    def _setup_scene(self):
        """Set up the simulation scene."""
        self.robot = Articulation(self.cfg.robot_cfg)
        self.ee_frame = FrameTransformer(self.cfg.ee_frame_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        self.cube_marker = FrameTransformer(self.cfg.cube_marker_cfg)
        self.camera = Camera(self.cfg.camera_cfg)
        table_cfg = self.cfg.table_cfg
        table_cfg.spawn.func(
            table_cfg.prim_path, table_cfg.spawn,
            translation=table_cfg.init_state.pos,
            orientation=table_cfg.init_state.rot
        )
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object
        self.scene.sensors["ee_frame"] = self.ee_frame
        self.scene.sensors["cube_marker"] = self.cube_marker
        self.scene.sensors["camera"] = self.camera
        # Clone environments
        self.scene.clone_environments(copy_from_source=False)

        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
    
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Store actions before physics step."""
        self.actions = actions.clone()
        self.actions[:, :6] = self.action_scale_robot * self.actions[:, :6]
        self.actions[:, 5:6] = torch.clamp(self.actions[:, 5:6], min=0.0, max=0.5)

    def _apply_action(self) -> None:
        # apply arm actions
        self.robot.set_joint_position_target(self.actions[:, :5], joint_ids=self.dof_idx[:5])
        # apply gripper actions
        self.robot.set_joint_position_target(self.actions[:, 5:6], joint_ids=self.dof_idx[5:6])
        self.last_actions = self.actions.clone()
        # if self.common_step_counter % 1000==0:
        #     print(f"last actions: {self.last_actions}")

    def _get_observations(self) -> dict:
        """Get observations for the policy."""
        # Joint positions (relative to initial positions)
        joint_pos_rel = self._joint_pos_rel()
        # Joint velocities (relative to initial velocities)
        joint_vel_rel = self._joint_vel_rel()
        # Object position in robot root frame
        object_pos_b = self._object_position_in_robot_root_frame()
        # Target position in robot root frame
        target_pos_b = self._target_position_in_robot_root_frame()
        # get camera rgb
        camera_rgb = self._get_camera_rgb()
        # Concatenate all observations
        states = torch.cat([
            joint_pos_rel,      # 6 dims
            joint_vel_rel,      # 6 dims
            object_pos_b,       # 3 dims
            target_pos_b,       # 3 dims
            self.last_actions,  # 6 dims
        ], dim=-1)
        obs = {
            "proprioceptive": states,
            "camera": camera_rgb,
        }
        observations = {
            "policy": obs
        }
        return observations

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards based on the manager-based environment reward structure."""
        reaching_object = self._object_ee_distance(self.cfg.object_ee_distance_std)
        
        action_rate_penalty = self._action_rate_penalty(self.actions, self.last_actions)

        joint_vel_penalty = self._joint_vel_penalty()
        
        if self.common_step_counter > 12000:
            action_rate_penalty_weight = -5e-4
            joint_vel_penalty_weight = -5e-4
        else:
            action_rate_penalty_weight = self.cfg.action_penalty_weight
            joint_vel_penalty_weight = self.cfg.joint_vel_penalty_weight
        # Combine all rewards with weights
        total_reward = (
            self.cfg.reaching_reward_weight * reaching_object +
            action_rate_penalty_weight * action_rate_penalty +
            joint_vel_penalty_weight * joint_vel_penalty
        )
        # if self.common_step_counter % 1000 == 0:
        #     print(f"reward at step {self.common_step_counter} is {total_reward.unsqueeze(-1)}")
        return total_reward.unsqueeze(-1)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get done flags based on manager-based environment termination conditions."""
        # 1. Episode timeout
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # 2. Object dropping (root height below minimum)
        object_height = self.object.data.root_pos_w[:, 2]
        object_dropping = object_height < 1.0
        return object_dropping, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset specific environments based on manager-based environment reset logic."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        # Call super to manage internal buffers (episode length, etc.)
        super()._reset_idx(env_ids)
        # Get the origins for the environments being reset
        env_origins = self.scene.env_origins[env_ids]  # shape: (num_envs, 3)
        default_root_state = self.object.data.default_root_state[env_ids]
        default_object_pos = default_root_state[:, :3] + env_origins
        # Randomize the object position
        object_pos_x = sample_uniform(lower=-self.cfg.randomization_range_cube_x, upper=self.cfg.randomization_range_cube_x, size=(self.num_envs,), device=self.device)
        object_pos_y = sample_uniform(lower=-self.cfg.randomization_range_cube_y, upper=self.cfg.randomization_range_cube_y, size=(self.num_envs,), device=self.device)
        object_pos_z = sample_uniform(lower=-self.cfg.randomization_range_cube_z, upper=self.cfg.randomization_range_cube_z, size=(self.num_envs,), device=self.device)
        object_pos = torch.stack([object_pos_x, object_pos_y, object_pos_z], dim=1)
        object_pos = default_object_pos + object_pos
        object_quat = default_root_state[:, 3:7]
        object_vel = default_root_state[:, 7:13]
        self.object.data.root_pos_w[env_ids] = object_pos
        self.object.data.root_quat_w[env_ids] = object_quat
        self.object.data.root_vel_w[env_ids] = object_vel
        default_root_state[:, :3] = object_pos
        self.object.write_root_state_to_sim(default_root_state, env_ids)

        # Randomize the target position
        target_pos_x = sample_uniform(lower=-self.cfg.randomization_range_target_x, upper=self.cfg.randomization_range_target_x, size=(self.num_envs,), device=self.device)
        target_pos_y = sample_uniform(lower=-self.cfg.randomization_range_target_y, upper=self.cfg.randomization_range_target_y, size=(self.num_envs,), device=self.device)
        target_pos_z = sample_uniform(lower=-self.cfg.randomization_range_target_z, upper=self.cfg.randomization_range_target_z, size=(self.num_envs,), device=self.device)
        target_pos = torch.stack([target_pos_x, target_pos_y, target_pos_z], dim=1)
        target_pos = default_object_pos + target_pos
        self.target_poses[env_ids] = target_pos

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self.last_actions[env_ids] = 0

    def _joint_pos_rel(self) -> torch.Tensor:
        """Get joint positions relative to initial positions."""
        return self.robot.data.joint_pos[:, self.dof_idx] - self.robot.data.default_joint_pos[:, self.dof_idx]

    def _joint_vel_rel(self) -> torch.Tensor:
        """Get joint velocities relative to initial velocities."""
        return self.robot.data.joint_vel[:, self.dof_idx] - self.robot.data.default_joint_vel[:, self.dof_idx]

    def _object_position_in_robot_root_frame(self) -> torch.Tensor:
        """Get object position in robot root frame."""
        object_pos_w = self.object.data.root_pos_w[:, :3]
        object_pos_b, _ = subtract_frame_transforms(
            self.robot.data.root_state_w[:, :3], 
            self.robot.data.root_state_w[:, 3:7], 
            object_pos_w
        )
        return object_pos_b

    def _target_position_in_robot_root_frame(self) -> torch.Tensor:
        """Get target position in robot root frame."""
        target_pos = self.target_poses[:, :3]
        target_pos_b, _ = subtract_frame_transforms(
            self.robot.data.root_state_w[:, :3], 
            self.robot.data.root_state_w[:, 3:7], 
            target_pos
        )
        return target_pos_b

    def _get_camera_rgb(self) -> torch.Tensor:
        """Get camera RGB images."""
        camera_data = self.camera.data.output
        if camera_data is None or "rgb" not in camera_data:
            print("[WARNING] Camera data is not available. Returning zero tensor.")
            return torch.zeros((self.num_envs, self.cfg.CAMERA_HEIGHT, self.cfg.CAMERA_WIDTH, 3), device=self.device)
        return camera_data["rgb"] / 255.0

    def _object_ee_distance(self, std: float) -> torch.Tensor:
        """Get object-end effector distance."""
        cube_pos_w = self.object.data.root_pos_w[:, :3]
        ee_w = self.ee_frame.data.target_pos_w[..., 0, :]
        cube_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)
        return 1 - torch.tanh(cube_ee_distance / std)
        
    def _action_rate_penalty(self, actions, prev_actions) -> torch.Tensor:
        """Penalize the rate of change of the actions using L2 squared kernel."""
        return torch.sum(torch.square(actions - prev_actions), dim=1)

    def _joint_vel_penalty(self) -> torch.Tensor:
        """Penalize the joint velocities using L2 norm."""
        return torch.sum(torch.square(self.robot.data.joint_vel[:, self.dof_idx]), dim=1) 