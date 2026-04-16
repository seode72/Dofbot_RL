import torch
from isaaclab.managers import SceneEntityCfg


def left_contact_binary(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor_left_finger"),
):
    sensor = env.scene.sensors[sensor_cfg.name]
    force_matrix = sensor.data.force_matrix_w
    force_mag = torch.norm(force_matrix, dim=-1)
    contact = (force_mag > 1e-6).any(dim=(1, 2))
    return contact.float().unsqueeze(-1)


def right_contact_binary(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor_right_finger"),
):
    sensor = env.scene.sensors[sensor_cfg.name]
    force_matrix = sensor.data.force_matrix_w
    force_mag = torch.norm(force_matrix, dim=-1)
    contact = (force_mag > 1e-6).any(dim=(1, 2))
    return contact.float().unsqueeze(-1)


def cube_pos_local_obs(env) -> torch.Tensor:
    """Cube position in env-local frame (world pos - env_origin). Shape: [N, 3]."""
    cube = env.scene["cube"]
    origins = env.scene.env_origins  # [N, 3]
    return cube.data.root_pos_w[:, :3] - origins


def left_finger_pos_obs(env) -> torch.Tensor:
    """Left finger tip position in env-local frame. Shape: [N, 3]."""
    robot = env.scene["robot"]
    origins = env.scene.env_origins
    left_idx = robot.find_bodies("Finger_Left_03")[0][0]
    return robot.data.body_pos_w[:, left_idx, :3] - origins


def right_finger_pos_obs(env) -> torch.Tensor:
    """Right finger tip position in env-local frame. Shape: [N, 3]."""
    robot = env.scene["robot"]
    origins = env.scene.env_origins
    right_idx = robot.find_bodies("Finger_Right_03")[0][0]
    return robot.data.body_pos_w[:, right_idx, :3] - origins


def left_ground_contact_obs(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor_left_ground"),
):
    sensor = env.scene.sensors[sensor_cfg.name]
    force = sensor.data.force_matrix_w
    if force.dim() > 3:
        force = force[:, -1]
    contact = (torch.norm(force, dim=-1) > 1e-6).any(dim=1).float()
    return contact.unsqueeze(-1)


def right_ground_contact_obs(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor_right_ground"),
):
    sensor = env.scene.sensors[sensor_cfg.name]
    force = sensor.data.force_matrix_w
    if force.dim() > 3:
        force = force[:, -1]
    contact = (torch.norm(force, dim=-1) > 1e-6).any(dim=1).float()
    return contact.unsqueeze(-1)


def finger_center_to_cube_vec_obs(env) -> torch.Tensor:
    """cube - finger_center 벡터 (상대 벡터). Shape: [N, 3]."""
    robot = env.scene["robot"]
    cube = env.scene["cube"]
    left_idx = robot.find_bodies("Finger_Left_03")[0][0]
    right_idx = robot.find_bodies("Finger_Right_03")[0][0]
    finger_center = 0.5 * (robot.data.body_pos_w[:, left_idx, :3] + robot.data.body_pos_w[:, right_idx, :3])
    return cube.data.root_pos_w[:, :3] - finger_center  # [N, 3]
