import torch


# def cube_released_termination(env) -> torch.Tensor:
#     """
#     cube를 한 번 잡았다가 contact가 사라지면 종료.
#     """

#     contact_sensor = env.scene["contact_sensor_cube"]

#     force_matrix = contact_sensor.data.net_forces_w
#     contact = (torch.norm(force_matrix, dim=-1).max(dim=1)[0] > 1e-4)

#     # env에 grasp flag 없으면 생성
#     if not hasattr(env, "cube_grasped"):
#         env.cube_grasped = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

#     # contact 발생하면 grasp 상태 기록
#     env.cube_grasped |= contact

#     # grasp 상태인데 contact 사라지면 release
#     released = env.cube_grasped & (~contact)

#     return released

def terminate_on_excessive_joint_velocity(
    env,
    max_joint_vel: float = 25.0,
) -> torch.Tensor:
    """어떤 joint든 속도가 너무 커지면 종료"""
    robot = env.scene["robot"]
    joint_vel_max = torch.abs(robot.data.joint_vel).max(dim=1).values
    return joint_vel_max > max_joint_vel


# def terminate_on_cube_out_of_bounds(
#     env,
#     max_dist_xy: float = 0.5,
#     min_z: float = -0.05,
#     max_z: float = 0.3,
# ) -> torch.Tensor:
#     """CHANGED: cube가 너무 멀리 가거나 너무 위/아래로 벗어나면 종료."""
#     cube = env.scene["cube"]
#     cube_pos = cube.data.root_pos_w[:, :3]

#     xy_norm = torch.norm(cube_pos[:, :2], dim=1)
#     too_far = xy_norm > max_dist_xy
#     too_low = cube_pos[:, 2] < min_z
#     too_high = cube_pos[:, 2] > max_z

#     return too_far | too_low | too_high