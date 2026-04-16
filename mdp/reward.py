import torch
from isaaclab.managers import SceneEntityCfg


def _get_finger_positions_and_cube(env):
    """각 손가락 끝 + cube 위치 반환."""
    robot = env.scene["robot"]
    cube = env.scene["cube"]

    left_idx = robot.find_bodies("Finger_Left_03")[0][0]
    right_idx = robot.find_bodies("Finger_Right_03")[0][0]

    left_pos = robot.data.body_pos_w[:, left_idx, :3]
    right_pos = robot.data.body_pos_w[:, right_idx, :3]
    cube_pos = cube.data.root_pos_w[:, :3]

    return left_pos, right_pos, cube_pos


def finger_center_to_cube_distance(env) -> torch.Tensor:
    """finger center → cube 유클리드 거리. [N]"""
    left_pos, right_pos, cube_pos = _get_finger_positions_and_cube(env)
    finger_center = 0.5 * (left_pos + right_pos)
    return torch.norm(finger_center - cube_pos, dim=-1)

# Stage 1: 접근 - center 기반

def reward_reach_cube_exp(env, temperature: float = 0.1) -> torch.Tensor:
    """
    finger center -> cube 거리의 exp reward
    """
    left_pos, right_pos, cube_pos = _get_finger_positions_and_cube(env)
    finger_center = 0.5 * (left_pos + right_pos)
    dist = torch.norm(finger_center - cube_pos, dim=-1)
    return torch.exp(-dist / temperature)

# Stage 2: 정렬 - 각 손가락이 개별로 cube에 가까이

def reward_per_finger_distance(
    env,
    temperature: float = 0.05,
) -> torch.Tensor:
    """
    left -> cube, right -> cube 각각의 exp reward를 곱함

    - 합이면 한쪽만 가까이 가도 보상 절반을 먹을 수 있음
    - 곱이면 한쪽이 0이면 전체가 0 -> 양쪽 다 가까워야만 높은 보상
    - 손가락 벌린 채 hover 시, left_r=0.8, right_r=0.01 => 곱=0.008 (거의 0)
    - 양쪽 다 가까이 접근 시, left_r=0.8, right_r=0.8 => 곱=0.64 (높음)

    이 구조 때문에 hover hack이 구조적으로 불가능
    """
    left_pos, right_pos, cube_pos = _get_finger_positions_and_cube(env)

    left_dist = torch.norm(left_pos - cube_pos, dim=-1)
    right_dist = torch.norm(right_pos - cube_pos, dim=-1)

    left_r = torch.exp(-left_dist / temperature)
    right_r = torch.exp(-right_dist / temperature)

    return left_r * right_r


# Stage 3: 오므리기 - cube 근처에서 손가락 닫기

def reward_finger_closing(
    env,
    activation_dist: float = 0.08,
) -> torch.Tensor:
    """
    cube 근처(8cm)에 왔을 때, 두 손가락 사이 gap이 좁으면 보상

    이게 contact까지의 gradient를 만들어줌
    - hover 상태: gap 크다 -> 보상 낮음
    - 오므리는 중: gap 줄어든다 -> 보상 올라감
    - 거의 닫힘: gap 거의 0 -> 보상 거의 1.0
    => contact 보상(binary)까지 끊기지 않는 gradient 경로 완성

    cube에서 먼 경우엔 0 반환 → 헛되이 오므리지 않음
    """
    left_pos, right_pos, cube_pos = _get_finger_positions_and_cube(env)

    finger_center = 0.5 * (left_pos + right_pos)
    dist_to_cube = torch.norm(finger_center - cube_pos, dim=-1)

    near_cube = (dist_to_cube < activation_dist).float()

    finger_gap = torch.norm(left_pos - right_pos, dim=-1)

    closing_reward = torch.exp(-finger_gap / 0.05)  # 0.02 -> 0.05: gap 6cm에서 gradient 너무 희박했음

    return near_cube * closing_reward


# Stage 4: 접촉

def reward_single_finger_contact(env, near_threshold=0.08) -> torch.Tensor:
    """한쪽 손가락만 큐브에 닿았을 때 보상"""
    left_sensor = env.scene["contact_sensor_left_finger"]
    right_sensor = env.scene["contact_sensor_right_finger"]

    left_contact = (torch.norm(left_sensor.data.net_forces_w, dim=-1) > 1e-3).any(dim=1)
    right_contact = (torch.norm(right_sensor.data.net_forces_w, dim=-1) > 1e-3).any(dim=1)

    single_only = (left_contact ^ right_contact)
    near_enough = finger_center_to_cube_distance(env) < near_threshold

    return (single_only & near_enough).float()


def reward_dual_finger_contact(
    env,
    near_threshold: float = 0.05,
) -> torch.Tensor:
    """양쪽 손가락 모두 큐브에 닿았을 때 보상. 현재 최종 목표."""   # 이제 목표를 확장할 때가 되었다!!!
    left_sensor = env.scene["contact_sensor_left_finger"]
    right_sensor = env.scene["contact_sensor_right_finger"]

    left_force = torch.norm(left_sensor.data.net_forces_w, dim=-1)
    right_force = torch.norm(right_sensor.data.net_forces_w, dim=-1)

    left_contact = (left_force > 1e-3).any(dim=1)
    right_contact = (right_force > 1e-3).any(dim=1)

    dual_contact = (left_contact & right_contact)

    dist = finger_center_to_cube_distance(env)
    near_enough = dist < near_threshold

    return dual_contact.float() * near_enough.float()


def reward_close_approach(env) -> torch.Tensor:
    """
    finger center -> cube 거리에 대해 T=0.03의 tight한 exp reward.
    reach_cube(T=0.08)보다 가까울수록 훨씬 강한 gradient를 제공하여
    마지막 5cm 구간에서 contact까지 밀어주는 역할.
    11cm에서는 거의 0(=0.026)이라 기존 reward와 충돌 없이,
    5cm 이내로 들어오면 급격히 올라 contact를 유도함.
    """
    dist = finger_center_to_cube_distance(env)
    return torch.exp(-dist / 0.03)