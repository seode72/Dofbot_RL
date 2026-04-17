#!/usr/bin/env python3
"""
policy_server.py
SAC policy server — TCP로 obs 수신, load_best_agent.py 모델로 action 추론 후 반환.

사용법:
    python policy_server.py
    python policy_server.py --checkpoint path/to/best_agent.pt --host 0.0.0.0 --port 5000 --device cuda
"""

import argparse
import socket
import pickle
import struct
import sys
import os
from typing import Optional, Dict, Any, List

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_best_agent import PolicyNetwork, load_best_agent

# --------------------------------------------------------------------------- #
# 기본 설정
# --------------------------------------------------------------------------- #
DEFAULT_CHECKPOINT = "Dofbot_RL-HSH/Dofbot_RL-HSH/20260415_003530/best_agent.pt"
OBS_DIM    = 17   # joint_pos(7) + joint_vel(7) + cube_pos(3)
ACTION_DIM = 7    # joint1~4, Wrist, Finger_L, Finger_R

# 실제 로봇 6관절 순서:
#   [0] joint1  [1] joint2  [2] joint3  [3] joint4  [4] Wrist  [5] Gripper
#
# 모델 7관절 순서 (학습 시):
#   [0] joint1  [1] joint2  [2] joint3  [3] joint4  [4] Wrist
#   [5] Finger_L  [6] Finger_R
#
# 변환 규칙:
#   obs  실제→모델 : Gripper(1개) → Finger_L=Finger_R (동일 값 복제)
#   action 모델→실제 : Finger_L, Finger_R 평균 → Gripper(1개)


# --------------------------------------------------------------------------- #
# 6관절 ↔ 7관절 변환
# --------------------------------------------------------------------------- #
def real6_to_model7(joints_6: List[float]) -> List[float]:
    """실제 로봇 6관절 → 모델 입력 7관절.

    학습 시 Finger_L, Finger_R는 부호가 반대로 움직입니다.
    (GRASP_GOAL: Finger_L=+0.1745, Finger_R=-0.1745)
    따라서: Finger_L = +gripper, Finger_R = -gripper
    """
    j1, j2, j3, j4, wrist, gripper = joints_6
    return [j1, j2, j3, j4, wrist, gripper, -gripper]


def model7_to_real6(action_7: List[float]) -> List[float]:
    """모델 출력 7관절 action → 실제 로봇 6관절.

    Finger_L = +gripper, Finger_R = -gripper 대칭 구조이므로
    gripper = Finger_L 값을 그대로 사용합니다.
    (검증: (Finger_L - Finger_R) / 2 와 동일)
    """
    j1, j2, j3, j4, wrist, finger_l, finger_r = action_7
    gripper = (finger_l - finger_r) / 2.0  # 대칭 평균
    return [j1, j2, j3, j4, wrist, gripper]


# --------------------------------------------------------------------------- #
# 소켓 유틸
# --------------------------------------------------------------------------- #
def recv_all(sock: socket.socket, n: int) -> Optional[bytes]:
    data = b""
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data


def send_packet(sock: socket.socket, obj: Dict[str, Any]) -> None:
    payload = pickle.dumps(obj, protocol=2)
    header  = struct.pack(">I", len(payload))
    sock.sendall(header + payload)


def receive_packet(sock: socket.socket) -> Optional[Dict[str, Any]]:
    header = recv_all(sock, 4)
    if not header:
        return None
    size    = struct.unpack(">I", header)[0]
    payload = recv_all(sock, size)
    if not payload:
        return None
    return pickle.loads(payload, encoding="latin1")


# --------------------------------------------------------------------------- #
# 추론
# --------------------------------------------------------------------------- #
def compute_action(
    policy: PolicyNetwork,
    obs: Dict[str, Any],
    device: str,
) -> List[float]:
    """
    obs 딕셔너리에서 관측값을 꺼내 PolicyNetwork로 action을 추론합니다.

    기대 키 (실제 로봇 기준):
        joint_pos  : list[float] 길이 6  (joint1~4, Wrist, Gripper)
        joint_vel  : list[float] 길이 6
        cube_pos   : list[float] 길이 3  (env-local 좌표계)
                     또는 cube1_pos 키도 허용

    반환값:
        action : list[float] 길이 6  (joint1~4, Wrist, Gripper)
    """
    joint_pos_6 = obs.get("joint_pos", [0.0] * 6)
    joint_vel_6 = obs.get("joint_vel", [0.0] * 6)
    cube_pos    = obs.get("cube_pos",  obs.get("cube1_pos", [0.0, 0.0, 0.0]))

    print(f"[GPU] joint_pos(6) : {[round(x, 3) for x in joint_pos_6]}")
    print(f"[GPU] joint_vel(6) : {[round(x, 3) for x in joint_vel_6]}")
    print(f"[GPU] cube_pos     : {[round(x, 3) for x in cube_pos]}")

    # 실제 6관절 → 모델 7관절로 변환
    joint_pos_7 = real6_to_model7(joint_pos_6)
    joint_vel_7 = real6_to_model7(joint_vel_6)

    obs_vec    = np.array(joint_pos_7 + joint_vel_7 + cube_pos, dtype=np.float32)  # (17,)
    obs_tensor = torch.FloatTensor(obs_vec).unsqueeze(0).to(device)                 # (1, 17)

    with torch.no_grad():
        mean_action, _ = policy(obs_tensor)
        action_7 = mean_action.squeeze(0).cpu().numpy().tolist()                    # (7,)

    # 모델 7관절 action → 실제 6관절로 변환
    action_6 = model7_to_real6(action_7)

    print(f"[GPU] action(6)    : {[round(a, 4) for a in action_6]}")
    return action_6


# --------------------------------------------------------------------------- #
# 서버 루프
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="SAC policy server for Dofbot")
    parser.add_argument("--host",       type=str,   default="0.0.0.0")
    parser.add_argument("--port",       type=int,   default=5000)
    parser.add_argument("--checkpoint", type=str,   default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device",     type=str,   default="cpu")
    args = parser.parse_args()

    # 모델 로드 (서버 기동 시 1회만)
    print(f"[GPU] Loading model: {args.checkpoint}")
    policy, _, _, meta = load_best_agent(
        checkpoint_path=args.checkpoint,
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        device=args.device,
    )
    print(f"[GPU] Model ready — best_succ_rate={meta['best_succ_rate']:.2f}%  step={meta['step']}")

    # 소켓 서버 시작
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)

    # 실제 LAN IP 출력 (UDP trick — 실제 패킷 안 보냄)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = socket.gethostbyname(socket.gethostname())
    print(f"[GPU] Server IP : {local_ip}")
    print(f"[GPU] Listening on {args.host}:{args.port}")
    print(f"[GPU] → Jetson에서 연결: --host {local_ip}")

    conn, addr = server.accept()
    print(f"[GPU] Connected: {addr}")

    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy_server.txt")
    print(f"[GPU] Logging actions to: {log_path}")

    try:
        with open(log_path, "w") as log_f:
            log_f.write("step\taction_j1\taction_j2\taction_j3\taction_j4\taction_wrist\taction_gripper\n")
            step = 0
            while True:
                obs = receive_packet(conn)
                if obs is None:
                    print("[GPU] Client disconnected.")
                    break

                action   = compute_action(policy, obs, args.device)
                response = {"action": action}
                send_packet(conn, response)

                # 로그 기록
                log_f.write(f"{step}\t" + "\t".join(f"{a:.6f}" for a in action) + "\n")
                log_f.flush()
                step += 1

    except KeyboardInterrupt:
        print("\n[GPU] Stopped by user.")
    except Exception as e:
        print(f"[GPU] Error: {e}")
    finally:
        conn.close()
        server.close()


if __name__ == "__main__":
    main()
