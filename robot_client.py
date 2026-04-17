#!/usr/bin/env python3
"""
robot_client.py
실제 로봇(Dofbot)에서 실행하는 클라이언트.
① 먼저 로봇을 시뮬 default 자세로 이동
② 그 후 서버와 소켓 통신 시작

사용법:
    python robot_client.py --host 서버IP --port 5000
    python robot_client.py --host 192.168.1.139 --port 5000
    python robot_client.py --host 10.126.36.101 --port 6006


"""

import argparse
import os
import socket
import pickle
import struct
import time
import threading
import math
from typing import Optional, Dict, Any, List

from Arm_Lib import Arm_Device

# ------------------------------------------------------------------ #
# 설정
# ------------------------------------------------------------------ #
SERVO_IDS    = [1, 2, 3, 4, 5, 6]
MOVE_TIME_MS = 50    # 서보 이동 시간 (ms) — 30Hz 대응

# 시뮬 default 관절값 (rad) — cfg에서 확인
DEFAULT_RAD = [0.0, 0.0, -1.1, -1.5, 0.0, 0.0]

# 시뮬 action 스케일
SCALES = [0.7, 0.7, 0.7, 0.7, 0.2, 0.15]

arm         = Arm_Device()
serial_lock = threading.Lock()


# ------------------------------------------------------------------ #
# 단위 변환 (서보 90° = 관절 0 rad 가정)
# ------------------------------------------------------------------ #
def deg_to_rad(deg: float) -> float:
    return (deg - 90.0) * math.pi / 180.0

def rad_to_deg(rad: float) -> float:
    return rad * 180.0 / math.pi + 90.0

def clamp(x: float, lo: int = 0, hi: int = 180) -> int:
    return max(lo, min(hi, int(round(x))))


# ------------------------------------------------------------------ #
# 소켓 유틸
# ------------------------------------------------------------------ #
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
    sock.sendall(struct.pack(">I", len(payload)) + payload)

def receive_packet(sock: socket.socket) -> Optional[Dict[str, Any]]:
    header = recv_all(sock, 4)
    if not header:
        return None
    payload = recv_all(sock, struct.unpack(">I", header)[0])
    if not payload:
        return None
    return pickle.loads(payload, encoding="latin1")


# ------------------------------------------------------------------ #
# 로봇 인터페이스
# ------------------------------------------------------------------ #
def read_joint_rad() -> List[float]:
    """서보 각도(도°) 읽어서 라디안으로 변환"""
    with serial_lock:
        degs = [float(arm.Arm_serial_servo_read(i) or 90) for i in SERVO_IDS]
    return [deg_to_rad(d) for d in degs]

def get_cube_pos() -> List[float]:
    """TODO: 카메라/비전으로 교체"""
    return [0.0, 0.20, 0.03]

# EMA 스무딩 계수 (0.1~0.3 권장, 작을수록 느리고 부드러움)
ALPHA = 0.2

# 이전 모터 명령 (초기값: default 자세)
_prev_cmd_degs: List[float] = [clamp(rad_to_deg(r)) for r in DEFAULT_RAD]


def apply_action(action_6: List[float]) -> None:
    """모델 action → EMA 스무딩 → 서보 전송
    최종_cmd = (1 - alpha) × 이전_cmd + alpha × 신규_목표
    """
    global _prev_cmd_degs

    # 신규 목표 각도 계산
    target_degs = [
        clamp(rad_to_deg(DEFAULT_RAD[i] + action_6[i] * SCALES[i]))
        for i in range(6)
    ]

    # EMA 스무딩
    smoothed_degs = [
        clamp((1.0 - ALPHA) * _prev_cmd_degs[i] + ALPHA * target_degs[i])
        for i in range(6)
    ]

    _prev_cmd_degs = smoothed_degs

    with serial_lock:
        arm.Arm_serial_servo_write6(
            smoothed_degs[0], smoothed_degs[1], smoothed_degs[2],
            smoothed_degs[3], smoothed_degs[4], smoothed_degs[5],
            MOVE_TIME_MS,
        )
    print(f"[Jetson] target: {target_degs}  smoothed: {smoothed_degs}")


# ------------------------------------------------------------------ #
# ① 초기 자세로 이동
# ------------------------------------------------------------------ #
def move_to_default(settle_time: float = 3.0) -> None:
    """
    시뮬 default 관절값(DEFAULT_RAD)에 해당하는 서보 각도로 이동 후
    settle_time 초 동안 대기합니다.

    서보 기준 default 각도 (90°=0rad 가정):
      joint1~2: 90°,  joint3: ~27°,  joint4: ~4°
      wrist:    90°,  gripper: 90°
    """
    default_degs = [clamp(rad_to_deg(r)) for r in DEFAULT_RAD]
    print(f"[Jetson] Moving to default pose: {default_degs} deg")

    with serial_lock:
        arm.Arm_serial_servo_write6(
            default_degs[0], default_degs[1], default_degs[2],
            default_degs[3], default_degs[4], default_degs[5],
            2000,  # 2초에 걸쳐 천천히 이동
        )

    print(f"[Jetson] Waiting {settle_time}s for robot to settle ...")
    time.sleep(settle_time)
    print("[Jetson] Default pose reached. Starting policy loop.")


# ------------------------------------------------------------------ #
# ② 메인 루프
# ------------------------------------------------------------------ #
def main() -> None:
    parser = argparse.ArgumentParser(description="Dofbot policy client")
    parser.add_argument("--host",        type=str,   default="127.0.0.1")
    parser.add_argument("--port",        type=int,   default=5000)
    parser.add_argument("--hz",          type=float, default=30.0)
    parser.add_argument("--settle_time", type=float, default=3.0,
                        help="default 자세 도달 후 대기 시간 (초)")
    args = parser.parse_args()
    period = 1.0 / args.hz

    # ① 로그 파일 먼저 생성
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot_client.txt")
    log_f = open(log_path, "w")
    log_f.write("step\t"
                "jp1\tjp2\tjp3\tjp4\tjp_wrist\tjp_grip\t"
                "jv1\tjv2\tjv3\tjv4\tjv_wrist\tjv_grip\t"
                "cube_x\tcube_y\tcube_z\n")
    log_f.flush()
    print(f"[Jetson] Logging obs to: {log_path}")

    # ② 먼저 default 자세로 이동
    move_to_default(settle_time=args.settle_time)

    # ③ 서버 연결
    print(f"[Jetson] Connecting to {args.host}:{args.port} ...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.host, args.port))
    sock.settimeout(10.0)
    print("[Jetson] Connected. Starting control loop.")

    prev_rad = None
    prev_t   = None

    try:
        step = 0
        while True:
                t0  = time.time()

                # 관절 상태 읽기
                joint_rad = read_joint_rad()
                dt        = (t0 - prev_t) if prev_t else 0.0
                joint_vel = [(c - p) / dt if dt > 0 else 0.0
                             for c, p in zip(joint_rad, prev_rad or joint_rad)]
                cube_pos  = get_cube_pos()

                # 속도값은 계산하지만 에이전트에게는 전달하지 않음
                obs = {
                    "joint_pos": joint_rad,
                    "joint_vel": [0.0] * 6,   # 완전 차단
                    "cube_pos":  cube_pos,
                }

                # obs 로그 기록
                row = [str(step)]
                row += [f"{v:.6f}" for v in joint_rad]
                row += [f"{v:.6f}" for v in joint_vel]
                row += [f"{v:.6f}" for v in cube_pos]
                log_f.write("\t".join(row) + "\n")
                log_f.flush()

                send_packet(sock, obs)
                resp = receive_packet(sock)
                if resp is None:
                    print("[Jetson] Server disconnected.")
                    break

                apply_action(resp["action"])

                prev_rad = joint_rad
                prev_t   = t0
                step    += 1
                time.sleep(max(0.0, period - (time.time() - t0)))

    except KeyboardInterrupt:
        print("\n[Jetson] Stopped by user.")
    except Exception as e:
        print(f"[Jetson] Error: {e}")
    finally:
        sock.close()
        log_f.close()


if __name__ == "__main__":
    main()
