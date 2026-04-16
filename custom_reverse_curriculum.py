import h5py
import numpy as np
import torch
from collections import deque

# Demo마다 정보 저장
class DemoCurriculumMetadata:
    def __init__(self, total_steps: int, per_demo_buffer_size: int = 3):
        self.total_steps = total_steps
        # Initialize start_step to the last step (end of demo, goal state)
        self.start_step = max(total_steps - 1, 0)   # 초기 start step = Demo의 마지막 Step
        self.success_rate_buffer = deque([0] * per_demo_buffer_size, maxlen=per_demo_buffer_size)
        self.episode_steps_back = deque([-1] * per_demo_buffer_size, maxlen=per_demo_buffer_size)
        self.solved = False

class ReverseCurriculumManager:
    """
    Reverse Curriculum Generation buffer based on demonstration trajectories.
    Corresponds to RFCL's Demo Setup, holding demo trajectories and curriculum state.
    """
    def __init__(
        self,
        h5_path: str,
        reverse_step_size: int = 8,
        per_demo_buffer_size: int = 24,
        threshold: float = 1.0,
    ):
        self.reverse_step_size = reverse_step_size
        self.per_demo_buffer_size = per_demo_buffer_size    # 성공률 평가 buffer 길이
        self.threshold = threshold
        
        self.demos = {} # joint position trajectory
        self.cube_demos = {}   # cube_pos(3) + cube_quat(4) + cube_lin_vel(3) + cube_ang_vel(3) = (13,)
        self.joint_vel_demos = {} # joint_vel(7)
        self.prev_action_demos = {} # prev_action(7)
        self.demo_metadata = {} # 각 Demo의 Curriculum State
        
        print(f"[RCG] Loading demos from {h5_path}")
        with h5py.File(h5_path, "r") as f:
            for key in f.keys():
                traj = f[key]
                # shape: [T+1, 27]
                env_states = np.array(traj["env_states"])
                # Extract robot joint pos [0:7]
                joint_states = env_states[:, 0:7]
                
                demo_id = len(self.demos)
                self.demos[demo_id] = torch.tensor(joint_states, dtype=torch.float32)
                
                # Extract joint velocities: index [7:14]
                joint_vel_states = env_states[:, 7:14]
                self.joint_vel_demos[demo_id] = torch.tensor(joint_vel_states, dtype=torch.float32)
                
                # Extract full cube state: pos(3) + quat(4) + lin_vel(3) + ang_vel(3) at indices [14:27]
                cube_states = env_states[:, 14:27]
                self.cube_demos[demo_id] = torch.tensor(cube_states, dtype=torch.float32)

                # Extract prev_action: index [33:40]
                prev_actions = env_states[:, 33:40]
                self.prev_action_demos[demo_id] = torch.tensor(prev_actions, dtype=torch.float32)

                self.demo_metadata[demo_id] = DemoCurriculumMetadata(
                    total_steps=len(joint_states),
                    per_demo_buffer_size=per_demo_buffer_size
                )
        
        self.demo_ids = list(self.demos.keys())
        print(f"[RCG] Loaded {len(self.demo_ids)} demos.")

    def record(self, demo_id: int, steps_back: int, success: bool):
        """
        Record the success of an episode to evaluate the start_step frontier.
        """
        if demo_id not in self.demo_metadata:
            return
            
        metadata = self.demo_metadata[demo_id]
        
        # Only record successes if they belong to the current frontier
        current_steps_back = metadata.total_steps - metadata.start_step
        if steps_back == current_steps_back:
            metadata.success_rate_buffer.append(1 if success else 0)
            metadata.episode_steps_back.append(steps_back)
            
            # Step curriculum per-demo
            running_success_rate = sum(metadata.success_rate_buffer) / len(metadata.success_rate_buffer)    # 최근 Buffer 평균 성공률 계산
            if running_success_rate >= self.threshold:  # train.py에서는 threshold = 0.9 (90% 이상 성공률이면 frontier 이동)
                # reset buffer
                for _ in range(self.per_demo_buffer_size):
                    metadata.success_rate_buffer.append(0)
                    metadata.episode_steps_back.append(-1)
                
                if metadata.start_step > 0:
                    metadata.start_step = max(metadata.start_step - self.reverse_step_size, 0)
                    print(f"[RCG] Demo {demo_id} stepping back to {metadata.start_step}")
                else:
                    if not metadata.solved:
                        print(f"[RCG] Demo {demo_id} is reverse solved!")
                        metadata.solved = True


    def generate_next(self, device: str):
        """
        Generate a start state for the next episode.
        Return shape: (joint_pos: [7], demo_id, steps_back)
        """
        # 원본 rfcl: demo_id_density = t_i / total_steps (더 많이 뒤로 간 데모 → 더 자주 샘플링)
        densities = np.zeros(len(self.demo_ids))
        for i, did in enumerate(self.demo_ids):
            md = self.demo_metadata[did]
            t_i = md.start_step
            densities[i] = t_i / md.total_steps
            if t_i == 0:
                densities[i] = 1e-6  # solved 데모도 극소 확률로 뽑힘
        densities = densities / densities.sum()

        demo_id = np.random.choice(self.demo_ids, p=densities)
        metadata = self.demo_metadata[demo_id]
        
        # geometric sampling
        x_start_steps_density_list = [0.5, 0.25, 0.125, 0.0625, 0.0625] # 기하 분포 (frontier 근처 5개 step)
        
        candidates = []
        densities = []
        # Gather up to 5 steps starting from current frontier
        for i in range(5):
            candidates.append(min(metadata.start_step + i, metadata.total_steps - 1))
            densities.append(x_start_steps_density_list[i])
            
        start_step = np.random.choice(candidates, p=densities)
        steps_back = metadata.total_steps - start_step
        
        joint_state = self.demos[demo_id][start_step].to(device)
        joint_vel_state = self.joint_vel_demos[demo_id][start_step].to(device)
        cube_state = self.cube_demos[demo_id][start_step].to(device)
        prev_action = self.prev_action_demos[demo_id][start_step].to(device)
        return joint_state, joint_vel_state, cube_state, prev_action, demo_id, steps_back

    def save_state(self):
        saved_md = {}
        for k, v in self.demo_metadata.items():
            saved_md[k] = {
                "start_step": v.start_step,
                "solved": v.solved,
                "success_rate_buffer": list(v.success_rate_buffer),
                "episode_steps_back": list(v.episode_steps_back)
            }
        return {"demo_metadata": saved_md}

    def load_state(self, state_dict):
        saved_md = state_dict.get("demo_metadata", {})
        for k, v in saved_md.items():
            if k in self.demo_metadata:
                self.demo_metadata[k].start_step = v["start_step"]
                self.demo_metadata[k].solved = v["solved"]
                self.demo_metadata[k].success_rate_buffer = deque(v["success_rate_buffer"], maxlen=self.per_demo_buffer_size)
                self.demo_metadata[k].episode_steps_back = deque(v["episode_steps_back"], maxlen=self.per_demo_buffer_size)

    def log_state(self):
        # average start_step frac
        fracs = []
        solved_count = 0
        details = {}
        for k, v in self.demo_metadata.items():
            if v.solved:
                solved_count += 1
                
            if v.total_steps > 1:
                f = v.start_step / (v.total_steps - 1)
            else:
                f = 0.0
            fracs.append(f)
            details[f"demo_{k}"] = f
        
        return {
            "mean_start_step_frac": sum(fracs) / len(fracs) if fracs else 0.0,
            "solved_frac": solved_count / len(self.demo_metadata) if self.demo_metadata else 0.0,
            "details": details
        }
