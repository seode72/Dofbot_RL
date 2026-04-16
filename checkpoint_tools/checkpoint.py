"""
checkpoint.py — SAC/RFCL 전용 체크포인트 유틸리티
=================================================
standalone SAC 에이전트의 저장/로드/검색 기능.
"""
import os
import re   # 정규표현식(Regular Expression) 모듈 -- 문자열에서 숫자 등 추출
import torch


def find_latest_checkpoint(experiment_root: str) -> str | None:
    """Recursively find the latest checkpoint under experiment root."""
    if not os.path.exists(experiment_root):
        return None

    ckpts: list[tuple[int, str]] = []

    for root, _, files in os.walk(experiment_root):
        for f in files:
            if not f.endswith(".pt"):
                continue
            if f == "best_agent.pt":
                continue
            if "_memory" in f:
                continue

            if f.startswith("agent_") or f.startswith("checkpoint_"):
                numbers = re.findall(r"\d+", f)
                if numbers:
                    step = int(numbers[-1])
                    ckpts.append((step, os.path.join(root, f)))

    if not ckpts:
        return None

    ckpts.sort(key=lambda x: x[0])
    return ckpts[-1][1]


def get_step_from_checkpoint_path(checkpoint_path: str) -> int:
    """Extract total step from checkpoint filename."""
    filename = os.path.basename(checkpoint_path)
    matches = re.findall(r"\d+", filename)
    if not matches:
        return 0
    return int(matches[-1])


def load_checkpoint(agent, checkpoint_path: str, device: str) -> tuple[int, int]:
    """Load SAC checkpoint and return (resumed_step, completed_eps).

    standalone SAC 에이전트의 checkpoint_modules를 순회하며
    일치하는 키가 있으면 state_dict를 로드한다.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid checkpoint format: expected dict, got {type(checkpoint)}")

    print(f"[INFO] Loading SAC checkpoint from: {checkpoint_path}")
    print(f"[INFO] Checkpoint keys: {list(checkpoint.keys())}")

    # checkpoint_modules에 등록된 모든 모듈을 순회하며 로드
    loaded_count = 0
    if hasattr(agent, "checkpoint_modules"):
        for name, module in agent.checkpoint_modules.items():
            if name in checkpoint and module is not None:
                if hasattr(module, "load_state_dict"):
                    try:
                        module.load_state_dict(checkpoint[name])
                        print(f"[INFO] Loaded SAC module: {name}")
                        loaded_count += 1
                    except Exception as e:
                        print(f"[ERROR] Failed to load checkpoint for '{name}': {e}")
                else:
                    print(f"[WARN] Module '{name}' found in agent but does not have load_state_dict()")
            elif name in agent.checkpoint_modules and name not in checkpoint:
                print(f"[DEBUG] Module '{name}' not found in checkpoint file (expected if not saved yet)")
    
    print(f"[INFO] Total SAC modules loaded: {loaded_count}")

    step = int(checkpoint.get("step", get_step_from_checkpoint_path(checkpoint_path)))
    completed_eps = int(checkpoint.get("completed_eps", 0))
    current_stage = checkpoint.get("current_stage", None)
    print(f"[INFO] Resumed step: {step}, completed_eps: {completed_eps}, current_stage: {current_stage}")
    print("[INFO] SAC checkpoint load finished")

    return step, completed_eps, current_stage


# Agent의 경험 데이터 (Replay Buffer 등) 불러오기 -> 현재 메모리에 복원
def load_memory(memory, memory_path: str, device: str) -> bool:
    """Load skrl memory states from a file."""
    if not os.path.exists(memory_path):
        print(f"[WARN] Memory file not found: {memory_path}")
        return False
    
    try:
        checkpoint = torch.load(memory_path, map_location=device)
        
        # skrl memory attribute auto-detection
        target_dict = None
        for attr in ["memory", "_memory", "tensors_dict", "tensors"]:
            if hasattr(memory, attr):
                target_dict = getattr(memory, attr)
                break
        
        if target_dict is None:
            print(f"[ERROR] Could not find storage attribute in memory object! Available: {dir(memory)}")
            return False

        if "memory" in checkpoint:
            # skrl memory.memory is often a dict of tensors
            for name, tensor in checkpoint["memory"].items():
                if name in target_dict:
                    if torch.isnan(tensor).any():
                        print(f"[ERROR] NaN detected in memory tensor '{name}'! Skipping this tensor.")
                        continue
                    # Ensure shapes match before copy
                    if target_dict[name].shape == tensor.shape:
                        target_dict[name].copy_(tensor)
                    else:
                        print(f"[WARN] Shape mismatch for '{name}': {target_dict[name].shape} vs {tensor.shape}")
            
            # Restore internal pointers
            for attr in ["memory_index", "filled"]:
                if hasattr(memory, attr):
                    setattr(memory, attr, checkpoint.get(attr, checkpoint.get(f"_{attr}", getattr(memory, attr))))
                elif hasattr(memory, f"_{attr}"):
                    setattr(memory, f"_{attr}", checkpoint.get(attr, checkpoint.get(f"_{attr}", getattr(memory, f"_{attr}"))))
                
            print(f"[INFO] Memory loaded successfully from: {memory_path}")
            return True
    except Exception as e:
        print(f"[ERROR] Failed to load memory: {e}")
    return False


def save_checkpoint(agent, checkpoint_path: str, step: int, completed_eps: int = 0, current_stage: int | None = None, best_succ_rate: float | None = None) -> None:
    """Save SAC checkpoint.

    standalone SAC 에이전트의 checkpoint_modules를 순회하며
    state_dict가 있는 모듈은 모두 저장한다.
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    payload = {"step": int(step), "completed_eps": int(completed_eps)}
    if current_stage is not None:
        payload["current_stage"] = int(current_stage)
    if best_succ_rate is not None:
        payload["best_succ_rate"] = float(best_succ_rate)

    if hasattr(agent, "checkpoint_modules"):
        for name, module in agent.checkpoint_modules.items():
            if module is not None and hasattr(module, "state_dict"):
                payload[name] = module.state_dict()

    torch.save(payload, checkpoint_path)
    print(f"[INFO] SAC checkpoint saved: {checkpoint_path}")
# Agent의 Replay Memory 전체를 파일로 백업
def save_memory(memory, memory_path: str) -> None:
    """Save skrl memory states to a file."""
    try:
        os.makedirs(os.path.dirname(memory_path), exist_ok=True)
        
        # skrl memory attribute auto-detection
        target_dict = None
        for attr in ["memory", "_memory", "tensors_dict", "tensors"]:
            if hasattr(memory, attr):
                target_dict = getattr(memory, attr)
                break
        
        if target_dict is None:
            raise AttributeError(f"Could not find storage attribute in memory object. Available: {dir(memory)}")

        payload = {
            "memory": target_dict,
            "memory_index": getattr(memory, "memory_index", getattr(memory, "_memory_index", 0)),
            "filled": getattr(memory, "filled", getattr(memory, "_filled", False))
        }
        torch.save(payload, memory_path)
        print(f"[INFO] Memory saved successfully: {memory_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save memory: {e}")
