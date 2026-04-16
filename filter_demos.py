import h5py
import numpy as np
import os
import shutil
import json
import pickle

def filter_demos():
    source_dir = "/home/pnudtn11/IsaacLab/scripts/dofbot_0412/demos"
    target_dir = "/home/pnudtn11/IsaacLab/scripts/dofbot_0412/demos_filtered"
    
    source_h5 = os.path.join(source_dir, "demos.h5")
    target_h5 = os.path.join(target_dir, "demos.h5")
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"[INFO] Created directory: {target_dir}")

    # 1. H5 filtering
    print(f"[INFO] Filtering {source_h5} ...")
    with h5py.File(source_h5, "r") as f_src:
        with h5py.File(target_h5, "w") as f_dst:
            for traj_name in f_src.keys():
                print(f"  Processing {traj_name}...")
                g_src = f_src[traj_name]
                g_dst = f_dst.create_group(traj_name)
                
                # Copy actions, success, rewards, env_states as is
                for key in ["actions", "success", "rewards", "env_states"]:
                    if key in g_src:
                        g_dst.create_dataset(key, data=g_src[key][:])
                
                # Filter obs: keep indices 7 to 23 (joint_pos, joint_vel, cube_pos)
                if "obs" in g_src:
                    obs_full = g_src["obs"][:]
                    # 7:14 (joint_pos), 14:21 (joint_vel), 21:24 (cube_pos)
                    # total slice: 7:24
                    obs_filtered = obs_full[:, 7:24]
                    g_dst.create_dataset("obs", data=obs_filtered)
                    print(f"    - obs shape: {obs_full.shape} -> {obs_filtered.shape}")

    # 2. Copy JSON
    source_json = os.path.join(source_dir, "demos.json")
    target_json = os.path.join(target_dir, "demos.json")
    if os.path.exists(source_json):
        shutil.copy2(source_json, target_json)
        print(f"[INFO] Copied {source_json} -> {target_json}")

    # 3. Copy PKL
    source_pkl = os.path.join(source_dir, "states_dataset.pkl")
    target_pkl = os.path.join(target_dir, "states_dataset.pkl")
    if os.path.exists(source_pkl):
        shutil.copy2(source_pkl, target_pkl)
        print(f"[INFO] Copied {source_pkl} -> {target_pkl}")

    print("\n[SUCCESS] Filtering complete.")
    print(f"New demo files are in: {target_dir}")

if __name__ == "__main__":
    filter_demos()
