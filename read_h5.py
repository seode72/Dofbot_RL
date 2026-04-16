import h5py, numpy as np

with h5py.File("scripts/dofbot_0412/demos/demos.h5", "r") as f:
    print(list(f.keys()))            # ['traj_0']

    traj = f["traj_0"]
    print(list(traj.keys()))
    # ['actions', 'env_states', 'obs', 'rewards', 'success']

    print("obs shape:       ", np.array(traj["obs"]).shape)        # [T+1, 35]
    print("actions shape:   ", np.array(traj["actions"]).shape)    # [T, 7]
    print("success shape:   ", np.array(traj["success"]).shape)    # [T]
    print("env_states shape:", np.array(traj["env_states"]).shape) # [T+1, 27]

    # 성공 스텝 수 확인
    success = np.array(traj["success"])
    print("success steps:", success.sum(), "/", len(success))

    # env_states 해석 (27-dim)
    # [0:7]   = robot joint_pos
    # [7:14]  = robot joint_vel
    # [14:17] = cube local_pos (env_origin 기준)
    # [17:21] = cube quaternion
    # [21:24] = cube linear_vel
    # [24:27] = cube angular_vel
    states = np.array(traj["env_states"])
    print("cube pos at step 0:", states[0, 14:17])
