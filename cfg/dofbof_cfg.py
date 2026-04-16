import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOFBOT_USD_PATH = os.path.join(BASE_DIR, "..", "usd", "dofbot.usd")

# -----------------------------------------------------------------------------
# Servo spec reference
# -----------------------------------------------------------------------------
# 15 kg*cm ~= 1.47 N*m
#  6 kg*cm ~= 0.59 N*m

##  30cm에서 이상 없는 파라미터 ##
##  20cm에서는 Joint 터짐     ## -- Dual, Single Contact Reward는 받음
# ARM_SERVO_TORQUE = 5.2  # 1.47 ->
# SERVO_VELOCITY_LIMIT = 3.0
# FINGER_SERVO_TORQUE = 0.59
# stiffness=15.0, 0.65
# armature=0.01(joint)
# armature=0.03(wrist, finger)
# friction=0.0
##############################

# ARM_SERVO_TORQUE = 1.47
# SERVO_VELOCITY_LIMIT = 3.0
# FINGER_SERVO_TORQUE = 0.59
# stiffness=15.0, 0.65
# armature=0.01(joint)
# armature=0.03(wrist, finger)
# friction=0.0
##############################

#0329
ARM_SERVO_TORQUE = 3.0    # 5.2 ->  3.0: contact 시 충격 완화
SERVO_VELOCITY_LIMIT = 4.0
FINGER_SERVO_TORQUE = 1.0  # 2.0 -> 1.0: 손가락 contact 충격 완화

DOFBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=DOFBOT_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=0.5,  # 1.0 -> 0.5: contact 시 충격 완화
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,   # True ->
            solver_position_iteration_count=32, # 16 ->
            solver_velocity_iteration_count=8,  # 4 ->
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": -1.1,
            "joint4": -1.5,
            "Wrist_Twist_RevoluteJoint": 0.0,
            "Finger_Left_01_RevoluteJoint": 0.0,
            "Finger_Right_01_RevoluteJoint": 0.0,
        },
        pos=(0.0, 0.0, 0.03),
    ),
    actuators={
        "joint1": DCMotorCfg(
            joint_names_expr=["joint1"],
            # stiffness=10000.0,
            # damping=100.0,
            stiffness=10.4,
            damping=2.0,   # 0.7 -> 2.0: 임계댐핑(0.91) 초과로 과소댐핑 해소
            effort_limit=ARM_SERVO_TORQUE,
            effort_limit_sim=ARM_SERVO_TORQUE,
            saturation_effort=ARM_SERVO_TORQUE,
            velocity_limit=SERVO_VELOCITY_LIMIT,
            velocity_limit_sim=SERVO_VELOCITY_LIMIT,
            armature=0.02,
            friction=0.0,
        ),
        "joint2": DCMotorCfg(
            joint_names_expr=["joint2"],
            # stiffness=10000.0,
            # damping=100.0,
            stiffness=10.4,
            damping=2.0,   # 0.7 -> 2.0: 임계댐핑(0.91) 초과로 과소댐핑 해소
            effort_limit=ARM_SERVO_TORQUE,
            effort_limit_sim=ARM_SERVO_TORQUE,
            saturation_effort=ARM_SERVO_TORQUE,
            velocity_limit=SERVO_VELOCITY_LIMIT,
            velocity_limit_sim=SERVO_VELOCITY_LIMIT,
            armature=0.02,
            friction=0.0,
        ),
        "joint3": DCMotorCfg(
            joint_names_expr=["joint3"],
            # stiffness=10000.0,
            # damping=100.0,
            stiffness=10.4,
            damping=2.0,   # 0.7 -> 2.0: 임계댐핑(0.91) 초과로 과소댐핑 해소
            effort_limit=ARM_SERVO_TORQUE,
            effort_limit_sim=ARM_SERVO_TORQUE,
            saturation_effort=ARM_SERVO_TORQUE,
            velocity_limit=SERVO_VELOCITY_LIMIT,
            velocity_limit_sim=SERVO_VELOCITY_LIMIT,
            armature=0.02,
            friction=0.0,
        ),
        "joint4": DCMotorCfg(
            joint_names_expr=["joint4"],
            # stiffness=10000.0,
            # damping=100.0,
            stiffness=10.4,
            damping=2.0,   # 0.7 -> 2.0: 임계댐핑(0.91) 초과로 과소댐핑 해소
            effort_limit=ARM_SERVO_TORQUE,
            effort_limit_sim=ARM_SERVO_TORQUE,
            saturation_effort=ARM_SERVO_TORQUE,
            velocity_limit=SERVO_VELOCITY_LIMIT,
            velocity_limit_sim=SERVO_VELOCITY_LIMIT,
            armature=0.02,
            friction=0.0,
        ),
        "Wrist_Twist_RevoluteJoint": DCMotorCfg(
            joint_names_expr=["Wrist_Twist_RevoluteJoint"],
            stiffness=10.4,
            damping=2.0,   # 0.7 -> 2.0: 팔 관절과 동일하게 과소댐핑 해소
            effort_limit=ARM_SERVO_TORQUE,
            effort_limit_sim=ARM_SERVO_TORQUE,
            saturation_effort=ARM_SERVO_TORQUE,
            velocity_limit=4.0,
            velocity_limit_sim=4.0,
            armature=0.02,
            friction=0.0,
        ),
        "Finger_Left_01_RevoluteJoint": DCMotorCfg(
            joint_names_expr=["Finger_Left_01_RevoluteJoint"],
            stiffness=2.0,
            damping=1.2,   # 0.49 → 1.2: 임계댐핑(0.63) 초과 -> contact velocity spike 억제
            effort_limit=FINGER_SERVO_TORQUE,
            effort_limit_sim=FINGER_SERVO_TORQUE,
            saturation_effort=FINGER_SERVO_TORQUE,
            velocity_limit=4.0,
            velocity_limit_sim=4.0,
            armature=0.05,
            friction=0.0,
        ),
        "Finger_Right_01_RevoluteJoint": DCMotorCfg(
            joint_names_expr=["Finger_Right_01_RevoluteJoint"],
            stiffness=2.0,
            damping=1.2,   # 0.49 → 1.2: 임계댐핑(0.63) 초과 -> contact velocity spike 억제
            effort_limit=FINGER_SERVO_TORQUE,
            effort_limit_sim=FINGER_SERVO_TORQUE,
            saturation_effort=FINGER_SERVO_TORQUE,
            velocity_limit=4.0,
            velocity_limit_sim=4.0,
            armature=0.05,
            friction=0.0,
        ),
    },
)