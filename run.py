
from upkie.utils.raspi import configure_agent_process, on_raspi

if on_raspi():
    configure_agent_process()
    reg_freq = False
else:
    reg_freq = False

from tqdm import tqdm
import numpy as np
from env.envs import make_rays_pink_env
from loop_rate_limiters import RateLimiter
import gymnasium as gym
import upkie.envs
import gin
from config.settings import EnvSettings
from upkie.utils.robot_state import RobotState

upkie.envs.register()

gin.parse_config_file(f"config/settings.gin")
env_settings = EnvSettings()

gym.envs.registration.register(
    id="UpkieServos-v5", entry_point="env.upkie_servos:UpkieServos"
)

agent_frequency = env_settings.agent_frequency
max_episode_duration = 25000
spine_config = env_settings.spine_config
spine_config["base_orientation"]  = {'rotation_base_to_imu':np.array([1,0,0,0,-1,0,0,0,-1],dtype=float)}
velocity_env = gym.make(
    env_settings.env_id,
    max_episode_steps=int(max_episode_duration * agent_frequency),
    frequency=100,
    regulate_frequency=reg_freq,
    shm_name="upkie",
    spine_config=spine_config,
    fall_pitch=np.pi / 2,
    init_state=RobotState(position_base_in_world=np.array([2, 2, 0.58]))
)

env = make_rays_pink_env(
    velocity_env,
    env_settings,
    eval_mode=False,
)


def main():
    use_vs = False
    rate_limiter = RateLimiter(frequency=10)
    s, i = env.reset()

    for _ in tqdm(range(200000)):
        rate_limiter.sleep()
        
        # Send obstacle points
        if use_vs : 
            vs_twist  = i['vs_twist']
            action = np.array([vs_twist])
            print('twist', action)
        else : 
            target_forward = -i["spine_observation"]["joystick"]["left_axis"][1]
            target_yaw = -i["spine_observation"]["joystick"]["left_axis"][0]
            action = np.array([target_forward, target_yaw])
        
        s, r, d, t, i = env.step(action)
        if i["spine_observation"]["joystick"]["triangle_button"] or d:
            obs, i = env.reset()
        if i["spine_observation"]["joystick"]["cross_button"] or d:
            print('took a picture!!!!! ')
            env.request_reinit = True
        if i["spine_observation"]["joystick"]["square_button"] or d:
            if use_vs : 
                print("use vs set to False")
                use_vs = False
            else : 
                print('use vs set to true')
                use_vs=True


if __name__ == "__main__":
    main()
