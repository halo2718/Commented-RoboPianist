import os
import argparse
import numpy as np
import torch
import gym
from gym import wrappers

os.environ['MUJOCO_GL']="egl"
from IPython.display import HTML
from base64 import b64encode
import numpy as np
from robopianist.suite.tasks import self_actuated_piano
from robopianist.suite.tasks import piano_with_shadow_hands
from robopianist.suite.tasks import piano_with_one_shadow_hand
from dm_env_wrappers import CanonicalSpecWrapper
from robopianist.wrappers import PianoSoundVideoWrapper
from robopianist import music
from mujoco_utils import composer_utils
from robopianist.models.hands import HandSide
import dm_env

from model import GaussianPolicy
from utils import grad_false

def play_video(filename: str):
    mp4 = open(filename, "rb").read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()

    return HTML(
        """
  <video controls>
        <source src="%s" type="video/mp4">
  </video>
  """
        % data_url
    )

def get_observations(timestep):
    obs = []
    for k, v in timestep.observation.items():
        if not len(obs):
            obs = v
        else:
            obs = np.concatenate((obs, v), axis=-1)
    return np.array(obs)

def get_env():
    # task = piano_with_one_shadow_hand.PianoWithOneShadowHand(
    #     change_color_on_activation=True,
    #     midi=music.load("TwinkleTwinkleRousseau"),
    #     trim_silence=True,
    #     control_timestep=0.05,
    #     gravity_compensation=True,
    #     primitive_fingertip_collisions=False,
    #     reduced_action_space=False,
    #     n_steps_lookahead=10,
    #     disable_fingering_reward=False,
    #     disable_colorization=False,
    #     attachment_yaw=0.0,
    #     hand_side=HandSide.RIGHT
    # )
    task = self_actuated_piano.SelfActuatedPiano(
        midi=music.load("TwinkleTwinkleRousseau"),
        change_color_on_activation=True,
        trim_silence=True,
        control_timestep=0.01,
    )

    env = composer_utils.Environment(
        task=task, strip_singleton_obs_buffer_dim=True, recompile_physics=False
    )

    env = PianoSoundVideoWrapper(
        env,
        record_every=1,
        camera_id="piano/back",
        record_dir="./test_res",
    )

    env = CanonicalSpecWrapper(env)
    return env

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, default='HalfCheetah-v2')
    parser.add_argument('--log_name', type=str, default='sac-seed0-datetime')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    log_dir = os.path.join('logs', args.env_id, args.log_name)

    env = get_env()
    device = torch.device(
        "cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    action_shape = env.action_spec().shape
    observ_shape = get_observations(env.reset()).shape
    policy = GaussianPolicy(
        observ_shape[0],
        action_shape[0],
        hidden_units=[256, 256, 256]).to(device)

    model_path = 'runs/droq-actuate/droq-actuate_2023-06-03T09:00:22.588338'
    policy.load(os.path.join(model_path, 'model', 'policy.pth'))
    grad_false(policy)

    def exploit(state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            _, _, action = policy.sample(state)
        return action.cpu().numpy().reshape(-1)

    state = get_observations(env.reset())
    episode_reward = 0.
    done = False
    while not done:
        action = exploit(state)
        timestep = env.step(action)
        next_state = get_observations(timestep)
        reward = timestep.reward
        done = timestep.last()
        episode_reward += reward
        state = next_state


if __name__ == '__main__':
    run()
