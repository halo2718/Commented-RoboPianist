import hydra
import torch
import logging
from mytrain import utils
from hydra.utils import instantiate
from mytrain.core import train
from dotmap import DotMap
from omegaconf import OmegaConf
from mytrain.buffer import get_buffer
logger = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision('high')


import os
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


class MyBox:
    def __init__(self, action_size) -> None:
        self.high = 0.09
        self.low = 0.00
        self.shape = np.array([action_size])


@hydra.main(config_path="cfgs", config_name="config", version_base="1.3")
def main(cfg):
    # task = piano_with_one_shadow_hand.PianoWithOneShadowHand(
    #     change_color_on_activation=True,
    #     midi=music.load("WellTemperedClavierBookIiPreludeNo11InFMajor"),
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

    # eval_task = piano_with_one_shadow_hand.PianoWithOneShadowHand(
    #     change_color_on_activation=True,
    #     midi=music.load("WellTemperedClavierBookIiPreludeNo11InFMajor"),
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
        midi=music.load("NocturneRousseau"),
        change_color_on_activation=True,
        trim_silence=True,
        control_timestep=0.01,
    )
    eval_task = self_actuated_piano.SelfActuatedPiano(
        midi=music.load("NocturneRousseau"),
        change_color_on_activation=True,
        trim_silence=True,
        control_timestep=0.01,
    )

    env = composer_utils.Environment(
        task=task, strip_singleton_obs_buffer_dim=True, recompile_physics=False
    )

    eval_env = composer_utils.Environment(
        task=eval_task, strip_singleton_obs_buffer_dim=True, recompile_physics=False
    )
    eval_env.reset()

    action_spec = env.action_spec()
    print(f"Action dimension: {action_spec.shape}")

    timestep = env.reset()
    dim = 0
    for k, v in timestep.observation.items():
        print(f"\t{k}: {v.shape} {v.dtype}")
        dim += int(np.prod(v.shape))
        break
    print(f"Observation dimension: {dim}")

    state_size = dim
    action_size = int(np.prod(action_spec.shape))
    log_dict = utils.get_log_dict(cfg.agent._target_)
    action_space = MyBox(action_size)

    timestep2state = lambda x: np.concatenate([x.observation[k].flatten() for k in x.observation.keys()][:1]).astype(np.float32)
    seed = 114514

    utils.set_seed_everywhere(seed)  # TODO: 
    buffer = get_buffer(cfg.buffer, state_size=state_size, action_size=action_size, device=device, seed=seed)
    agent = instantiate(cfg.agent, state_size=state_size, action_size=action_size, action_space=action_space, device=device)
    logger.info(f"Training seed {seed} for {cfg.train.timesteps} timesteps with {agent} and {buffer}")
    # get_attr of omega_conf is slow, so we convert it to dotmap
    train_cfg = DotMap(OmegaConf.to_container(cfg.train, resolve=True))

    # TRAIN
    eval_mean = train(train_cfg, env, eval_env, agent, buffer, seed, log_dict)
    logger.info(f"Finish training seed {seed} with everage eval mean: {eval_mean}")


    env = PianoSoundVideoWrapper(
        env,
        record_every=1,
        camera_id="piano/back",
        record_dir=".",
    )

    env = CanonicalSpecWrapper(env)

    # agent.load(f'/home/hyz/hw/Commented-RoboPianist/runs/2023-05-30/15-14-43_/models/final_model_seed_{seed}.pt') 
    
    # env._random_state = seed
    # timestep = env.reset()
    # while not timestep.last():
    #     state = timestep2state(timestep)
    #     action = agent.get_action(state, sample=True)
    #     n_action = np.zeros((23,))
    #     n_action[:22] = action[:22].copy()
    #     n_action[-1] = action[-1].copy()
    #     timestep = env.step(n_action)
    #     print(timestep)

    # play_video(env.latest_filename)


if __name__ == "__main__":
    main()
