import os
import argparse
import datetime
import gym
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

from agent import SacAgent

# TODO use shared util.utilTH in SAC-extention
from util.utilsTH import SparseRewardEnv

# TODO remove IQN agent part
#from IQNagent import IQNSacAgent


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

def get_env(wrap=False):
    # task = self_actuated_piano.SelfActuatedPiano(
    #     midi=music.load("TwinkleTwinkleRousseau"),
    #     change_color_on_activation=True,
    #     trim_silence=True,
    #     control_timestep=0.01,
    # )
    task = piano_with_one_shadow_hand.PianoWithOneShadowHand(
        change_color_on_activation=True,
        midi=music.load("TwinkleTwinkleRousseau"),
        trim_silence=True,
        control_timestep=0.05,
        gravity_compensation=True,
        primitive_fingertip_collisions=False,
        reduced_action_space=False,
        n_steps_lookahead=10,
        disable_fingering_reward=False,
        disable_colorization=False,
        attachment_yaw=0.0,
        hand_side=HandSide.RIGHT
    )

    env = composer_utils.Environment(
        task=task, strip_singleton_obs_buffer_dim=True, recompile_physics=False
    )

    if wrap:
        env = PianoSoundVideoWrapper(
            env,
            record_every=1,
            camera_id="piano/back",
            record_dir=".",
        )

    env = CanonicalSpecWrapper(env)
    return env

def run():
    parser = argparse.ArgumentParser()
    # replaced wtih SAC-extention args 20210705
    # parser.add_argument("-env", type=str, default="HalfCheetah-v2",
    #                     help="Environment name, default = HalfCheetahBulletEnv-v0")
    parser.add_argument('-seed', type=int, default=12345)
    #added byTH 20210705
    # common
    parser.add_argument("-info", type=str, help="Information or name of the run")
    parser.add_argument("-frames", type=int, default=2_000_000,
                        help="The amount of training interactions with the environment, default is 1mio")
    parser.add_argument("-gpu_id", type=int, default=0,
                        help="GPU device ID to be used in GPU experiment, default is 1e6")
    # evaluation
    parser.add_argument("-eval_every", type=int, default=10000,
                        help="Number of interactions after which the evaluation runs are performed, default = 1000")
    parser.add_argument("-eval_runs", type=int, default=3, help="Number of evaluation runs performed, default = 1")
    # sparse env
    parser.add_argument("-sparsity_th", type=float, default=0.0,
                        help="threshold for make reward sparse (i.e., lambda in PolyRL paper), default is 0.0")
    # stabilization
    parser.add_argument("-huber", type=int, default=0, choices=[0, 1],
                        help="Using Huber loss for training critics if set to 1 (TH), default=0") # TODO remove
    parser.add_argument("-layer_norm", type=int, default=0, choices=[0, 1],
                        help="Using layer normalization for training critics if set to 1 (TH), default=0")
    # multi-step and per
    parser.add_argument("-n_step", type=int, default=1, help="Using n-step bootstrapping, default=1")
    parser.add_argument("-per", type=int, default=0, choices=[0, 1],
                        help="Adding Priorizied Experience Replay to the agent if set to 1, default = 0")
    # dist RL added @ 20210711
    parser.add_argument("-dist", "--distributional", type=int, default=0, choices=[0, 1],
                        help="Using a distributional IQN Critic if set to 1, default=0") # TODO remove
    # learning per steps
    parser.add_argument("-updates_per_step", type=int, default=1,
                        help="Number of training updates per one environment step, default = 1")
    # th 20210724
    parser.add_argument("-target_entropy", type=float, default=None, help="target entropy , default=Num action")
    # for MBPO and redq setting, Hopper: -1, HC: -3, Walker: -3, Ant: -4, Humaoid: -2
    #
    parser.add_argument("-method", default="sac", choices=["sac", "redq", "duvn", "monosac"], help="method, default=sac")
    # learning per steps
    parser.add_argument("-batch_size", type=int, default=256,
                        help="Number of training batch, default = 256")
    #
    parser.add_argument("-target_drop_rate", type=float, default=0.0, help="drop out rate of target value function, default=0")
    #
    parser.add_argument("-critic_update_delay", type=int, default=1, help="number of critic learning delay (tau and UDP is rescaled), default=1 (no delay)") # TODO remove

    # 20210813
    # dist RL added @ 20210711
    parser.add_argument("-profile", type=int, default=0, choices=[0, 1],
                        help="Using profile for cpu/gpu speed and memory usage if set to 1, default=0")


    args = parser.parse_args()

    # You can define configs in the external json or yaml file.
    configs = {
        'num_steps': args.frames, #3000000,
        'batch_size': args.batch_size, #, 256,
        'lr': 0.0003,
        'hidden_units': [256, 256, 256],
        'memory_size': 1e6,
        'gamma': 0.99,
        'tau': 0.005,
        'entropy_tuning': True,
        'ent_coef': 0.2,  # It's ignored when entropy_tuning=True.
        'multi_step': args.n_step, #1,
        'per': args.per, #False,  # prioritized experience replay
        'alpha': 0.6,  # It's ignored when per=False.
        'beta': 0.4,  # It's ignored when per=False.
        'beta_annealing': (1.0 - 0.4) / (1.0 * args.frames * args.updates_per_step), # 0.0001,  # It's ignored when per=False.
        'grad_clip': None,
        'updates_per_step': args.updates_per_step * args.critic_update_delay, # args.updates_per_step, #1,
        'start_steps': 5000, #10000,
        'log_interval': 10,
        'target_update_interval': 1,
        'eval_interval': args.eval_every, # 10000,
        'cuda': args.gpu_id, # args.cuda,
        'seed': args.seed,

        # adde by TH
        'eval_runs': args.eval_runs,
        'huber': args.huber, # TODO remove
        'layer_norm': args.layer_norm,
        #
        'target_entropy': args.target_entropy,
        'method': args.method,
        'target_drop_rate': args.target_drop_rate,
        'critic_update_delay': args.critic_update_delay
    }

    train_env = get_env()
    eval_env = get_env(wrap=True)

    # action_spec = env.action_spec()
    # print(f"Action dimension: {action_spec.shape}")

    # timestep = env.reset()
    # dim = 0
    # for k, v in timestep.observation.items():
    #     print(f"\t{k}: {v.shape} {type(v)} {v}")
    #     dim += int(np.prod(v.shape))
    # print(f"Observation dimension: {dim}")

    # make sparse en: TH 20210705
    # if args.sparsity_th > 0.0 :
    #     print("Evaluation in sparse reward setting with lambda = " + str(args.sparsity_th))
    #     env = SparseRewardEnv(env, rew_thresh=args.sparsity_th)
    #     env._max_episode_steps = env.wrapped_env._max_episode_steps

    label = args.info + "_" + str(datetime.datetime.now().isoformat())
    log_dir = os.path.join('runs', args.info, label)
    agent = SacAgent(train_env=train_env, eval_env=eval_env, log_dir=log_dir, **configs)

    # if args.distributional: # TODO remove
    #     raise NotImplementedError()
    #     #print(" Use IQN agent")
    #     #agent = IQNSacAgent(env=env, log_dir=log_dir, **configs)
    # else:
    #     if args.profile:
    #         pass
    #         # agent = SacAgent4Profile(env=env, log_dir=log_dir, **configs)
    #     else:
    #         agent = SacAgent(env=env, log_dir=log_dir, **configs)
    agent.run()


if __name__ == '__main__':
    run()
