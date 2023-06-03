

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

env = PianoSoundVideoWrapper(
    env,
    record_every=1,
    camera_id="piano/back",
    record_dir=".",
)

env = CanonicalSpecWrapper(env)

action_spec = env.action_spec()
print(f"Action dimension: {action_spec.shape}")

timestep = env.reset()
# print(timestep)
# print(timestep.reward)
dim = 0
for k, v in timestep.observation.items():
    print(f"\t{k}: {v.shape} {v.dtype} {v}")
    dim += int(np.prod(v.shape))
print(f"Observation dimension: {dim}")

class Policy:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._idx = 0
        self._actions = np.load("./examples/twinkle_twinkle_actions.npy")

    def __call__(self, timestep: dm_env.TimeStep) -> np.ndarray:
        del timestep  # Unused.
        actions = self._actions[self._idx]
        print(actions)
        self._idx += 1
        return actions
    
policy = Policy()

timestep = env.reset()
while not timestep.last():
    action = policy(timestep)
    n_action = np.zeros((23,))
    n_action[:22] = action[:22].copy()
    n_action[-1] = action[-1].copy()
    timestep = env.step(n_action)
    
play_video(env.latest_filename)