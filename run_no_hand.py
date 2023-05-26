

import os
os.environ['MUJOCO_GL']="egl"

from IPython.display import HTML
from base64 import b64encode
import numpy as np
from robopianist.suite.tasks import self_actuated_piano
from robopianist.suite.tasks import piano_with_shadow_hands
from dm_env_wrappers import CanonicalSpecWrapper
from robopianist.wrappers import PianoSoundVideoWrapper
from robopianist import music
from mujoco_utils import composer_utils
import dm_env


def play_video(filename: str):
    mp4 = open(filename, "rb").read()
    print(filename)
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()

    return HTML(
        """
  <video controls>
        <source src="%s" type="video/mp4">
  </video>
  """
        % data_url
    )
    
task = self_actuated_piano.SelfActuatedPiano(
    midi=music.load("NocturneRousseau"),
    change_color_on_activation=True,
    trim_silence=True,
    control_timestep=0.01,
)

env = composer_utils.Environment(
    recompile_physics=False, task=task, strip_singleton_obs_buffer_dim=True
)

env = PianoSoundVideoWrapper(
    env,
    record_every=1,
    camera_id="piano/back",
    record_dir=".",
)

action_spec = env.action_spec()
min_ctrl = action_spec.minimum
max_ctrl = action_spec.maximum
print(f"Action dimension: {action_spec.shape}")
print(min_ctrl)
print(max_ctrl)

print("Observables:")
timestep = env.reset()
dim = 0
for k, v in timestep.observation.items():
    print(f"\t{k}: {v.shape} {v.dtype}")
    dim += np.prod(v.shape)
print(f"Observation dimension: {dim}")


class Oracle:
    def __call__(self, timestep: dm_env.TimeStep) -> np.ndarray:
        if timestep.reward is not None:
            assert timestep.reward == 0
        # Only grab the next timestep's goal state.
        goal = timestep.observation["goal"][: task.piano.n_keys]
        key_idxs = np.flatnonzero(goal)
        # For goal keys that should be pressed, set the action to the maximum
        # actuator value. For goal keys that should be released, set the action to
        # the minimum actuator value.
        action = min_ctrl.copy()
        action[key_idxs] = max_ctrl[key_idxs]
        # Grab the sustain pedal action.
        action[-1] = timestep.observation["goal"][-1]
        return action
    
policy = Oracle()

timestep = env.reset()
while not timestep.last():
    action = policy(timestep)
    timestep = env.step(action)
    
play_video(env.latest_filename)


