import logging
import numpy as np
from .buffer import ReplayBuffer, PrioritizedReplayBuffer
from copy import deepcopy
from .utils import visualize
from tqdm import trange
logger = logging.getLogger(__name__)


def eval(env, agent, episodes, seed):
    returns = []
    timestep2state = lambda x: np.concatenate([x.observation[k].flatten() for k in x.observation.keys()][:1]).astype(np.float32)
    for episode in range(episodes):
        env._random_state = seed + episode
        timestep = env.reset()
        state = timestep2state(timestep)
        done, truncated = False, False
        reward_all = 0

        while not done:
            # print(agent.get_action(state))
            timestep = env.step(agent.get_action(state))
            state = timestep2state(timestep)
            reward = timestep.reward
            done = timestep.last()
            reward_all += reward
        returns.append(reward)
    return np.mean(returns), np.std(returns)


def train(cfg, env, eval_env, agent, buffer, seed, log_dict):
    timestep2state = lambda x: np.concatenate([x.observation[k].flatten() for k in x.observation.keys()][:1]).astype(np.float32)
    for key in log_dict.keys():
        log_dict[key].append([])

    done, best_reward = False, -np.inf
    env._random_state = seed
    timestep = env.reset()
    state = timestep2state(timestep)
    reward_all = 0
    for step in trange(1, cfg.timesteps + 1):
        if done:
            env._random_state = seed
            timestep = env.reset()
            state = timestep2state(timestep)
            done = False
            log_dict['train_returns'][-1].append(reward_all)
            log_dict['train_steps'][-1].append(step - 1)
            reward_all = 0

        action = agent.get_action(state, sample=True)
        timestep = env.step(action)
        next_state = timestep2state(timestep)
        reward = timestep.reward
        done = timestep.last()
        reward_all += reward

        buffer.add((state, action, reward, next_state, int(done)))
        state = next_state

        if step > cfg.batch_size + cfg.nstep:
            if isinstance(buffer, PrioritizedReplayBuffer):
                batch, weights, tree_idxs = buffer.sample(cfg.batch_size)
                ret_dict = agent.update(batch, weights=weights)
                buffer.update_priorities(tree_idxs, ret_dict['td_error'])

            elif isinstance(buffer, ReplayBuffer):
                batch = buffer.sample(cfg.batch_size)
                ret_dict = agent.update(batch)
            else:
                raise RuntimeError("Unknown buffer")

            for key in ret_dict.keys():
                log_dict[key][-1].append(ret_dict[key])

        if step % cfg.eval_interval == 0:
            eval_mean, eval_std = eval(eval_env, agent=agent, episodes=cfg.eval_episodes, seed=seed)
            log_dict['eval_steps'][-1].append(step - 1)
            log_dict['eval_returns'][-1].append(eval_mean)
            logger.info(f"Seed: {seed}, Step: {step}, Eval mean: {eval_mean}, Eval std: {eval_std}")
            if eval_mean > best_reward:
                best_reward = eval_mean
                agent.save(f'best_model_seed_{seed}.pt')

        if step % cfg.plot_interval == 0:
            visualize(step, f'{agent} with {buffer}', log_dict)

    agent.save(f'final_model_seed_{seed}.pt')
    visualize(step, f'{agent} with {buffer}', log_dict)

    # env = RecordVideo(eval_env, f'final_videos_seed_{seed}', name_prefix='eval', episode_trigger=lambda x: x %
    #                   3 == 0 and x < cfg.eval_episodes, disable_logger=True)
    eval_mean, eval_std = eval(env, agent=agent, episodes=cfg.eval_episodes)

    agent.load(f'best_model_seed_{seed}.pt')  # use best model for visualization
    # env = RecordVideo(eval_env, f'best_videos_seed_{seed}', name_prefix='eval', episode_trigger=lambda x: x %
    #                   3 == 0 and x < cfg.eval_episodes, disable_logger=True)
    eval_mean, eval_std = eval(env, agent=agent, episodes=cfg.eval_episodes)
    # merge_videos(f'final_videos_seed_{seed}')
    # merge_videos(f'best_videos_seed_{seed}')
    env.close()
    return eval_mean
