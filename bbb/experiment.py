import bbb
import numpy as np

from bbb.logging import get_logger, ExperimentLogger

logger = get_logger()
experiment_logger = ExperimentLogger(logger=logger)


def run_experiment(agent, environment, heatup_episodes=0, training_episodes=100,
                   evaluation_episodes=50, with_render=False):
    run_episodes(
        agent, environment, heatup_episodes)

    training_rewards = run_episodes(
        agent, environment, training_episodes,
        with_update=True, verbose=True)

    logger.info(f'evaluation ({evaluation_episodes} episodes)...')
    agent.finalize()
    evaluation_rewards = run_episodes(
        agent, environment, evaluation_episodes,
        with_memory=False, with_render=with_render)

    logger.info(f'evaluation ({evaluation_episodes} episodes):')
    logger.info(f'\treward min  = {np.min(evaluation_rewards):.2f}')
    logger.info(f'\treward mean = {np.mean(evaluation_rewards):.2f}')
    logger.info(f'\treward max  = {np.max(evaluation_rewards):.2f}')


def run_episodes(agent, environment, num_episodes, **kwds):
    episode_rewards = []
    for episode_index in range(num_episodes):
        total_reward = run_episode(agent, environment, **kwds)
        episode_rewards.append(total_reward)
    return episode_rewards


@experiment_logger.log_episode()
def run_episode(agent, environment, max_iterations=500, with_render=False, **kwds):
    environment.reset()
    is_terminal = False
    num_iterations = 0
    total_reward = 0

    while (not is_terminal) and (num_iterations < max_iterations):
        reward, is_terminal, _ = step_episode(agent, environment, **kwds)
        total_reward += reward
        num_iterations += 1
        if with_render:
            environment.render()

    return total_reward


@experiment_logger.log_step()
def step_episode(agent, environment, with_memory=True, with_update=False,
                 with_render=False, verbose=False, **kwds):
    state = environment.state
    action, max_qvalue = agent.act(environment, with_qvalue=True)
    next_state, reward, is_terminal, _ = environment.step(action)

    if is_terminal:
        reward = 0

    if with_memory:
        agent.remember(state, action, reward, next_state, is_terminal)
    if with_update:
        agent.update(**kwds)
    return reward, is_terminal, max_qvalue

