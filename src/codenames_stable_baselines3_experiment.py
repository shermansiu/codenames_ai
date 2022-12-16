import sys
import gym
import pathlib
import typing as tp
from scipy.special import softmax
import pandas as pd
import os
from default_game import *
from codenames import *
from codenames_env import *
from plot_utils import *
import stable_baselines3
from stable_baselines3.common import env_checker
from stable_baselines3 import DQN, SAC, PPO, A2C, HerReplayBuffer
import matplotlib.pyplot as plt
from stable_baselines3 import TD3
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common import env_checker

env_checker.check_env(CodenamesEnvHack(glove, wordlist))
env_checker.check_env(CodenamesEnvHackDiscrete(glove, wordlist))

from stable_baselines3.common.logger import configure
import pathlib

EXPERIMENT_TYPE = "codenames"
EXPERIMENT_NAME = "codenames"
EXPERIMENT_DISPLAY_NAME = "Codenames - Self-similarity matrix"
EXPERIMENT_DISPLAY_NAME_DISCRETE = "Codenames - One-hot word embedding"
CODENAMES_DIR = pathlib.Path(__file__).resolve().parent.parent
tmp_path = str(CODENAMES_DIR.joinpath(f"logs/{EXPERIMENT_NAME}/sb3_log/_"))[:-1]
# set up logger
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
tmp_path_discrete = str(CODENAMES_DIR.joinpath("logs/{EXPERIMENT_NAME}/sb3_log_discrete/_"))
# set up logger
new_logger_discrete = configure(tmp_path_discrete, ["stdout", "csv", "tensorboard"])


def make_env(env_name):
    log_dir = pathlib.Path(tmp_path).joinpath(env_name)
    log_dir.mkdir(parents=True, exist_ok=True)
    return Monitor(CodenamesEnvHack(glove, wordlist), str(log_dir))


def make_env_discrete(env_name):
    log_dir = pathlib.Path(tmp_path_discrete).joinpath(env_name)
    return Monitor(CodenamesEnvHackDiscrete(glove, wordlist), str(log_dir))


log_freq = 25_000
num_train_steps = 100_000


n_sampled_goal = 4
models = dict(
    sac = SAC('MultiInputPolicy', make_env("sac"), verbose=1, target_update_interval=25),
    sac_her = SAC(
        'MultiInputPolicy',
        make_env("sac_her"),
        verbose=1, target_update_interval=25,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
          n_sampled_goal=n_sampled_goal,
          goal_selection_strategy="future",
          # IMPORTANT: because the env is not wrapped with a TimeLimit wrapper
          # we have to manually specify the max number of steps per episode
          max_episode_length=25,
          online_sampling=True,
        ),
        buffer_size=int(1e6)
    ),
    ppo = PPO('MultiInputPolicy', make_env("ppo"), verbose=1, n_steps=25, n_epochs=80),
    a2c = A2C('MultiInputPolicy', make_env("a2c"), verbose=1, n_steps=25)
)
models_discrete = dict(
    sac = SAC('MultiInputPolicy', make_env_discrete("sac"), verbose=1, target_update_interval=25),
    sac_her = SAC(
        'MultiInputPolicy',
        make_env_discrete("sac_her"),
        verbose=1, target_update_interval=25,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
          n_sampled_goal=n_sampled_goal,
          goal_selection_strategy="future",
          # IMPORTANT: because the env is not wrapped with a TimeLimit wrapper
          # we have to manually specify the max number of steps per episode
          max_episode_length=25,
          online_sampling=True,
        ),
        buffer_size=int(1e6)
    ),
    ppo = PPO('MultiInputPolicy', make_env_discrete("ppo"), verbose=1, n_steps=25, n_epochs=80),
    a2c = A2C('MultiInputPolicy', make_env_discrete("a2c"), verbose=1, n_steps=25)
)

from tqdm.auto import tqdm

class ProgressBarCallback(BaseCallback):
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """
    def __init__(self, pbar):
        super(ProgressBarCallback, self).__init__()
        self._pbar = pbar

    def _on_step(self):
        # Update the progress bar:
        self._pbar.n = self.num_timesteps
        self._pbar.update(0)

# this callback uses the 'with' block, allowing for correct initialisation and destruction
class ProgressBarManager:
    def __init__(self, total_timesteps): # init object with total timesteps
        self.pbar = None
        self.total_timesteps = total_timesteps
        
    def __enter__(self): # create the progress bar and callback, return the callback
        self.pbar = tqdm(total=self.total_timesteps)
            
        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb): # close the callback
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()
        
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            try:
                x, y = ts2xy(load_results(self.log_dir), 'timesteps')
                if len(x) > 0:
                    # Mean training reward over the last 100 episodes
                    mean_reward = np.mean(y[-100:])
                    if self.verbose > 0:
                        print(f"Num timesteps: {self.num_timesteps}")
                        print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                    # New best model, you could save the agent here
                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        # Example for saving best model
                        if self.verbose > 0:
                            print(f"Saving new best model to {self.save_path}")
                        self.model.save(self.save_path)
            except Exception:
                pass
        return True


# We don't support discrete mode right now
tmp_path_discrete = tmp_path
def get_path_for(env_name, is_discrete=False):
    if is_discrete:
        return str(pathlib.Path(tmp_path_discrete).joinpath(env_name))
    return str(pathlib.Path(tmp_path).joinpath(env_name))


class Run:
    def __init__(self, name, model, total_timesteps=1, mode="self_sim", num_eval_episodes=10, should_render=False, log_interval=500):
        assert mode in {"self_sim", "discrete"}
        print("Model:", name)
        self.log_dir = get_path_for(name, mode == "discrete")
        self.save_callback = SaveOnBestTrainingRewardCallback(check_freq=log_freq, log_dir=self.log_dir)
        self.logger = configure(get_path_for(name + "_logger", mode == "discrete"), ["stdout", "csv", "tensorboard"])
        print("Start training")
        model.set_logger(self.logger)
        with ProgressBarManager(total_timesteps) as progress_callback: # this the garanties that the tqdm progress bar closes correctly
            model.learn(total_timesteps=total_timesteps, log_interval=log_interval, callback=CallbackList([progress_callback, self.save_callback]))
        # Save the agent
        print("Saving...")
        model.save(f"{EXPERIMENT_NAME}_{name}_v0.0.1_end.chkpt.zip")
        # Evaluate the trained agent
        print("Evaluating...")
        if mode == "self_sim":
            self.eval_monitor_trained = make_env(name + "_eval")
        else:
            self.eval_monitor_trained = make_env_discrete(name + "_eval")
        mean_reward, std_reward = evaluate_policy(model, self.eval_monitor_trained, n_eval_episodes=num_eval_episodes, deterministic=True, render=should_render)
        print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
        self.print_monitor_stat("Lengths", self.eval_monitor_trained.episode_lengths)
        self.print_monitor_stat("Returns", self.eval_monitor_trained.episode_returns)

    def print_monitor_stat(self, stat_name, stat):
        print(stat_name, stat, np.mean(stat), np.std(stat))


runs = {k: Run(k, m, num_train_steps, should_render=False) for k, m in models.items()}
runs_discrete = {k: Run(k, m, num_train_steps, mode="discrete", should_render=False) for k, m in models_discrete.items()}

plot_gridtest_experiment(EXPERIMENT_NAME, EXPERIMENT_DISPLAY_NAME)
plot_gridtest_experiment(EXPERIMENT_NAME, EXPERIMENT_DISPLAY_NAME_DISCRETE, sb3_log_suffix="discrete")
