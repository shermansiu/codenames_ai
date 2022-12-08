import collections
import typing as tp
import gym
from gym import spaces
import numpy as np


class GridTestEnv(gym.Env):
    """Grid test environment for gym."""

    environment_name = "GridTest v0.0.1"
    metadata = {"render.modes": ["human"]}

    def __init__(
        self, length: int = 5, seed: tp.Optional[int] = None, max_steps: tp.Optional[int] = None
    ):
        super().__init__()
        self.length = length
        self.size = self.length*self.length
        self.action_space = spaces.Discrete(self.size)
        observation_shape = (self.length, self.length, 1)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=observation_shape, dtype=np.uint8
        )
        self.step_reward_if_win = 0
        self.step_reward_if_not_end = -1
        self.reward_range = (-1, 0)
        self.id = "GridTest"
        self.current_step = 0
        self.max_steps = max_steps if max_steps is not None else 99
        self.seed(seed)
        self.start_new_game()

    def seed(self, seed: tp.Optional[int] = None) -> tp.List[int]:
        """Seed the environment.

        Code adapted from a future version of gym.
        """
        if seed is not None and not (isinstance(seed, int) and 0 <= seed):
            raise gym.error.Error(
                f"Seed must be a non-negative integer or omitted, not {seed}"
            )

        seed_seq = np.random.SeedSequence(seed)
        seed_int = seed_seq.entropy
        self.np_random = np.random.Generator(np.random.PCG64(seed_seq))
        return [seed_int]

    def start_new_game(self):
        self.current_step = 0
        self.correct = self.np_random.integers(self.size)
        self.board = np.zeros(self.size, dtype=np.int32)
        self.board[self.correct] = 1

    def reset(self):
        # Reset the state of the environment to an initial state
        self.start_new_game()
        return self.current_observation()

    def current_observation(self):
        return np.clip(self.board.reshape(-1, self.length, 1)*128+128, 0, 255).astype(np.uint8)

    def is_correct(self):
        return bool(self.board[self.correct] == 0)

    def is_done(self):
        return self.current_step >= self.max_steps or self.is_correct()

    def compute_reward(self):
        if self.is_correct():
            return self.step_reward_if_win
        return self.step_reward_if_not_end

    def step(self, action):
        self.current_step += 1
        if not self.is_correct():
            self.board[action] = max(self.board[action], 0) - 1
        return (
            self.current_observation(),
            self.compute_reward(),
            self.is_done(),
            self.debug_info(),
        )

    def render(self, mode="human", close=False):
        # Render the environment to the screen
        print("Correct:", self.correct)
        print("Board:", self.current_observation())
        print()

    def debug_info(self):
        return dict(
            correct=self.correct,
            board=self.board
        )


class GridTestLinearRewardEnv(GridTestEnv):

    def compute_reward(self):
        if self.is_done():
            return self.step_reward_if_win
        return -self.current_step


class GridTestOneMoveEnv(GridTestEnv):
    
    def start_new_game(self):
        super().start_new_game()
        self.has_moved = False

    def is_done(self):
        return self.has_moved

    def step(self, action):
        self.has_moved = True
        return super().step(action)

    
class GridTestStackedEnv(GridTestEnv):

    def __init__(
        self, *args, num_frames: int = 4, **kwargs
    ):
        self.num_frames = num_frames
        super().__init__(*args, **kwargs)
        observation_shape = (self.num_frames, self.length, self.length, 1)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=observation_shape, dtype=np.uint8
        )

    def start_new_game(self):
        super().start_new_game()
        self.frames = collections.deque([np.full((self.length, self.length, 1), 128, dtype=np.uint8) for _ in range(self.num_frames)], maxlen=self.num_frames)

    def current_observation(self):
        return np.stack(self.frames)

    def step(self, action):
        self.frames.append(super().current_observation())
        return super().step(action)


class WhackAMoleEnv(gym.Env):
    """Codenames environment for gym."""

    environment_name = "Whack-a-mole v0.0.1"
    metadata = {"render.modes": ["human"]}

    def __init__(
        self, length: int = 5, seed: tp.Optional[int] = None
    ):
        super().__init__()
        self.length = length
        self.size = self.length*self.length
        self.action_space = spaces.MultiDiscrete(self.size)
        observation_shape = (self.length, self.length, 1)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=observation_shape, dtype=np.uint8
        )
        self.step_reward_if_win = 0
        self.step_reward_if_not_end = -1
        self.reward_range = (-1, 0)
        self.current_step = 0
        self.max_steps = max_steps if max_steps is not None else 99
        self.id = "WhackAMole"
        self.seed(seed)
        self.start_new_game()

    def seed(self, seed: tp.Optional[int] = None) -> tp.List[int]:
        """Seed the environment.

        Code adapted from a future version of gym.
        """
        if seed is not None and not (isinstance(seed, int) and 0 <= seed):
            raise gym.error.Error(
                f"Seed must be a non-negative integer or omitted, not {seed}"
            )

        seed_seq = np.random.SeedSequence(seed)
        seed_int = seed_seq.entropy
        self.np_random = np.random.Generator(np.random.PCG64(seed_seq))
        return [seed_int]

    def start_new_game(self):
        self.current_step = 0
        self.correct = self.np_random.integers(self.size)
        self.board = np.zeros(self.size, dtype=np.int32)
        self.board[self.correct] = 1

    def reset(self):
        # Reset the state of the environment to an initial state
        self.start_new_game()
        return self.current_observation()

    def current_observation(self):
        return np.clip(self.board.reshape(-1, self.length, 1)*128+128, 0, 255).astype(np.uint8)

    def is_done(self):
        return self.current_step >= self.max_steps or bool(self.board[self.correct] == 0)

    def compute_reward(self):
        if self.is_done():
            return self.step_reward_if_win
        return self.step_reward_if_not_end

    def step(self, action):
        self.current_step += 1
        if not self.is_done():
            self.board[action] = max(self.board[action], 0) - 1
        return (
            self.current_observation(),
            self.compute_reward(),
            self.is_done(),
            self.debug_info(),
        )

    def render(self, mode="human", close=False):
        # Render the environment to the screen
        print("Correct:", self.correct)
        print("Board:", self.current_observation())
        print()

    def debug_info(self):
        return dict(
            correct=self.correct,
            board=self.board
        )
