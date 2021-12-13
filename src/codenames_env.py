import typing as tp
import gym
from gym import spaces
import numpy as np
import codenames as cn


NUM_WORDS = 25
NUM_HINT_STRATEGIES = 2
NUM_EMBEDDING_TYPES = 1
NUM_LABELS = 4
CANDIDATE_LIMIT = 3
# observation_shape = (NUM_WORDS, NUM_WORDS, NUM_EMBEDDING_TYPES)
observation_shape = (NUM_WORDS, NUM_WORDS)

DESIRED_GOAL = (np.int8(0), np.ones(2, dtype=np.int8))


def resize_vector(vector, limit):
    """Resize a 1D vector to a given length, padding with "0" when necessary."""
    vector = vector[:limit]
    return np.pad(vector, (0, limit - vector.shape[0]))


def observation_space():
    return spaces.Tuple(
        (
            spaces.Box(
                low=-1, high=1, shape=observation_shape, dtype=np.float32
            ),
            spaces.MultiDiscrete([NUM_LABELS] * NUM_WORDS),
            spaces.MultiBinary(NUM_WORDS),
        )
    )


def goal_space():
    return spaces.Tuple(
        (spaces.Box(low=0, high=9, shape=(), dtype=np.int8), spaces.MultiBinary(2))
    )


class CodenamesEnv(gym.GoalEnv):
    """Codenames environment for gym."""

    environment_name = "Codenames v0.0.1"
    metadata = {"render.modes": ["human"]}

    def __init__(
        self, glove: cn.Glove, wordlist: cn.WordList, seed: tp.Optional[int] = None
    ):
        super().__init__()
        self.action_space = spaces.Tuple(
            (
                spaces.MultiBinary(NUM_WORDS),
                spaces.Discrete(CANDIDATE_LIMIT * NUM_HINT_STRATEGIES),  # noqa
            )
        )
        self.observation_space = spaces.Dict(
            {
                "observation": observation_space(),
                "desired_goal": goal_space(),
                "achieved_goal": goal_space(),
            }
        )
        self.glove = glove
        self.wordlist = wordlist
        self.step_reward_if_lose = -25
        self.step_reward_if_win = 0
        self.step_reward_if_not_end = -1
        self.reward_range = (-float("inf"), self.step_reward_if_not_end)
        self.reward_threshold = -4
        self.trials = 100
        self.max_episode_steps = -self.step_reward_if_lose
        self.id = "Codenames"
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

    def current_observation(self):
        chosen = self.board.chosen.astype(np.int8)
        return self.self_similarity, self.team_indices, chosen

    def achieved_goal(self):
        bad_word_indicator = np.array(
            [
                self.board.remaining_words_for_team(self.opponent) != 0,
                self.board.remaining_words_for_team("ASSASSIN") != 0,
            ],
            dtype=np.int8,
        )
        return (
            np.int8(self.board.remaining_words_for_team(self.team)),
            bad_word_indicator,
        )

    def desired_goal(self):
        return DESIRED_GOAL

    def current_goal_observation(self):
        return {
            "observation": self.current_observation(),
            "desired_goal": self.desired_goal(),
            "achieved_goal": self.achieved_goal(),
        }

    def generate_candidates(self, targets: tp.Sequence[str], limit: int):
        """TODO: For the future, when we can pick among several candidates"""
        mean_candidates = self.guesser.give_hint_candidates(targets, strategy="mean")[0]
        minimax_candidates = self.guesser.give_hint_candidates(
            targets, strategy="minimax"
        )[0]
        all_candidates = np.concatenate(
            [
                resize_vector(mean_candidates, limit),
                resize_vector(minimax_candidates, limit),
            ]
        )
        return all_candidates

    def _is_done(self):
        return (
            self.board.remaining_words_for_team(self.team) == 0
            or self.board.remaining_words_for_team(self.opponent) == 0
            or self.board.remaining_words_for_team("ASSASSIN") == 0
        )

    def is_done(self, achieved_goal):
        return (achieved_goal[0] == 0 or achieved_goal[1].sum() < 2).item()

    def _compute_reward(self):
        if self.board.remaining_words_for_team(self.team) == 0:
            return self.step_reward_if_win
        if (
            self.board.remaining_words_for_team(self.opponent) == 0
            or self.board.remaining_words_for_team("ASSASSIN") == 0
        ):
            return self.step_reward_if_lose
        return self.step_reward_if_not_end

    def compute_reward(self, achieved_goal, desired_goal, info: dict):
        if achieved_goal[0] == 0:
            return self.step_reward_if_win
        if achieved_goal[1].sum() < 2:
            return self.step_reward_if_lose
        return self.step_reward_if_not_end

    def step(self, action):
        words_to_choose, candidate_index = action
        targets = self.board.words[words_to_choose.astype(bool)]
        candidate = self.generate_candidates(targets, CANDIDATE_LIMIT)[candidate_index]
        hint = cn.Hint(candidate, len(targets), self.team)
        guesses, _ = self.guesser.guess(hint)
        self.hints.append(hint)
        self.guessed_words.append([])
        for guess in guesses:
            label = self.board.choose_word(guess)
            hint.num_guessed += 1
            self.guessed_words[-1].append((guess, label))
            if label != self.team:
                break
            hint.num_guessed_correctly += 1
        achieved_goal = self.achieved_goal()
        return (
            self.current_goal_observation(),
            self.compute_reward(achieved_goal, self.desired_goal(), dict()),
            self.is_done(achieved_goal),
            self.debug_info(),
        )

    def start_new_game(self):
        self.board = cn.Board(self.wordlist, rng=self.np_random)
        self.view = cn.CliView(self.board)
        self.guesser = cn.GloveGuesser(self.glove, self.board)
        self.self_similarity = cn.batched_cosine_similarity(
            self.guesser.board_vectors, self.guesser.board_vectors
        ).clip(-1, 1)
        self.team = self.np_random.choice(["RED", "BLUE"])
        self.opponent = self.board.opponent_of(self.team)
        if self.team == "RED":
            self.board.end_turn()
        self.team_labels = self.board.orient_labels_for_team(self.team)
        self.team_indices = cn.find_x_in_y(cn.bot_labels, self.team_labels)
        self.guessed_words = []
        self.hints = []

    def reset(self):
        # Reset the state of the environment to an initial state
        self.start_new_game()
        return self.current_goal_observation()

    def render(self, mode="human", close=False):
        # Render the environment to the screen
        self.view.spymaster_view()
        bag_state = self.board.bag_state()
        bag_counts = {k: len(v) for k, v in bag_state.items()}
        displayable_guessed_words = [
            [f"{word}_{label[:2]}" for word, label in rnd] for rnd in self.guessed_words
        ]
        print(f"The bot is on the {self.team} team.")
        print("Remaining words:", bag_state)
        print("Remaining word count:", bag_counts)
        print("Hint history:", self.hints)
        print("Guessed words", displayable_guessed_words)
        print()

    def debug_info(self):
        return dict(
            board_state=self.view.spymaster_words_to_display(),
            hints=self.hints,
            guessed_words=self.guessed_words,
            team=self.team,
        )


class CodenamesEnvNoHER(CodenamesEnv):
    """The environment with no hindsight experience replay support.

    Some agents don't support hindsight experience replay.
    """

    def __init__(
        self, glove: cn.Glove, wordlist: cn.WordList, seed: tp.Optional[int] = None
    ):
        super().__init__(glove, wordlist, seed)
        self.observation_space = self.observation_space.spaces["observation"]

    def current_goal_observation(self):
        return super().current_goal_observation()["observation"]


def hacked_action_space():
    num_hint_candidates = CANDIDATE_LIMIT * NUM_HINT_STRATEGIES
    assert num_hint_candidates % 2 == 0
    height = NUM_WORDS + num_hint_candidates // 2
    width = 2
    # int64 for compatibility with
    # https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/blob/master/agents/Base_Agent.py
    return spaces.Box(low=0, high=1, shape=(height, width), dtype=np.float32)


def decode_action(hacked_action):
    words_to_choose = hacked_action[:NUM_WORDS].argmax(-1).astype(np.int8)
    candidate_index = hacked_action[NUM_WORDS:].flatten().argmax()
    return words_to_choose, candidate_index


def hacked_observation_space():
    height = NUM_WORDS * NUM_EMBEDDING_TYPES + NUM_LABELS + 2
    width = NUM_WORDS
    return spaces.Box(low=-1, high=1, shape=(height, width), dtype=np.float32)


def one_hot_encode(indices, dim):
    num_entries = indices.shape[0]
    encoding = np.zeros((num_entries, dim), dtype=np.float32)
    encoding[np.arange(num_entries), indices] = 1
    return encoding


def encode_observation(observation):
    self_similarity = observation[0].reshape(-1, NUM_WORDS)
    labels = one_hot_encode(observation[1], NUM_LABELS).T
    chosen_mask = one_hot_encode(observation[2], 2).T
    return np.vstack([self_similarity, labels, chosen_mask]).astype(np.float32)


def hacked_goal_space():
    return spaces.Box(low=0, high=9, shape=(1, NUM_WORDS), dtype=np.int8)


def encode_goal(goal):
    return resize_vector(
        np.concatenate([[goal[0]], goal[1]]), NUM_WORDS
    )[None, :]


ENCODED_DESIRED_GOAL = encode_goal(DESIRED_GOAL)


class CodenamesEnvHack(CodenamesEnv):
    """Codenames environment for gym.

    Hack the action and observation spaces to make more agents work with it.
    """

    environment_name = "Codenames v0.0.1"
    metadata = {"render.modes": ["human"]}

    def __init__(
        self, glove: cn.Glove, wordlist: cn.WordList, seed: tp.Optional[int] = None
    ):
        super().__init__(glove, wordlist, seed)
        self.action_space = hacked_action_space()
        self.observation_space = spaces.Dict(
            {
                "observation": hacked_observation_space(),
                "desired_goal": hacked_goal_space(),
                "achieved_goal": hacked_goal_space(),
            }
        )

    def current_observation(self):
        return encode_observation(super().current_observation())

    def achieved_goal(self):
        return encode_goal(super().achieved_goal())

    def desired_goal(self):
        return ENCODED_DESIRED_GOAL

    def is_done(self, achieved_goal):
        return achieved_goal[0] == 0 or achieved_goal[1:].sum() < 2

    def compute_reward(self, achieved_goal, desired_goal, info: dict):
        if achieved_goal[0] == 0:
            return self.step_reward_if_win
        if achieved_goal[1:].sum() < 2:
            return self.step_reward_if_lose
        return self.step_reward_if_not_end

    def step(self, action):
        return super().step(decode_action(action))


class CodenamesEnvHackNoHER(CodenamesEnvHack):
    """The environment with no hindsight experience replay support.

    Some agents don't support hindsight experience replay.
    """

    def __init__(
        self, glove: cn.Glove, wordlist: cn.WordList, seed: tp.Optional[int] = None
    ):
        super().__init__(glove, wordlist, seed)
        self.observation_space = self.observation_space.spaces["observation"]

    def current_goal_observation(self):
        return super().current_goal_observation()["observation"]
