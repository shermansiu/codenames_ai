import numpy as np
import pathlib
import attr
import typing as tp
import nptyping as npt
import abc
from scipy.special import softmax


if tp.TYPE_CHECKING:
    import os  # noqa


PathLike = tp.Union[str, "os.PathLike[str]"]


default_rng = np.random.default_rng()
labels = ["BLUE"] * 9 + ["RED"] * 8 + ["BYSTANDER"] * 7 + ["ASSASSIN"]
bot_labels = np.array(["OURS", "THEIRS", "BYSTANDER", "ASSASSSIN"])
valid_teams = {"BLUE", "RED"}
unique_labels = np.unique(labels).tolist()


def regularize(list_of_tokens: tp.Iterable[str]) -> tp.List[str]:
    """Regularize the tokens."""
    return [token.strip().upper() for token in list_of_tokens]


def find_x_in_y(x: npt.NDArray, y: npt.NDArray) -> npt.NDArray:
    # https://stackoverflow.com/a/8251757
    index = np.argsort(x)
    sorted_x = x[index]
    sorted_index = np.searchsorted(sorted_x, y)
    y_index = np.take(index, sorted_index, mode="clip")
    return y_index[x[y_index] == y]


def standardize_length(ragged_matrix: tp.Sequence[tp.Sequence]) -> tp.List[tp.Sequence]:
    """Standardize the length of a ragged matrix.

    Example:
        [[3], [4, 5]] -> [[3, 3], [4, 5]]
        [[3], [4, 5], [6, 7, 8]] -> [[3, 3, 3, 3, 3, 3], [4, 5, 4, 5, 4, 5], [6, 7, 8, 6, 7, 8]]
    """
    lengths = [len(i) for i in ragged_matrix]
    lcm = np.lcm.reduce(lengths)
    duplication_count = lcm // lengths
    return [row * n_rep for row, n_rep in zip(ragged_matrix, duplication_count)]


class WordList:
    """The list of words."""

    def __init__(
        self,
        wordlist_path: PathLike,
        illegals_paths: tp.Optional[tp.List[PathLike]] = None,
        allowed_paths: tp.Optional[tp.List[PathLike]] = None,
    ):
        path = pathlib.Path(wordlist_path)
        with path.open() as f:
            self.words = regularize(f.read().splitlines())
        self.illegals = self.load_texts(illegals_paths) if illegals_paths else set()
        self.allowed = self.load_texts(allowed_paths) if allowed_paths else set()
        # If it is illegal for the board, it will be detected later on
        self.allowed.update(self.words)

    def load_texts(self, paths: tp.List[PathLike]) -> tp.Set[str]:
        texts = set()
        for pth in paths:
            path = pathlib.Path(pth)
            with path.open() as f:
                texts.update(f.read().splitlines())
        return set(regularize(texts))


def is_superstring_or_substring(word: str, target: str) -> bool:
    return target in word or word in target


class Board:
    def __init__(self, wordlist: WordList, rng: tp.Optional[object] = None) -> None:
        self.wordlist = wordlist
        if rng is None:
            rng = default_rng
        self.rng = np.random.default_rng(rng)
        self.words = self.rng.choice(wordlist.words, 25, replace=False)
        self.word2index = {word: i for i, word in enumerate(self.words)}
        self.labels: tp.List[str] = self.rng.permutation(labels)
        self.reset_game()

    def is_related_word(self, word: str) -> bool:
        word = word.upper()
        return any(is_superstring_or_substring(word, target) for target in self.words)

    def is_illegal(self, word: str) -> bool:
        word = word.upper()
        return (
            self.is_related_word(word)
            or word in self.wordlist.illegals
            or word not in self.wordlist.allowed
        )

    def batch_is_illegal(self, words: npt.NDArray[str]) -> npt.NDArray[bool]:
        return np.array([self.is_illegal(w) for w in words])

    def reset_game(self) -> None:
        self.chosen = np.array([False] * 25)
        self.which_team_guessing = "BLUE"
        # self.hint_history = []
        # self.state_history = None

    def opponent_of(self, team: str):
        assert team in valid_teams
        return list(valid_teams.difference([team]))[0]

    def end_turn(self):
        self.which_team_guessing = self.opponent_of(self.which_team_guessing)

    def choose_word(self, word: str) -> str:
        if word.upper() not in self.words:
            raise KeyError(f"Word '{word}' is not on the board.")
        index = self.word2index[word]
        if self.chosen[index]:
            raise ValueError(f"Word '{word}' has already been chosen!")
        self.chosen[index] = True
        return self.labels[index]

    def words_that_are_label(self, label):
        return self.words[self.labels == label]

    @property
    def blue_words(self):
        return self.words_that_are_label("BLUE")

    @property
    def red_words(self):
        return self.words_that_are_label("RED")

    @property
    def bystander_words(self):
        return self.words_that_are_label("BYSTANDER")

    @property
    def assassin_words(self):
        """There is only one assassin in a regular game, but for the sake of generality, here we go!"""
        return self.words_that_are_label("ASSASSIN")

    def indices_for_label(self, label):
        return np.where(self.labels == label)[0]

    @property
    def blue_indices(self):
        return self.indices_for_label("BLUE")

    @property
    def red_indices(self):
        return self.indices_for_label("RED")

    @property
    def bystander_indices(self):
        return self.indices_for_label("BYSTANDER")

    @property
    def assassin_indices(self):
        """There is only one assassin in a regular game, but for the sake of generality, here we go!"""
        return self.indices_for_label("ASSASSIN")

    def jump_to_random_state(self) -> None:
        """Jump to a valid random state before the end of the game.

        There must be 1 assassin, 1-9 blue words, 1-8 red words, and 0-7 bystanders.
        Thus, 0-8 blue, 0-7 red and 1-6 bystander words are chosen.
        """
        self.reset_game()
        num_blue = self.rng.integers(0, 9)
        num_red = self.rng.integers(0, 8)
        num_bystanders = self.rng.integers(0, 7)
        chosen_blue = self.rng.choice(self.blue_indices, num_blue, replace=False)
        chosen_red = self.rng.choice(self.red_indices, num_red, replace=False)
        chosen_bystanders = self.rng.choice(
            self.bystander_indices, num_bystanders, replace=False
        )
        chosen_indices = np.concatenate([chosen_blue, chosen_red, chosen_bystanders])
        self.chosen[chosen_indices] = True
        self.which_team_guessing = self.rng.choice(["BLUE", "RED"])

    def bag_state(self) -> tp.Dict[str, tp.Set[str]]:
        return {
            label: set(self.words[(self.labels == label) & ~self.chosen])
            for label in unique_labels
        }

    def remaining_words(self) -> npt.NDArray[str]:
        return self.words[~self.chosen]

    def remaining_words_for_team(self, team: str) -> int:
        return (~self.chosen & (self.labels == team)).sum()

    def orient_label(self, my_team, opponent_team, label):
        if label == my_team:
            return "OURS"
        elif label == opponent_team:
            return "THEIRS"
        return label

    def orient_labels_for_team(self, my_team: str) -> npt.NDArray[str]:
        opponent_team = self.opponent_of(my_team)
        return np.array(
            [self.orient_label(my_team, opponent_team, label) for label in self.labels]
        )


class CliView:
    def __init__(self, board: Board):
        self.board = board

    def spymaster_words_to_display(self):
        words = []
        # Arguably more readable than the equivalent list comprehension
        for w, l, c in zip(self.board.words, self.board.labels, self.board.chosen):
            w += f"_{l[:2]}"
            if c:
                w = w.lower()
            words.append(w)
        return words

    def operative_words_to_display(self):
        words = []
        # Arguably more readable than the equivalent list comprehension
        for w, l, c in zip(self.board.words, self.board.labels, self.board.chosen):
            if c:
                w += f"_{l[:2]}"
                w = w.lower()
            words.append(w)
        return words

    def generic_view(self, words_to_display):
        words = words_to_display()
        print(np.array(words).reshape(5, 5))
        print(f"It is {self.board.which_team_guessing}'s turn.")

    def spymaster_view(self):
        self.generic_view(self.spymaster_words_to_display)

    def operative_view(self):
        self.generic_view(self.operative_words_to_display)


@attr.s(auto_attribs=True)
class Hint:
    word: str
    count: tp.Optional[int]
    team: str
    num_guessed: int = attr.ib(default=0)
    num_guessed_correctly: int = attr.ib(default=0)


class TextVectorEngine(metaclass=abc.ABCMeta):
    # TODO: add self.vectors, self.tokens, self.token2id here too

    @abc.abstractmethod
    def is_valid_token(self, token):
        pass

    @abc.abstractmethod
    def tokenize(self, phrase):
        pass


class Glove(TextVectorEngine):
    def __init__(self, glove_vector_path, glove_tokens_path):
        gv_path = pathlib.Path(glove_vector_path)
        gt_path = pathlib.Path(glove_tokens_path)
        assert gv_path.exists()
        assert gt_path.exists()
        with gv_path.open("rb") as f:
            self.vectors = np.load(gv_path)
        with gt_path.open() as f:
            self.tokens = f.read().splitlines()
            self.tokens = np.array(regularize(self.tokens))
        self.token2id = {t: i for i, t in enumerate(self.tokens)}

    def is_valid_token(self, token: str) -> bool:
        return token.strip().upper() in self.token2id

    def is_tokenizable(self, phrase: str) -> bool:
        return all(token is not None for token in self.tokenize(phrase))

    def tokenize(self, phrase):
        """Simple one-word tokenization. Ignores punctuation."""
        if isinstance(phrase, str):
            phrase = phrase.strip().upper().split()
            return [
                self.token2id[x] if self.is_valid_token(x) else None for x in phrase
            ]
        else:
            phrase = regularize(phrase)
            return [self.tokenize(token) for token in phrase]

    def vectorize(self, phrase: tp.Union[str, tp.Sequence[str]]) -> npt.NDArray:
        if isinstance(phrase, str):
            return self.vectors[self.tokenize(phrase)]
        tokens = np.array(standardize_length(self.tokenize(phrase)))
        return self.vectors[tokens].mean(axis=1)


def batched_norm(vec: np.ndarray) -> np.ndarray:
    """Normalize a batch of vectors

    Args:
        vec: (batch, dim)
    """
    return vec / np.linalg.norm(vec, axis=1)[:, None]


def batched_cosine_similarity(a: npt.NDArray, b: npt.NDArray) -> npt.NDArray:
    """Take the batched cosine similarity."""
    a_norm = batched_norm(a)  # (batch1, dim)
    b_norm = batched_norm(b)  # (batch2, dim)
    return a_norm @ b_norm.T  # (batch1, batch2)


GuessStrategyLookup = tp.Dict[
    str,
    tp.Callable[[npt.NDArray, npt.NDArray, int], tp.Tuple[npt.NDArray, npt.NDArray]],
]


class GloveGuesser:
    def __init__(self, glove: Glove, board: Board, limit: int = 10):
        self.glove = glove
        self.board = board
        self.limit = limit
        self.word_suggestion_strategy_lookup = {
            "mean": self.generate_word_suggestions_mean,
            "minimax": self.generate_word_suggestions_minimax,
        }
        self.guess_strategy_lookup: GuessStrategyLookup = {
            "greedy": self.guess_greedy,
            "softmax": self.guess_softmax,
        }
        self.board_vectors = self.glove.vectorize(self.board.words)

    def indices_illegal_words(self, chosen_words: npt.NDArray):
        return self.board.batch_is_illegal(chosen_words)

    def generate_word_suggestions_mean(
        self, words: tp.List[str], limit: int = 10
    ) -> tp.Tuple[tp.Sequence[str], tp.Sequence[float]]:
        for word in words:
            if not self.glove.is_tokenizable(word):
                raise ValueError(f"Hint {word} is not a valid hint word!")
        word_vector = self.glove.vectorize(" ".join(words)).mean(0)[None, :]
        similarity_scores = batched_cosine_similarity(word_vector, self.glove.vectors)[
            0
        ]
        indices = np.argpartition(-similarity_scores, limit)
        chosen_words = self.glove.tokens[indices][:limit]
        similarity_scores = similarity_scores[indices][:limit]
        return chosen_words, similarity_scores

    def generate_word_suggestions_minimax(
        self, words: tp.List[str], limit: int = 10
    ) -> tp.Tuple[tp.Sequence[str], tp.Sequence[float]]:
        for word in words:
            if not self.glove.is_tokenizable(word):
                raise ValueError(f"Hint {word} is not a valid hint word!")
        word_vector = self.glove.vectorize(" ".join(words))
        similarity_scores = batched_cosine_similarity(
            word_vector, self.glove.vectors
        ).min(axis=0)
        indices = np.argpartition(-similarity_scores, limit)
        chosen_words = self.glove.tokens[indices][:limit]
        similarity_scores = similarity_scores[indices][:limit]
        return chosen_words, similarity_scores

    def filter_words(
        self,
        chosen_words: npt.NDArray[str],
        similarity_scores: npt.NDArray[float],
        similarity_threshold=0.0,
    ):
        words_to_filter = self.indices_illegal_words(chosen_words) | (
            similarity_scores < similarity_threshold
        )
        return chosen_words[~words_to_filter], similarity_scores[~words_to_filter]

    def re_rank(
        self,
        chosen_words: npt.NDArray[str],
        similarity_scores: npt.NDArray[float],
        limit: int,
    ):
        indices = np.argsort(-similarity_scores)
        chosen_words = chosen_words[indices][:limit]
        similarity_scores = similarity_scores[indices][:limit]
        return chosen_words, similarity_scores

    def give_hint_candidates(
        self, targets: tp.List[str], similarity_threshold=0.0, strategy: str = "minimax"
    ):
        generate_word_suggestions = self.word_suggestion_strategy_lookup[strategy]
        chosen_words, similarity_scores = generate_word_suggestions(
            targets, self.limit * 2
        )

        chosen_words, similarity_scores = self.filter_words(
            chosen_words, similarity_scores
        )

        return self.re_rank(chosen_words, similarity_scores, self.limit)

    def give_hint(
        self, targets: tp.List[str], similarity_threshold=0.0, strategy: str = "minimax"
    ):
        """Greedily choose the best hint."""
        chosen_words, _ = self.give_hint_candidates(
            targets, similarity_threshold, strategy
        )
        return chosen_words[0]

    def choose_hint_parameters(self, hint: Hint) -> tp.Tuple[str, int]:
        """TODO: Add strategy mixins"""
        num_words_remaining = self.board.remaining_words_for_team(
            self.board.which_team_guessing
        )
        if hint.count is None:
            limit = num_words_remaining
        else:
            limit = min(hint.count - hint.num_guessed_correctly, num_words_remaining)
        return hint.word, limit

    def remaining_word_vectors(self) -> npt.NDArray:
        return self.board_vectors[~self.board.chosen]

    def guess_greedy(
        self, remaining_words: npt.NDArray, similarity_scores: npt.NDArray, limit: int
    ) -> tp.Tuple[npt.NDArray, npt.NDArray]:
        indices = np.argsort(-similarity_scores)
        chosen_words = remaining_words[indices][:limit]
        similarity_scores = similarity_scores[indices][:limit]
        return chosen_words, similarity_scores

    def guess_softmax(
        self,
        remaining_words: npt.NDArray,
        similarity_scores: npt.NDArray,
        limit: int,
        temperature: float = 0.05,
    ) -> tp.Tuple[npt.NDArray, npt.NDArray]:
        chosen_words = np.random.choice(
            remaining_words,
            limit,
            p=softmax(similarity_scores / temperature),
            replace=False,
        )
        chosen_words_indices = find_x_in_y(remaining_words, chosen_words)
        return chosen_words, similarity_scores[chosen_words_indices]

    def guess(self, hint: Hint, strategy: str = "softmax") -> tp.Sequence[str]:
        word, limit = self.choose_hint_parameters(hint)
        if not self.glove.is_valid_token(word):
            raise ValueError(f"Hint {word} is not a valid hint word!")
        word_vector = self.glove.vectorize(word)
        remaining_words = self.board.remaining_words()
        remaining_word_vectors = self.remaining_word_vectors()
        similarity_scores = batched_cosine_similarity(
            word_vector, remaining_word_vectors
        )[0]
        guess_with_strategy = self.guess_strategy_lookup[strategy]
        return guess_with_strategy(remaining_words, similarity_scores, limit)
