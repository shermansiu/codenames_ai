import numpy as np
import pathlib
import attr
from typing import List, Sequence, Optional, Set, Tuple
import typing as tp
import nptyping as npt
import abc


rng = np.random.default_rng()
labels = ["BLUE"] * 9 + ["RED"] * 8 + ["BYSTANDER"] * 7 + ["ASSASSIN"]
unique_labels = np.unique(labels).tolist()


def regularize(list_of_tokens: List[str]) -> List[str]:
    """Regularize the tokens."""
    return [token.strip().upper() for token in list_of_tokens]


class WordList:
    """The list of words."""
    def __init__(
        self,
        wordlist_path: str,
        illegals_paths: Optional[List[str]] = None,
        allowed_paths: Optional[List[str]] = None,
    ):
        path = pathlib.Path(wordlist_path)
        with path.open() as f:
            self.words = regularize(f.read().splitlines())
        self.illegals = self.load_texts(illegals_paths) if illegals_paths else set()
        self.allowed = self.load_texts(allowed_paths) if allowed_paths else set()
        # If it is illegal for the board, it will be detected later on
        self.allowed.update(self.words)

    def load_texts(self, paths: List[str]) -> Set[str]:
        texts = set()
        for pth in paths:
            path = pathlib.Path(pth)
            with path.open() as f:
                texts.update(f.read().splitlines())
        return set(regularize(texts))


def is_superstring_or_substring(word: str, target: str) -> bool:
    return target in word or word in target


class Board:
    def __init__(self, wordlist: WordList) -> None:
        self.wordlist = wordlist
        self.words = rng.choice(wordlist.words, 25, replace=False)
        self.word2index = {word: i for i, word in enumerate(self.words)}
        self.labels = rng.permutation(labels)
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
        return words_that_are_label("BLUE")

    @property
    def red_words(self):
        return words_that_are_label("RED")

    @property
    def bystander_words(self):
        return words_that_are_label("BYSTANDER")

    @property
    def assassin_words(self):
        """There is only one assassin in a regular game, but for the sake of generality, here we go!"""
        return words_that_are_label("ASSASSIN")

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
        num_blue = rng.integers(0, 9)
        num_red = rng.integers(0, 8)
        num_bystanders = rng.integers(0, 7)
        chosen_blue = rng.choice(self.blue_indices, num_blue, replace=False)
        chosen_red = rng.choice(self.red_indices, num_red, replace=False)
        chosen_bystanders = rng.choice(
            self.bystander_indices, num_bystanders, replace=False
        )
        chosen_indices = np.concatenate([chosen_blue, chosen_red, chosen_bystanders])
        self.chosen[chosen_indices] = True
        self.which_team_guessing = rng.choice(["BLUE", "RED"])

    def bag_state(self) -> tp.Dict["label", tp.Set["words"]]:
        return {
            label: set(self.words[(self.labels == label) & ~self.chosen])
            for label in unique_labels
        }

    def remaining_words(self) -> npt.NDArray[str]:
        return self.words[~self.chosen]


class CliView:
    def __init__(self, board: Board):
        self.board = board

    def spymaster_words_to_display(self):
        words = []
        # Arguably more readable than the equivalent list comprehension
        for w, l, c in zip(self.board.words, self.board.labels, self.board.chosen):
            w += f"_{l[0]}"
            if c:
                w = w.lower()
            words.append(w)
        return words

    def operative_words_to_display(self):
        words = []
        # Arguably more readable than the equivalent list comprehension
        for w, l, c in zip(self.board.words, self.board.labels, self.board.chosen):
            if c:
                w += f"_{l[0]}"
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


@attr.s(frozen=True, auto_attribs=True)
class Hint:
    word: str
    count: Optional[int]
    remaining: int = attr.ib(default=0)


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

    def is_valid_token(self, token):
        return token.strip().upper() in self.token2id

    def tokenize(self, phrase):
        """Simple one-word tokenization. Ignores punctuation."""
        if isinstance(phrase, str):
            phrase = phrase.strip().upper().split()
        else:
            phrase = regularize(phrase)
        return [self.token2id[x] if self.is_valid_token(x) else None for x in phrase]

    def vectorize(self, phrase):
        return self.vectors[self.tokenize(phrase)]


def batched_norm(vec: np.ndarray) -> np.ndarray:
    """Normalize a batch of vectors

    Args:
        vec: (batch, dim)
    """
    return vec / np.linalg.norm(vec, axis=1)[:, None]


def batched_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Take the batched cosine similarity."""
    a_norm = batched_norm(a)  # (batch1, dim)
    b_norm = batched_norm(b)  # (batch2, dim)
    return a_norm @ b_norm.T  # (batch1, batch2)


class GloveGuesser:
    def __init__(self, glove: Glove, board: Board, limit: int = 10):
        self.glove = glove
        self.board = board
        self.limit = limit
        self.strategy_lookup = {
            "mean": self.generate_word_suggestions_mean,
            "minimax": self.generate_word_suggestions_minimax,
        }

    def indices_illegal_words(self, chosen_words: npt.NDArray):
        return self.board.batch_is_illegal(chosen_words)

    def generate_word_suggestions_mean(
        self, words: List[str], limit: int = 10
    ) -> Tuple[Sequence[str], Sequence[float]]:
        for word in words:
            if not self.glove.is_valid_token(word):
                raise ValueError(f"Hint {word} is not a valid hint word!")
        word_vector = self.glove.vectorize(" ".join(words)).mean(0)[None, :]
        similarity_scores = batched_cosine_similarity(word_vector, glove.vectors)[0]
        indices = np.argpartition(-similarity_scores, limit)
        chosen_words = glove.tokens[indices][:limit]
        similarity_scores = similarity_scores[indices][:limit]
        return chosen_words, similarity_scores

    def generate_word_suggestions_minimax(
        self, words: List[str], limit: int = 10
    ) -> Tuple[Sequence[str], Sequence[float]]:
        for word in words:
            if not self.glove.is_valid_token(word):
                raise ValueError(f"Hint {word} is not a valid hint word!")
        word_vector = self.glove.vectorize(" ".join(words))
        similarity_scores = batched_cosine_similarity(word_vector, glove.vectors).min(
            axis=0
        )
        indices = np.argpartition(-similarity_scores, limit)
        chosen_words = glove.tokens[indices][:limit]
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
        self, targets: List[str], similarity_threshold=0.0, strategy: str = "minimax"
    ):
        generate_word_suggestions = self.strategy_lookup[strategy]
        chosen_words, similarity_scores = generate_word_suggestions(
            targets, self.limit * 2
        )

        chosen_words, similarity_scores = self.filter_words(
            chosen_words, similarity_scores
        )

        return self.re_rank(chosen_words, similarity_scores, self.limit)

    def give_hint(
        self, targets: List[str], similarity_threshold=0.0, strategy: str = "minimax"
    ):
        """Greedily choose the best hint."""
        chosen_words, _ = self.give_hint_candidates(
            targets, similarity_threshold, strategy
        )
        return chosen_words[0]

    def choose_hint_parameters(self, hint: Hint) -> tp.Tuple[str, int]:
        """TODO: Add strategy mixins"""
        return hint.word, hint.count - hint.remaining

    def guess(self, hint: Hint, strategy: str = "greedy") -> Sequence[str]:
        word, limit = self.choose_hint_parameters(hint)
        if not self.glove.is_valid_token(word):
            raise ValueError(f"Hint {word} is not a valid hint word!")
        word_vector = self.glove.vectorize(word)
        remaining_words = self.board.remaining_words()
        board_vectors = self.glove.vectorize(remaining_words)
        similarity_scores = batched_cosine_similarity(word_vector, board_vectors)[0]
        indices = np.argpartition(-similarity_scores, limit)
        chosen_words = remaining_words[indices][:limit]
        similarity_scores = similarity_scores[indices][:limit]
        return chosen_words, similarity_scores
