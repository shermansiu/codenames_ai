import attr
import pathlib


@attr.s(auto_attribs=True)
class GlovePath:
    vectors: pathlib.Path
    words: pathlib.Path

    @property
    def paths(self):
        return self.vectors, self.words


BASE_PATH = pathlib.Path(__file__).parent.parent

# Word list
WORDLIST_ENG_PATH = BASE_PATH.joinpath("wordlist-eng.txt")

# Word vectors
GLOVE_6B_300D_PATH = GlovePath(
    BASE_PATH.parent.joinpath("codenames/dataset/glove.6B.300d.npy"),
    BASE_PATH.parent.joinpath("codenames/dataset/words"),
)

# Allowed
GOOGLE_10K_ENG_PATH = BASE_PATH.joinpath("google-10000-english-no-swears.txt")
WIKI_100K_PATH = BASE_PATH.joinpath("wiki-100k.txt")
CUSTOM_WHITELIST = BASE_PATH.joinpath("custom_whitelist.txt")

# Illegal
DIRECTIONAL_PATH = BASE_PATH.joinpath("directional.txt")
