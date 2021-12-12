import pathlib

BASE_PATH = pathlib.Path(__file__).parent.parent
WORDLIST_ENG_PATH = BASE_PATH.joinpath("wordlist-eng.txt")
GOOGLE_10K_ENG_PATH = BASE_PATH.joinpath("google-10000-english-no-swears.txt")
DIRECTIONAL_PATH = BASE_PATH.joinpath("directional.txt")
CUSTOM_WHITELIST = BASE_PATH.joinpath("custom_whitelist.txt")
