import codenames
import paths


wordlist = codenames.WordList(
    paths.WORDLIST_ENG_PATH,
    [paths.DIRECTIONAL_PATH],
    [paths.WIKI_100K_PATH, paths.GOOGLE_10K_ENG_PATH, paths.CUSTOM_WHITELIST],
)
board = codenames.Board(wordlist)
view = codenames.CliView(board)
glove = codenames.Glove(*paths.GLOVE_6B_300D_PATH.paths)
guesser = codenames.GloveGuesser(glove, board)
