#!/usr/bin/python
# Adapted from https://github.com/thomasahle/codenames/blob/master/convert.py
import subprocess
import numpy as np
import tqdm.auto
import fire

# Based on the preprocessing in https://github.com/FALCONN-LIB/FALCONN
def main(glove_name: str):
    matrix = []
    words = []
    glove_file = f'{glove_name}.txt'
    dimensions = eval(glove_name.split(".")[2][:-1])
    print("Calculating number of vectors...")
    result = subprocess.run(["wc", "-l", f"{glove_name}.txt"], stdout=subprocess.PIPE, text=True)
    num_lines = eval(result.stdout.split(" ")[0])
    with open(glove_file, 'r') as inf:
        for line in tqdm.auto.tqdm(inf, total=num_lines):
            word, *rest = line.split()
            words.append(word)
            row = list(map(float, rest))
            assert len(row) == dimensions
            matrix.append(np.array(row, dtype=np.float32))

    np.save(glove_name, np.array(matrix))

    with open(f'{glove_name}_vocab.txt', 'w') as ouf:
        ouf.write('\n'.join(words))


if __name__ == "__main__":
    fire.Fire(main)
