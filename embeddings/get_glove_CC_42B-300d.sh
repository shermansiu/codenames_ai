wget -N --no-check-certificate https://nlp.stanford.edu/data/glove.42B.300d.zip
unzip glove.42B.300d.zip glove.42B.300d.txt
python convert_glove.py glove.42B.300d
