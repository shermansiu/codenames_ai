wget -N --no-check-certificate https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip glove.6B.300d.txt
python convert_glove.py glove.6B.300d
