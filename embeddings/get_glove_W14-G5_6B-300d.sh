wget -N --no-check-certificate https://nlp.stanford.edu/data/glove.6B.300d.zip
unzip glove.6B.300d.zip glove.6B.300d.txt
python convert.py glove.6B.300d
