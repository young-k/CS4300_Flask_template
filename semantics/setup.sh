wget -P ../data/ http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
rm *.zip
curl https://s3.amazonaws.com/senteval/infersent/infersent.allnli.pickle
