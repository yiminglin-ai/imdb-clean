#!/bin/bash

echo Creating IMDB-Clean dataset

mkdir data
cd data

pids=""
RESULT=0

for i in $(seq 0 9); do
        wget https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_$i.tar &
        pids="$pids $!"
done

echo "This can take about 1 hr"

for pid in $pids; do
        wait $pid || let "RESULT=1"
done

pids=""
RESULT=0

for i in $(seq 0 9); do
        tar xvf imdb_$i.tar && rm -rf imdb_$i.tar &
        pids="$pids $!"

done

for pid in $pids; do
        wait $pid || let "RESULT=1"
done

cd ..

python create_imdb_clean_1024.py test
python create_imdb_clean_1024.py valid
python create_imdb_clean_1024.py train
