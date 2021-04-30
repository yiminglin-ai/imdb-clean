for i in $(seq 0 9); 
do 
        wget https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_$i.tar
done
#wget https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki.tar.gz


for i in $(seq 0 9); 
do 
        tar xvf imdb_$i.tar && rm -rf imdb_$i.tar 
done
#wget https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki.tar.gz

