mkdir -p data

echo 'Downloading celeba dataset ... '
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1jzB6-Ohgos8yeK8mrBws5OlUGVhrDF4L' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1jzB6-Ohgos8yeK8mrBws5OlUGVhrDF4L" -O celeba.zip && rm -rf /tmp/cookies.txt
unzip celeba.zip -d data/celeba

echo 'Downloading flickr dataset ... '
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1tg-Ur7d4vk1T8Bn0pPpUSQPxlPGBlGfv' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1tg-Ur7d4vk1T8Bn0pPpUSQPxlPGBlGfv" -O flickr.zip && rm -rf /tmp/cookies.txt
unzip flickr.zip -d data/flickr

echo 'Downloading lfw dataset ... '
wget http://vis-www.cs.umass.edu/lfw/lfw.tgz 
tar zxvf lfw.tgz 
mkdir data 
mv lfw data 

echo 'Downloading utk dataset ... '
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0BxYys69jI14kU0I1YUQyY1ZDRUE' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0BxYys69jI14kU0I1YUQyY1ZDRUE" -O utk.zip && rm -rf /tmp/cookies.txt
unzip utk.zip -d data/utk

echo 'Downloading nvidia mask testing dataset ... '
wget https://www.dropbox.com/s/01dfayns9s0kevy/test_mask.zip
unzip test_mask.zip
mv mask test_masks

echo 'Downloading nvidia mask training dataset ... '
wget https://www.dropbox.com/s/qp8cxqttta4zi70/irregular_mask.zip
unzip irregular_mask.zip
mv irregular_mask mask