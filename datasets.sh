mkdir -p data

echo 'Downloading celeba dataset ... '
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1jzB6-Ohgos8yeK8mrBws5OlUGVhrDF4L' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1jzB6-Ohgos8yeK8mrBws5OlUGVhrDF4L" -O celeba.zip && rm -rf /tmp/cookies.txt
unzip celeba.zip -d data/celeba

echo 'Downloading flickr dataset ... '
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1bgffRt1JPRv5k3vmF-NEATLkiPmq_e8w' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1bgffRt1JPRv5k3vmF-NEATLkiPmq_e8w" -O flickr.zip && rm -rf /tmp/cookies.txt
unzip flickr.zip 
mkdir -p data/flickr 
mv thumbnails128x128 data/flickr 

echo 'Downloading lfw dataset ... '
wget http://vis-www.cs.umass.edu/lfw/lfw.tgz 
tar zxvf lfw.tgz 
mv lfw data 

echo 'Downloading utk dataset ... '
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0BxYys69jI14kYVM3aVhKS1VhRUk' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0BxYys69jI14kYVM3aVhKS1VhRUk" -O utk.tar.gz && rm -rf /tmp/cookies.txt
tar xvzf utk.tar.gz
mkdir -p data/utkface/utkface
mv UTKFace/ data/utkf/utkface

echo 'Downloading nvidia mask testing dataset ... '
wget https://www.dropbox.com/s/01dfayns9s0kevy/test_mask.zip
unzip test_mask.zip
mv mask data/test_masks