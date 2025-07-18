sudo apt-get remove docker docker-engine docker.io containerd runc
sudo apt-get update
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

sudo apt-key fingerprint 0EBFCD88

sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt-get update
apt-cache madison docker-ce

export VERSION_STRING="18.06.3~ce~3-0~ubuntu"
sudo apt-get install docker-ce docker-ce-cli containerd.io

sudo docker -v 

sudo groupadd docker 
sudo gpasswd -a ${USER} docker
newgrp - docker    # after config, need restart 
sudo service docker restart

docker -v 
