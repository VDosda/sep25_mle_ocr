#!/bin/bash
docker stop $(docker ps -aq)
docker rm $(docker ps -aq)
docker container prune -f
docker image prune -a -f
docker volume prune -f
docker network prune -f
docker builder prune -a -f
sudo systemctl restart docker

