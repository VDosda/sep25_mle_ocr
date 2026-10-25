#!/bin/bash

PROJECT_PREFIX="$(basename "$(pwd)")"
IMAGE_VERSION="${IMAGE_VERSION:-1.0}"

echo "Réinitialisation Docker pour le préfixe : ${PROJECT_PREFIX}"

echo "Arrêt des conteneurs..."
docker ps -aq --filter "name=${PROJECT_PREFIX}" | xargs -r docker stop

echo "Suppression des conteneurs..."
docker ps -aq --filter "name=${PROJECT_PREFIX}" | xargs -r docker rm

echo "Suppression des volumes persistants..."
docker volume ls -q --filter "name=${PROJECT_PREFIX}" | xargs -r docker volume rm

echo "Suppression des réseaux..."
docker network ls -q --filter "name=${PROJECT_PREFIX}" | xargs -r docker network rm

echo "Réinitialisation terminée pour le préfixe : ${PROJECT_PREFIX}"
