#!/bin/bash

PROJECT_PREFIX="$(basename "$(pwd)")"
IMAGE_VERSION="${IMAGE_VERSION:-1.0}"

SERVICE="${PROJECT_PREFIX}-jupyter"
IMAGE="${PROJECT_PREFIX}-jupyter:${IMAGE_VERSION}"

echo "Redémarrage du service Jupyter : ${SERVICE}"

docker stop "${SERVICE}" 2>/dev/null
docker rm "${SERVICE}" 2>/dev/null

export DOCKER_BUILDKIT=1

docker compose build "${SERVICE}"
docker compose up -d "${SERVICE}"

echo "Service Jupyter redémarré avec l’image : ${IMAGE}"
