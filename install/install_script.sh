#!/bin/bash

# Обновление списка пакетов и установка необходимых зависимостей
apt update
apt install -y apt-transport-https ca-certificates curl software-properties-common

# Установка Docker
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
apt update
apt install -y docker-ce

# Добавление пользователя в группу docker
usermod -aG docker $USER

# Установка Docker Compose
DOCKER_COMPOSE_VERSION="2.32.0" # Укажите нужную версию Docker Compose
curl -L "https://github.com/docker/compose/releases/download/$DOCKER_COMPOSE_VERSION/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Проверка установленных версий
echo "Список установленных версий:"
docker --version
docker-compose --version