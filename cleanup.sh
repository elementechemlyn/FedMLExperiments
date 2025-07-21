#!/bin/bash
echo "Stopping and removing containers..."
for i in `docker ps -a -q --filter "name=fl_server" --filter "name=fl_client_"`; do
    docker stop $i
    docker rm $i
done
echo "Removing Docker network 'federated_ml_net'..."
docker network rm federated_ml_net
