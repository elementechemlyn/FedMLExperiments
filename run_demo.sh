#!/bin/bash

# Define number of clients (must match NUM_CLIENTS in server/app.py and split_data.py)
NUM_CLIENTS=3
SERVER_PORT=5000
DOCKER_NETWORK_NAME="federated_ml_net"
SERVER_CONTAINER_NAME="fl_server"

echo "--- Setting up Federated ML Demo ---"

# 1. Create data directory and split data
echo "1. Preparing data..."
mkdir -p data
python data/split_data.py $NUM_CLIENTS --output_dir data

# # 2. Build Docker images
echo "2. Building Docker images..."
docker build -t fl_server ./server
docker build -t fl_client ./client

# # 3. Create a Docker network
echo "3. Creating Docker network '$DOCKER_NETWORK_NAME'..."
docker network create $DOCKER_NETWORK_NAME || true # '|| true' to ignore error if network already exists

# # 4. Run the server container
echo "4. Starting server container '$SERVER_CONTAINER_NAME'..."
docker run -d --name $SERVER_CONTAINER_NAME --network $DOCKER_NETWORK_NAME -p $SERVER_PORT:$SERVER_PORT fl_server

# # Give the server a moment to start up
echo "Waiting for server to start (5 seconds)..."
sleep 5

# # Get the server's IP address within the Docker network
# # This is crucial for clients to connect to the server by its container name
SERVER_IP=$SERVER_CONTAINER_NAME # Docker's built-in DNS allows using container names as hostnames

# # 5. Run client containers
echo "5. Starting $NUM_CLIENTS client containers..."
for i in $(seq 0 $((NUM_CLIENTS-1))); do
    CLIENT_CONTAINER_NAME="fl_client_$i"
    CLIENT_DATA_FILE="client_${i}_data.npz"

    echo "  - Starting client $i ($CLIENT_CONTAINER_NAME)..."
    docker run -d --name $CLIENT_CONTAINER_NAME --network $DOCKER_NETWORK_NAME \
        -v "$(pwd)/data/$CLIENT_DATA_FILE:/app/data/$CLIENT_DATA_FILE" \
        fl_client python client.py "$i" "http://$SERVER_IP:$SERVER_PORT"
done

echo "--- Federated ML Demo Setup Complete ---"
echo "You can monitor the server logs with: docker logs -f $SERVER_CONTAINER_NAME"
echo "You can monitor client logs with: docker logs -f fl_client_0 (replace 0 with client ID)"
echo "To stop and clean up: ./cleanup.sh"

# # Optional: Create a cleanup script
cat <<EOF > cleanup.sh
#!/bin/bash
echo "Stopping and removing containers..."
for i in \`docker ps -a -q --filter "name=fl_server" --filter "name=fl_client_"\`; do
    docker stop \$i
    docker rm \$i
done
echo "Removing Docker network '$DOCKER_NETWORK_NAME'..."
docker network rm $DOCKER_NETWORK_NAME
EOF
chmod +x cleanup.sh
