import os
import sys
import time
import logging

import requests
from tensorflow import keras
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get client ID from command line arguments
if len(sys.argv) < 2:
    logging.error("Usage: python client.py <client_id> <server_url>")
    sys.exit(1)

client_id = sys.argv[1]
server_url = sys.argv[2]
data_file = f'data/client_{client_id}_data.npz'

logging.info(f"Client {client_id} starting. Connecting to server at {server_url}")

# Model definition (must match server's model)
# TODO Get this config from the server?
def create_model():
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Load local data
def load_local_data(file_path):
    if not os.path.exists(file_path):
        logging.error(f"Data file not found: {file_path}. Ensure split_data.py was run.")
        sys.exit(1)
    data = np.load(file_path)
    x_train, y_train = data['x_train'], data['y_train']
    x_test, y_test = data['x_test'], data['y_test']
    logging.info(f"Client {client_id}: Loaded local data. Train samples: {len(x_train)}, Test samples: {len(x_test)}")
    return (x_train, y_train), (x_test, y_test)

# Helper functions for weight serialization/deserialization
def deserialize_weights(weights_list):
    """Converts a list of lists (from JSON) back to a list of numpy arrays."""
    return [np.array(w) for w in weights_list]

def serialize_weights(weights):
    """Converts a list of numpy arrays to a list of lists (for JSON)."""
    return [w.tolist() for w in weights]

def fetch_global_model(server_url):
    """Fetches the global model weights from the server."""
    try:
        response = requests.get(f'{server_url}/get_global_model')
        response.raise_for_status() # Raise an exception for HTTP errors
        data = response.json()
        logging.info(f"Client {client_id}: Fetched global model for round {data['round']}.")
        return data['weights'], data['round']
    except requests.exceptions.ConnectionError:
        logging.error(f"Client {client_id}: Connection error to server at {server_url}. Retrying...")
        return None, None
    except requests.exceptions.RequestException as e:
        logging.error(f"Client {client_id}: Error fetching global model: {e}")
        return None, None

def submit_model_update(server_url, client_id, weights):
    """Submits the updated local model weights to the server."""
    try:
        response = requests.post(f'{server_url}/submit_model_update', json={
            'client_id': client_id,
            'weights': serialize_weights(weights)
        })
        response.raise_for_status()
        logging.info(f"Client {client_id}: Model update submitted successfully.")
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"Client {client_id}: Error submitting model update: {e}")
        return False

# Main federated learning loop for the client
def run_client_federated_training(client_id, server_url, x_train, y_train, x_test, y_test):
    local_model = create_model()
    current_server_round = -1 # Track the server's current round

    while True:
        # 1. Fetch global model
        weights, server_round = fetch_global_model(server_url)

        if weights is None:
            time.sleep(5) # Wait and retry if server not ready or error
            continue

        if server_round == current_server_round:
            # Server hasn't moved to a new round yet, or we've already submitted for this round
            # Wait for the next round to begin on the server
            logging.info(f"Client {client_id}: Server still on round {server_round}. Waiting for new round...")
            time.sleep(10) # Wait longer before re-checking
            continue

        current_server_round = server_round
        logging.info(f"Client {client_id}: Starting training for federated round {current_server_round}.")

        # Set global model weights to local model
        local_model.set_weights(deserialize_weights(weights))

        # 2. Train local model
        # Use a small number of epochs for demonstration
        try:
            local_model.fit(x_train, y_train, epochs=1, verbose=0) # verbose=0 to suppress per-epoch output
            loss, accuracy = local_model.evaluate(x_test, y_test, verbose=0)
            logging.info(f"Client {client_id}: Round {current_server_round} - Local training complete. Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
        except Exception as e:
            logging.error(f"Client {client_id}: Error during local training: {e}")
            time.sleep(5) # Wait and retry
            continue

        # 3. Submit updated model
        updated_weights = local_model.get_weights()
        if not submit_model_update(server_url, client_id, updated_weights):
            logging.error(f"Client {client_id}: Failed to submit update for round {current_server_round}. Retrying...")
            time.sleep(5) # Wait and retry submission
            continue

        # After submission, wait for the next round to be initiated by the server
        logging.info(f"Client {client_id}: Successfully submitted for round {current_server_round}. Waiting for next round...")
        time.sleep(15) # Give server time to aggregate and move to next round

if __name__ == '__main__':
    # Load data for this specific client
    (x_train, y_train), (x_test, y_test) = load_local_data(data_file)
    run_client_federated_training(client_id, server_url, x_train, y_train, x_test, y_test)
