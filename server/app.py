import threading
import logging

from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Global model definition (simple CNN for MNIST)
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

global_model = create_model()
global_model_weights = global_model.get_weights()

# Store client model updates
client_updates = {}
# Track which clients have submitted updates for the current round
clients_ready_for_aggregation = set()
# Total number of clients expected to participate
NUM_CLIENTS = 3 # This should match the num_clients in split_data.py and run_demo.sh

# Federated learning parameters
FEDERATED_ROUNDS = 5 # Number of rounds of federated training
current_round = 0
round_lock = threading.Lock() # To manage concurrent access to round state

# --- Server-side Test Data for Final Evaluation ---
# In a real federated learning scenario, the central server typically does not
# have access to raw data. For this demonstrator, we'll load a small portion
# of the MNIST test set on the server to show global model performance.
# N.B This data has been seen by one or more cients so this isn't a great test
logging.info("Loading server-side test data for final evaluation...")
(_, _), (x_test_server, y_test_server) = tf.keras.datasets.mnist.load_data()
x_test_server = x_test_server.astype('float32') / 255.0
x_test_server = np.expand_dims(x_test_server, -1)
# Use a subset of the test data to keep it light
x_test_server = x_test_server[:1000]
y_test_server = y_test_server[:1000]
logging.info(f"Server-side test data loaded: {len(x_test_server)} samples.")


# --- Helper Functions ---

def deserialize_weights(weights_list):
    """Converts a list of lists (from JSON) back to a list of numpy arrays."""
    return [np.array(w) for w in weights_list]

def serialize_weights(weights):
    """Converts a list of numpy arrays to a list of lists (for JSON)."""
    return [w.tolist() for w in weights]

def aggregate_models():
    global global_model_weights
    """Aggregates client model weights by simple averaging."""
    logging.info(f"Aggregating models for round {current_round} from {len(client_updates)} clients...")
    if not client_updates:
        logging.warning("No client updates received for aggregation.")
        return

    # Initialize aggregated weights with zeros, matching the shape of global_model_weights
    aggregated_weights = [np.zeros_like(w) for w in global_model_weights]

    # Sum up all client weights
    for client_id, weights_json in client_updates.items():
        client_weights = deserialize_weights(weights_json)
        for i in range(len(aggregated_weights)):
            aggregated_weights[i] += client_weights[i]

    # Average the weights
    num_participating_clients = len(client_updates)
    if num_participating_clients > 0:
        aggregated_weights = [w / num_participating_clients for w in aggregated_weights]
        global_model_weights = aggregated_weights
        global_model.set_weights(global_model_weights) # Update the global model instance
        logging.info(f"Aggregation complete for round {current_round}. Global model updated.")
    else:
        logging.warning("No clients participated in this round for aggregation.")

    # Clear client updates for the next round
    client_updates.clear()
    clients_ready_for_aggregation.clear()

def evaluate_global_model():
    """Evaluates the current global model on the server's test dataset."""
    logging.info("--- Starting Global Model Evaluation ---")
    if x_test_server is None or y_test_server is None:
        logging.error("Server test data not loaded. Cannot evaluate model.")
        return

    try:
        loss, accuracy = global_model.evaluate(x_test_server, y_test_server, verbose=0)
        logging.info(f"--- Global Model Evaluation Results ---")
        logging.info(f"  Test Loss: {loss:.4f}")
        logging.info(f"  Test Accuracy: {accuracy:.4f}")
        logging.info("-------------------------------------")
    except Exception as e:
        logging.error(f"Error during global model evaluation: {e}")



# --- Flask Routes ---

@app.route('/get_global_model', methods=['GET'])
def get_global_model():
    """Endpoint for clients to fetch the current global model weights."""
    logging.info(f"Client requested global model for round {current_round}.")
    return jsonify({
        'round': current_round,
        'weights': serialize_weights(global_model_weights)
    })

@app.route('/submit_model_update', methods=['POST'])
def submit_model_update():
    """Endpoint for clients to submit their updated model weights."""
    data = request.get_json()
    client_id = data.get('client_id')
    weights = data.get('weights')

    if not client_id or not weights:
        return jsonify({'message': 'Missing client_id or weights'}), 400

    with round_lock:
        if client_id in clients_ready_for_aggregation:
            logging.warning(f"Client {client_id} already submitted for round {current_round}. Ignoring duplicate.")
            return jsonify({'message': 'Already submitted for this round'}), 200 # Or 409 Conflict

        client_updates[client_id] = weights
        clients_ready_for_aggregation.add(client_id)
        logging.info(f"Received update from client {client_id}. Total updates: {len(client_updates)}/{NUM_CLIENTS}")

        # Check if all expected clients have submitted
        if len(clients_ready_for_aggregation) >= NUM_CLIENTS:
            logging.info(f"All {NUM_CLIENTS} clients submitted for round {current_round}. Initiating aggregation.")
            # Start aggregation in a new thread to not block the current request
            threading.Thread(target=perform_federated_round).start()
        else:
            logging.info(f"Waiting for {NUM_CLIENTS - len(clients_ready_for_aggregation)} more clients for round {current_round}.")

    return jsonify({'message': 'Model update received successfully'})

@app.route('/status', methods=['GET'])
def status():
    """Endpoint to check server status and current round."""
    return jsonify({
        'current_round': current_round,
        'federated_rounds_total': FEDERATED_ROUNDS,
        'clients_submitted_this_round': list(clients_ready_for_aggregation),
        'expected_clients': NUM_CLIENTS
    })

# --- Federated Learning Orchestration ---

def perform_federated_round():
    """
    Orchestrates a single round of federated learning:
    1. Aggregates models.
    2. Increments round counter.
    """
    global current_round
    with round_lock:
        if current_round >= FEDERATED_ROUNDS:
            logging.info("All federated rounds completed. Stopping server.")
            return

        logging.info(f"Starting aggregation for federated round {current_round}...")
        aggregate_models()
        current_round += 1
        logging.info(f"Completed federated round {current_round-1}. Moving to round {current_round}.")

        # Check if this was the last round
        if current_round >= FEDERATED_ROUNDS:
            logging.info(f"Federated rounds complete ({FEDERATED_ROUNDS} rounds). Triggering final evaluation.")
            evaluate_global_model()
        else:
            logging.info("Global model weights updated. Clients can now fetch new model for next round.")

# For this demo, we'll let the clients trigger the rounds by submitting.
# The server will aggregate once all clients submit for a given round.

if __name__ == '__main__':
    # This ensures the Flask app runs and is ready to receive requests.
    # The actual federated rounds are triggered by client submissions.
    logging.info("Federated Learning Server starting...")
    logging.info(f"Expecting {NUM_CLIENTS} clients for {FEDERATED_ROUNDS} rounds.")
    app.run(host='0.0.0.0', port=5000, debug=False) # debug=False for production-like environment
