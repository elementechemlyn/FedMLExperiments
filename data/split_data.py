import argparse

import numpy as np
import tensorflow as tf
import os

def split_mnist_data(num_clients=3, output_dir='data', seed=None):
    """
    Loads the MNIST dataset, shuffles it, and splits it into N parts
    for federated learning clients. Each client gets a portion of the
    training and testing data.

    Args:
        num_clients (int): The number of clients to simulate.
        output_dir (str): Directory to save the split data files.
        seed (int): Random seed for shuffling.

    Returns:
        None
    """
    print(f"Loading MNIST dataset and splitting for {num_clients} clients...")

    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize pixel values to be between 0 and 1
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Reshape data to include channel dimension for CNNs (if using later)
    # (60000, 28, 28) -> (60000, 28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # Combine train and test for shuffling and splitting
    x_combined = np.concatenate((x_train, x_test), axis=0)
    y_combined = np.concatenate((y_train, y_test), axis=0)

    # Shuffle the combined dataset
    indices = np.arange(len(x_combined))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    x_combined = x_combined[indices]
    y_combined = y_combined[indices]

    # Calculate samples per client
    total_samples = len(x_combined)
    samples_per_client = total_samples // num_clients

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Split data and save for each client
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client if i < num_clients - 1 else total_samples

        client_x = x_combined[start_idx:end_idx]
        client_y = y_combined[start_idx:end_idx]

        split_point = int(len(client_x) * 0.8) # 80% train, 20% test for each client

        client_x_train = client_x[:split_point]
        client_y_train = client_y[:split_point]
        client_x_test = client_x[split_point:]
        client_y_test = client_y[split_point:]


        file_path = os.path.join(output_dir, f'client_{i}_data.npz')
        np.savez_compressed(file_path,
                            x_train=client_x_train,
                            y_train=client_y_train,
                            x_test=client_x_test,
                            y_test=client_y_test)
        print(f"Saved data for client {i} to {file_path}")
        print(f"  Train samples: {len(client_x_train)}, Test samples: {len(client_x_test)}")

    print("Data splitting complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("simple_example")
    parser.add_argument("num_clients", help="The number of federated clients to create data for", type=int)
    parser.add_argument("--output_dir", help="The name of the folder to output files to", type=str, default="data")
    parser.add_argument("--seed", help="Random seed to use when shuffling data", type=int)
    args = parser.parse_args()
    num_clients = args.num_clients
    output_dir = args.output_dir
    seed = args.seed
    print(f"Making {num_clients} sets of files in {output_dir} with seed {seed}")
    split_mnist_data(num_clients,output_dir,seed) # You can change the number of clients here
