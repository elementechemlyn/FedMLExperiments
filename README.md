# Federated Learning Demo
[AI Generated README]

This project demonstrates a simple federated learning setup using Docker, Flask, and TensorFlow/Keras. Multiple client containers train a model on their own data and periodically send updates to a central server, which aggregates the updates to improve a global model.

## Project Structure

```
cleanup.sh
run_demo.sh
client/
    client.py
    Dockerfile
    requirements.txt
data/
    split_data.py
server/
    app.py
    Dockerfile
    requirements.txt
```

- **server/**: Flask server that coordinates federated learning and aggregates model updates.
- **client/**: Client code that trains a model on local data and communicates with the server.
- **data/**: Scripts and files for splitting and storing client datasets.
- **run_demo.sh**: Script to build images, split data, and launch the demo using Docker.
- **cleanup.sh**: Script to stop and remove containers and network (built by run_demo.sh).

## Quick Start

### Prerequisites

- [Docker](https://www.docker.com/)
- [Python 3.9+](https://www.python.org/) (for running data split script outside Docker)

### Steps

1. **Clone the repository**  
   ```sh
   git clone git@github.com:elementechemlyn/FedMLExperiments.git
   cd fedlearning
   ```

2. **Run the demo**  
   ```sh
   ./run_demo.sh
   ```

   This will:
   - Split the MNIST dataset among clients
   - Build Docker images for server and clients
   - Start the server and client containers on a Docker network

3. **Monitor logs**  
   - Server:  
     ```sh
     docker logs -f fl_server
     ```
   - Client (replace 0 with client ID):  
     ```sh
     docker logs -f fl_client_0
     ```

4. **Cleanup**  
   ```sh
   ./cleanup.sh
   ```

## Customization

- **Number of Clients**:  
  Change `NUM_CLIENTS` in [`run_demo.sh`](run_demo.sh), [`server/app.py`](server/app.py), and [`data/split_data.py`](data/split_data.py) to match your desired number of clients.

- **Model Architecture**:  
  Update the `create_model()` function in both [`server/app.py`](server/app.py) and [`client/client.py`](client/client.py) to experiment with different models.

- **Weight Aggregation**
 The function server/app.aggregate_models performs a simple averaging of weights.

## How It Works

- The server waits for model updates from all clients each round.
- Each client trains locally on its own data and sends updated weights to the server.
- The server aggregates the weights and sends the new global model back to the clients.
- This process repeats for several rounds.

## File Descriptions

- [`server/app.py`](server/app.py): Federated learning server logic.
- [`client/client.py`](client/client.py): Client-side training and communication.
- [`data/split_data.py`](data/split_data.py): Splits MNIST data among clients.
- [`run_demo.sh`](run_demo.sh): Orchestrates the demo setup and execution.
- [`cleanup.sh`](cleanup.sh): Cleans up Docker containers and network.

## TODO
- Implement FedProx
- Add functions to report and compare results
- Add support for fixed seeds in model creation
- When splitting data, set aside some validation data to test how the global model performs compared to the local models 
- Investigate ways to distribute the model architecture from server to clients
- Try some different datasets
- Add some tests
- Let clients know when all rounds are complete so they can exit
## License

MIT License

---

*This project is for educational purposes and not intended for production use
