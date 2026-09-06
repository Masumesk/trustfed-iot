import argparse
import pickle

from client_app.api_client import register_client, set_server_url
from config import DATASET, SERVER_URL
from data.load_dataset import load_dataset
from federated.client import Client

parser = argparse.ArgumentParser()
parser.add_argument("--id", type=int, required=True)
parser.add_argument("--server", type=str, default=SERVER_URL)
args = parser.parse_args()
client_id = args.id
set_server_url(args.server)
train_dataset, _, _ = load_dataset(DATASET, load_test=False)

with open("data/partition_cache.pkl", "rb") as f:
    client_indices = pickle.load(f)

client = Client(
    client_id=client_id,
    dataset=train_dataset,
    indices=client_indices[client_id],
    num_classes=10,
)

info = client.get_client_distribution()
response = register_client(info)
print(response)
