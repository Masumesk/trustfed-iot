class Server:
    def __init__(self, clients):
        self.clients = clients

        # Server-side state
        self.trust_scores = {
            client.client_id: 1.0
            for client in clients
        }

        self.cluster_assignments = {}
        self.selection_scores = {}
        self.client_roles = {}
        self.client_status = {}