from openenv.core.env_server import EnvClient
from server.models import CleanAction, DataObservation

class DataCleaningClient(EnvClient):
    """
    Client for interacting with the Data Integrity Lab environment.
    This allows agents to connect to the hosted Space.
    """
    def __init__(self, base_url: str):
        super().__init__(
            base_url=base_url,
            action_cls=CleanAction,
            observation_cls=DataObservation
        )