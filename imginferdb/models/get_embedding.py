from .autoencoder import AutoEncoder
from .byol import BYOL
from .efficientnet import EfficientNetEmbeddings
from .mobilenet import MobileNetEmbeddings


def get_embedding_model(embedding_model_name: str):
    if embedding_model_name == "EfficientNet":
        embedding_model = EfficientNetEmbeddings()
    elif embedding_model_name == "MobileNet":
        embedding_model = MobileNetEmbeddings()
    elif embedding_model_name == "AutoEncoder":
        embedding_model = AutoEncoder()
    elif embedding_model_name == "BYOL":
        embedding_model = BYOL()
    else:
        raise NotImplementedError(f"No Embedding Model Type: {embedding_model_name} defined")

    return embedding_model
