from byol_pytorch import BYOL
from torchvision import models

from .embedding_model import BaseEmbeddingModel


class BYOLEmbeddings(BaseEmbeddingModel):
    def __init__(self, image_size=256) -> None:
        self.image_size = image_size
        self.resnet = models.resnet50(pretrained=True)

        self.learner = BYOL(
            self.resnet, image_size=self.image_size, hidden_layer="avgpool"
        )

    def generate_embeddings(self):
        pass
