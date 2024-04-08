from .embedding_model import BaseEmbeddingModel


class AutoEncoderEmbeddings(BaseEmbeddingModel):
    def __init__(self) -> None:
        super().__init__()

    def generate_embeddings(self, test_dataloader):
        pass

    def build_embeddings(self, train_dataloader):
        pass
