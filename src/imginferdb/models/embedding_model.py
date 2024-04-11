from abc import ABC, abstractmethod


class BaseEmbeddingModel(ABC):
    @abstractmethod
    def generate_embeddings(self, test_dataloader):
        pass

    @abstractmethod
    def build_embeddings(self, train_dataloader):
        pass
