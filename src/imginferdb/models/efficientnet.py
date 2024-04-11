import torch
from torchvision import models, transforms

from .embedding_model import BaseEmbeddingModel


class EfficientNetEmbeddings(BaseEmbeddingModel):
    def __init__(self, version="b0", pretrained=True):
        # Load the pre-trained EfficientNet model
        model_name = f"efficientnet_{version}"

        self.model = getattr(models, model_name)(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )

        # Replace the classifier with an identity layer to get embeddings
        self.model.classifier = torch.nn.Identity()

        # Define preprocessing transformations
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def generate_embeddings(self, test_dataloader):
        # Apply preprocessing
        image = self.preprocess(image)

        # Add batch dimension and send to the same device as the model
        image = image.unsqueeze(0).to(next(self.model.parameters()).device)

        # Set model to evaluation mode and with no_grad
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model(image)

        return embeddings

    def build_embeddings(self, train_dataloader):
        pass
