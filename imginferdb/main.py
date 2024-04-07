from loguru import logger

from .config import load_config
from .models import get_embedding_model
from .utils.data_loader import get_dataloader


def generate_embeddings(dataloader, model):
    return model.generate_embeddings(dataloader)


def perform_downstream_task(embeddings, task_model):
    return task_model.run(embeddings)


def main():
    logger.info("Starting Application")
    # Load configurations
    embedding_config = load_config("configs/embeddings.yaml")
    dataset_config = load_config("configs/datasets.yaml")
    logger.info("Loaded Configurations")

    logger.info("Starting Experiments")
    for dataset_name in dataset_config["datasets"]:
        logger.info(f"Dataset: {dataset_name}")
        dataloader = get_dataloader(dataset_name)
        logger.info(f"Successfully got DataLoader for dataset: {dataset_name}")

        for embedding_model_name in embedding_config["embedding_models"]:
            logger.info(
                f"Starting Experiment for Embedding Model: {embedding_model_name}"
            )
            embedding_model = get_embedding_model(embedding_model_name)
            logger.info(f"Successfully loaded Embedding Model")

            logger.info("Generating Embeddings")
            embeddings = generate_embeddings(dataloader, embedding_model)
            logger.info("Successfully generated Embeddings")

            task_results = perform_downstream_task(
                embeddings, None
            )


if __name__ == "__main__":
    main()
