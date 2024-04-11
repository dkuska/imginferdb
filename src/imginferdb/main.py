from loguru import logger
from models import get_embedding_model
from utils import get_dataloader, load_config


def generate_embeddings(dataloader, model):
    pass
    # return model.generate_embeddings(dataloader)


def perform_downstream_task(embeddings, task_model):
    pass
    # return task_model.run(embeddings)


def main():
    # Load configurations
    config = load_config("imginferdb/config.yaml")

    logger.info("Loaded Configurations")

    logger.info("Starting Experiments")
    # Iterate over datasets
    for dataset_name, dataset_config in config["datasets"].items():
        logger.info(f"Dataset: {dataset_name}")
        # Load Dataset and get DataLoader
        train_dataloader, test_dataloader = get_dataloader(dataset_name)

        # Iterate over embedding models
        for embedding_model_name, embedding_model_config in config[
            "embedding_models"
        ].items():
            logger.info(f"Embedding: {embedding_model_name}")
            # Load Embedding Model
            embedding_model = get_embedding_model(embedding_model_name)
            logger.info("Successfully loaded Embedding Model")

            logger.info("Generating Embeddings")
            # Generate Embeddings
            embeddings = generate_embeddings(train_dataloader, embedding_model)
            logger.info("Successfully generated Embeddings")

            # Perform Downstream Task
            task_results = perform_downstream_task(embeddings, None)


if __name__ == "__main__":
    main()
