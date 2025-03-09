from scripts.topic_modelling.train_topic_model import TrainTopicModel
from utils.utilities import generate_hash_key, save_dict_to_json

if __name__=="__main__":
    hash_key = generate_hash_key()

    train_config = {
        "dataset": "SetFit/bbc-news",
        "data_cache_dir": "data/input",
        "data_col_name": "text",
        "embedding_model": "thenlper/gte-small",
        "embedding_model_cache_folder": "models/embedding_models",
        "topic_model_dir": "models/topic_models",
        "hash_key": hash_key
    }

    save_dict_to_json(data=train_config, dir_path=f"data/output/{hash_key}", filename="train_config.json")

    trainer = TrainTopicModel(**train_config)
    trainer.download_hf_dataset()
    trainer.train_topic_model()