from scripts.topic_modelling.train_topic_model import TrainTopicModel
from scripts.labelling.labelling import Labeller
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

    label_config = {
        "business": "any business related topic",
        "entertainment": "any entertainment related topic",
        "health": "any health related topic",
        "science": "any science related topic",
        "sports": "any sports related topic",
        "technology": "any technology related topic"
    }
    save_dict_to_json(data=train_config, dir_path=f"data/output/{hash_key}", filename="label_config.json")

    # Training Topic Model
    trainer = TrainTopicModel(**train_config)
    trainer.download_hf_dataset()
    trainer.train_topic_model()

    # Labelling Data
    labeller = Labeller(**train_config)
    topic_info_df, document_info_df = labeller.get_topic_model_info()
    labeller.label_good_quality_data(document_info_df=document_info_df, label_config=label_config)
    labeller.label_poor_quality_data(document_info_df=document_info_df, label_config=label_config)
    labeller.collate_labelled_data(document_info_df=document_info_df)