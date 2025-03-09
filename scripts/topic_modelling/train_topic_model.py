from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic

class TrainTopicModel:
    def __init__(self, dataset, data_cache_dir, data_col_name, embedding_model, embedding_model_cache_folder,
                 topic_model_dir, hash_key):
        self.dataset = dataset
        self.data_cache_dir = data_cache_dir
        self.data_col_name = data_col_name
        self.embedding_model = embedding_model
        self.embedding_model_cache_folder = embedding_model_cache_folder
        self.topic_model_dir = topic_model_dir
        self.hash_key = hash_key

    def download_hf_dataset(self):
        load_dataset(path=self.dataset, cache_dir=self.data_cache_dir)

    def train_topic_model(self):
        print("Loading Dataset ...")
        dataset = load_dataset(path=self.dataset, cache_dir=self.data_cache_dir)
        train_dataset = dataset["train"]
        text = train_dataset[self.data_col_name]

        print("Creating Embeddings ...")
        embedding_model = SentenceTransformer(
            model_name_or_path=self.embedding_model, cache_folder=self.embedding_model_cache_folder
        )
        embeddings = embedding_model.encode(text, show_progress_bar=True)

        print("Dimensionality Reduction ...")
        umap_model = UMAP(
            n_components=5, min_dist=0.0, metric='cosine', random_state=42
        )
        reduced_embeddings = umap_model.fit_transform(embeddings)

        print("Clustering ...")
        hdbscan_model = HDBSCAN(
            min_cluster_size=6, metric='euclidean', cluster_selection_method='eom', prediction_data=True
        ).fit(reduced_embeddings)
        clusters = hdbscan_model.labels_
        print("Count of clusters formed from the data : ", len(set(clusters)))

        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            verbose=True
        ).fit(text, embeddings)

        topic_model.save(path=f"{self.topic_model_dir}/trained_model_{self.hash_key}", save_embedding_model=True)