import re

import pandas as pd
from bertopic import BERTopic
from datasets import load_dataset
from keybert import KeyBERT
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from utils.llm_utils import LLMUtils
from utils.prompts import good_quality_labelling_prompt, poor_quality_labelling_prompt
from utils.pydantic_models import (
    GoodQualityLabellingResponseModel,
    PoorQualityLabellingResponseModel,
)
from utils.utilities import save_dict_to_json


class Labeller:
    """
    Class to label data using trained topic model, keybert and LLM
    """

    def __init__(
        self,
        dataset,
        data_cache_dir,
        data_col_name,
        embedding_model,
        embedding_model_cache_folder,
        topic_model_dir,
        hash_key,
    ):
        self.dataset = dataset
        self.data_cache_dir = data_cache_dir
        self.data_col_name = data_col_name
        self.embedding_model = embedding_model
        self.embedding_model_cache_folder = embedding_model_cache_folder
        self.topic_model_dir = topic_model_dir
        self.hash_key = hash_key
        self.info_dir = f"data/output/{self.hash_key}"
        self.llm = LLMUtils(llm_choice="azure_openai")
        self.keybert_model = KeyBERT()

    def get_topic_model_info(self):
        """
        Using the saved topic model to get topic info and document info files

        :return: topic_info_df - topic info dataframe, document_info_df - document info dataframe
        """
        print("Loading Dataset ...")
        dataset = load_dataset(path=self.dataset, cache_dir=self.data_cache_dir)
        train_dataset = dataset["train"]
        text = train_dataset[self.data_col_name]

        trained_topic_model = BERTopic.load(
            path=f"{self.topic_model_dir}/trained_model_{self.hash_key}"
        )
        topic_info_df = trained_topic_model.get_topic_info()
        document_info_df = trained_topic_model.get_document_info(docs=text)

        print("Saving Topic Info and Document Info ...")
        topic_info_df.to_csv(path_or_buf=f"{self.info_dir}/topic_info.csv", index=False)
        document_info_df.to_csv(
            path_or_buf=f"{self.info_dir}/document_info.csv", index=False
        )
        return topic_info_df, document_info_df

    def label_good_quality_data(self, document_info_df, label_config):
        """
        Using the topic created from the input dataset using the trained topic model LLM is used to label the data based
        on the label config. Good quality dataset is selected based on topic and probability.

        :param document_info_df: dataframe containing document info
        :param label_config: dictionary containing label config
        :return: None
        """
        print("Labelling Good Quality Data ...")

        filtered_doc_info_df = document_info_df[
            (document_info_df["Topic"] != -1) & (document_info_df["Probability"] > 0.59)
        ]

        topic_label_dict = {}
        all_topics = filtered_doc_info_df["Representation"].astype(str).unique()
        system_prompt = good_quality_labelling_prompt.get("system")
        for topic in all_topics:
            user_prompt = good_quality_labelling_prompt.get("user")
            user_prompt = user_prompt.replace("`topic`", topic).replace(
                "`label_config`", str(label_config)
            )
            llm_response = self.llm.chat_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=GoodQualityLabellingResponseModel,
            )
            topic_label_dict[topic] = llm_response.label

        for idx, row in filtered_doc_info_df.iterrows():
            topic = row["Representation"]
            label = topic_label_dict.get(str(topic))
            filtered_doc_info_df.at[idx, "Label"] = label

        filtered_doc_info_df.to_csv(
            path_or_buf=f"data/output/{self.hash_key}/good_quality_data.csv",
            index=False,
        )
        save_dict_to_json(
            data=topic_label_dict,
            dir_path=f"data/output/{self.hash_key}",
            filename="topic_label_dict.json",
        )

    def label_poor_quality_data(self, document_info_df, label_config):
        """
        Using the topic created from the input dataset using the trained topic model LLM is used to label the data based
        on the label config. Poor quality dataset is selected based on topic and probability.

        :param document_info_df: dataframe containing document info
        :param label_config: data containing label config
        :return: None
        """
        print("Labelling Poor Quality Data ...")
        iter_size = 5

        filtered_doc_info_df = document_info_df[
            (document_info_df["Topic"] == -1) | (document_info_df["Probability"] < 0.6)
        ]

        filtered_doc_info_df = filtered_doc_info_df.reset_index(drop=True)
        filtered_doc_info_df["ID"] = filtered_doc_info_df.index + 1
        for i in tqdm(
            range(0, len(filtered_doc_info_df), iter_size),
            desc="Labelling Poor Quality Data",
        ):
            chunk = filtered_doc_info_df.iloc[i : i + iter_size]
            topic_dict = {}
            for index, row in chunk.iterrows():
                if not isinstance(row["Document"], float):
                    extracted_keywords = self.keybert_model.extract_keywords(
                        row["Document"], keyphrase_ngram_range=(1, 1), top_n=5
                    )
                    topics = [keyword[0] for keyword in extracted_keywords]
                    topic_dict[row["ID"]] = topics
                    filtered_doc_info_df.loc[
                        filtered_doc_info_df["ID"] == row["ID"], "KeyBERT_Topics"
                    ] = str(topics)

            system_prompt = poor_quality_labelling_prompt.get("system")
            user_prompt = poor_quality_labelling_prompt.get("user")
            user_prompt = user_prompt.replace("`topic_dict`", str(topic_dict)).replace(
                "`label_config`", str(label_config)
            )
            llm_response = self.llm.chat_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=PoorQualityLabellingResponseModel,
            )
            labelled_data = llm_response.labelled_data
            for labelled_data in labelled_data:
                filtered_doc_info_df.loc[
                    filtered_doc_info_df["ID"] == labelled_data.id, "Label"
                ] = labelled_data.label

        filtered_doc_info_df.to_csv(
            path_or_buf=f"data/output/{self.hash_key}/poor_quality_data.csv",
            index=False,
        )

    def collate_labelled_data(self):
        """
        Collate labelled data from good and poor quality data

        :return: None
        """
        print("Collating Labelled Data ...")
        good_quality_data = pd.read_csv(
            f"data/output/{self.hash_key}/good_quality_data.csv"
        )
        poor_quality_data = pd.read_csv(
            f"data/output/{self.hash_key}/poor_quality_data.csv"
        )
        good_quality_data_df = good_quality_data[["Document", "Label"]]
        poor_quality_data_df = poor_quality_data[["Document", "Label"]]
        combined_df = pd.concat(
            [good_quality_data_df, poor_quality_data_df], ignore_index=True
        )

        dataset = load_dataset(path=self.dataset, cache_dir=self.data_cache_dir)
        train_dataset = dataset["train"].to_pandas()

        for idx, row in combined_df.iterrows():
            document = row["Document"]
            label = row["Label"]
            train_dataset.loc[
                train_dataset[self.data_col_name].str.contains(
                    re.escape(document), na=False
                ),
                "predicted_label",
            ] = label

        train_dataset.to_csv(
            path_or_buf=f"data/output/{self.hash_key}/final_labelled_data.csv",
            index=False,
        )
        accuracy = accuracy_score(
            train_dataset["label_text"], train_dataset["predicted_label"]
        )

        print(f"Accuracy: {accuracy:.2%}")
