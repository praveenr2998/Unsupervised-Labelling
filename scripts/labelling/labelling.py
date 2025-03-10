"""
TODO: Remove break statement from labelling loop
"""

from bertopic import BERTopic
from datasets import load_dataset
from utils.prompts import labelling_prompt
from utils.pydantic_models import LabellingResponseModel
from utils.llm_utils import LLMUtils
from utils.utilities import save_dict_to_json
from keybert import KeyBERT
import json

class Labeller:
    """
    Class to label data using trained topic model, keybert and LLM
    """
    def __init__(self, dataset, data_cache_dir, data_col_name, embedding_model, embedding_model_cache_folder,
                 topic_model_dir, hash_key):
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

        trained_topic_model = BERTopic.load(path=f"{self.topic_model_dir}/trained_model_{self.hash_key}")
        topic_info_df = trained_topic_model.get_topic_info()
        topic_info_df['ID'] = topic_info_df.index + 1
        document_info_df = trained_topic_model.get_document_info(docs=text)
        document_info_df['ID'] = document_info_df.index + 1

        print("Saving Topic Info and Document Info ...")
        topic_info_df.to_csv(path_or_buf=f"{self.info_dir}/topic_info.csv", index=False)
        document_info_df.to_csv(path_or_buf=f"{self.info_dir}/document_info.csv", index=False)
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
        iter_size = 5
        labelled_data_dict = {}

        filtered_doc_info_df = document_info_df[(document_info_df["Topic"] != -1) &
                                                (document_info_df["Probability"] > 0.59)]

        for i in range(0, len(filtered_doc_info_df), iter_size):
            chunk = filtered_doc_info_df.iloc[i:i + iter_size]
            topic_dict = {}
            for index, row in chunk.iterrows():
                topic_dict[row["ID"]] = row["Topic"]

            system_prompt = labelling_prompt.get("system")
            user_prompt = labelling_prompt.get("user")
            user_prompt = user_prompt.replace("`topic_dict`", str(topic_dict)).replace(
                "`label_config`", str(label_config))
            llm_response = self.llm.chat_completion(system_prompt=system_prompt, user_prompt=user_prompt,
                                                    response_model=LabellingResponseModel)
            labelled_data = llm_response.labelled_data
            for labelled_data in labelled_data:
                labelled_data_dict[labelled_data.id] = labelled_data.label

        save_dict_to_json(data=labelled_data_dict, dir_path=f"data/output/{self.hash_key}",
                          filename="labelled_data_confident.json")

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
        labelled_data_dict = {}

        filtered_doc_info_df = document_info_df[(document_info_df["Topic"] == -1) |
                                                (document_info_df["Probability"] < 0.6)]

        for i in range(0, len(filtered_doc_info_df), iter_size):
            chunk = filtered_doc_info_df.iloc[i:i + iter_size]
            topic_dict = {}
            for index, row in chunk.iterrows():
                extracted_keywords = self.keybert_model.extract_keywords(row["Document"], keyphrase_ngram_range=(1, 1),
                                                                         top_n=5)
                topics = [keyword[0] for keyword in extracted_keywords]
                topic_dict[row["ID"]] = topics

            system_prompt = labelling_prompt.get("system")
            user_prompt = labelling_prompt.get("user")
            user_prompt = user_prompt.replace("`topic_dict`", str(topic_dict)).replace(
                "`label_config`", str(label_config))
            llm_response = self.llm.chat_completion(system_prompt=system_prompt, user_prompt=user_prompt,
                                                    response_model=LabellingResponseModel)
            labelled_data = llm_response.labelled_data
            for labelled_data in labelled_data:
                labelled_data_dict[labelled_data.id] = labelled_data.label

        save_dict_to_json(data=labelled_data_dict, dir_path=f"data/output/{self.hash_key}",
                          filename="labelled_data_non_confident.json")

    def collate_labelled_data(self, document_info_df):
        """
        Collate labelled data from good and poor quality data

        :param document_info_df: dataframe containing document info
        :return: None
        """
        print("Collating Labelled Data ...")
        labelled_data_dict = json.load(open(f"data/output/{self.hash_key}/labelled_data_confident.json"))
        labelled_data_dict.update(json.load(open(f"data/output/{self.hash_key}/labelled_data_non_confident.json")))
        document_info_df["Label"] = document_info_df["ID"].apply(lambda x: labelled_data_dict.get(x))
        document_info_df.to_csv(path_or_buf=f"data/output/{self.hash_key}/final_labelled_data.csv", index=False)