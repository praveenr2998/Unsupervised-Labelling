labelling_prompt = {
    "system": "You are an expert data labeller who can label topics based on the provided label config",
    "user": """
    You are expected to map the topics to a label config and return the labelled data
    
    The topics are provided in the below format where each ID is the key and topics are the values
    '`topic_dict`'
    
    The label config is provided in the below format
    '`label_config`'
    
    Your task is to assign a label to each ID(containing many topics) in the data based on the label config and return the labelled data in the below format
    {ID: label}
    """
}