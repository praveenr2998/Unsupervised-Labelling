import hashlib
import datetime
import os
import json

def generate_hash_key():
    """
    Create hash key based on datetime

    :return: hash_key - generated hash key
    """
    now = datetime.datetime.now().isoformat()

    # Encode and hash using SHA-256
    hash_object = hashlib.sha256(now.encode())
    hash_key = hash_object.hexdigest()

    return hash_key


def save_dict_to_json(data, dir_path, filename):
    """
    Saves a dictionary as a JSON file in the specified directory.
    Creates the directory if it does not exist.

    :param data: Dictionary to save
    :param dir_path: Directory where the JSON file should be saved
    :param filename: Name of the JSON file (should include .json extension)
    """
    os.makedirs(dir_path, exist_ok=True)

    file_path = os.path.join(dir_path, filename)

    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)

    print(f"Dictionary saved to {file_path}")
