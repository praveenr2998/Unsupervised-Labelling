# UNSUPERVISED LABELLING
Labelling huge amount of data is a tedious task. This project aims to automate the process of labelling data using a combination of topic model and a LLM.



-----------------

## SETUP
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync
```
### ENVIRONMENT VARIABLES
```dotenv
AZURE_OPENAI_BASE=https://your-resource-name.openai.azure.com
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_VERSION=2023-05-15
AZURE_OPENAI_MODEL=gpt-35-turbo
```
NOTE: code can be replaced with any other LLM this setup used azure openai

-----------------