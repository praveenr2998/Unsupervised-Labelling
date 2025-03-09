import os
import openai

from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class LLMUtils:
    def __init__(self, llm_choice):
        if llm_choice == "azure_openai":
            self.client = AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_BASE"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_VERSION")
            )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10),
           retry=retry_if_exception_type(openai.OpenAIError))
    def chat_completion(self, system_prompt, user_prompt, response_model, chat_history=None):
        """
        LLM chat completion API

        :param system_prompt: str - system prompt
        :param user_prompt: str - user prompt
        :param response_model: pydantic model - llm response model
        :param chat_history: str - chat history
        :return: pydantic model - response from llm
        """
        if chat_history:
            chat_history.extend([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ])
            prompt = chat_history
        else:
            prompt = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

        completion = self.client.beta.chat.completions.parse(
            model=os.getenv("AZURE_OPENAI_MODEL"),
            messages=prompt,
            response_format=response_model,
        )
        parsed_response = completion.choices[0].message.parsed
        self.validate_llm_response(parsed_response=parsed_response, response_model=response_model)
        return parsed_response


    @staticmethod
    def validate_llm_response(parsed_response, response_model):
        """
        Used to validate LLM response using pydantic response model

        :param parsed_response: obj - response from LLM
        :param response_model: pydantic model - pydantic response model
        :return: None
        """
        try:
            response_model(**parsed_response.__dict__)
        except Exception as error:
            raise Exception("LLM response is not following the desired response model")