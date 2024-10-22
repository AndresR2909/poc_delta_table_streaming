import os
import logging
from langchain_openai import AzureChatOpenAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "suicide_intent_classification",
        "schema": {
            "type": "object",
            "properties": {
                "clase": {
                    "type": "integer",
                    "enum": [0, 1]  # 0 o 1 para la clasificación
                },
                "probabilidad": {
                    "type": "number",
                    "minimum": 0,   # Valor mínimo
                    "maximum": 1    # Valor máximo
                }
            },
            "required": ["clase", "probabilidad"],  # Claves requeridas
            "additionalProperties": False  # No se permiten propiedades adicionales
        },
        "strict": True
    }
}

class OpenAIChatModel():
    def __init__(self,temperature=0.1, azure= True, deployment = "gpt-4o-mini", api_version = "2024-02-15-preview" )-> None:
        self.temperature = temperature
        self.max_tokens = 800
        self.request_timeout = 120
        self.max_retries = 3
        self.response_format= response_format
        try:
            if azure:
                self.model = deployment
                self.api_version = api_version
                self.chat_model_client = self._get_azure_client()
            else:
                self.model = deployment
                self.chat_model_client = self._get_openia_client()
        
        except Exception as e:
            logging.error(f"Error connecting to azure endpoint: {e}")

    def _get_azure_client(self):
        chat_model_client = AzureChatOpenAI( openai_api_version =  self.api_version,
                                            api_key=AZURE_OPENAI_API_KEY,
                                            azure_deployment = self.model,
                                            temperature = self.temperature,
                                            timeout = self.request_timeout,
                                            max_retries = self.max_retries,
                                            max_tokens = self.max_tokens)
        return chat_model_client

    def _get_openia_client(self):
        chat_model_client = ChatOpenAI(
                                model=self.model,
                                api_key= OPENAI_API_KEY,
                                temperature = self.temperature,
                                timeout = self.request_timeout,
                                max_retries = self.max_retries,
                                response_format=self.response_format,
                                max_tokens = self.max_tokens)
        return chat_model_client