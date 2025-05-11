import httpx
import os  # Ensure os is imported

class MyHttpClient(httpx.Client):
    def __deepcopy__(self, memo):
        return self

# Helper function to retrieve API keys from environment variables
def get_api_key(env_variable_name, default=None):
    key = os.environ.get(env_variable_name)
    if key is None:
        print(f"[Warning] Environment variable {env_variable_name} not set. Model may not be usable.")
    return key

config_list = [
    {
        "model": "gpt-4o",
        "api_key": get_api_key("OPENAI_API_KEY_GPT4O"),  # Replaced hardcoded key
        "http_client": MyHttpClient(proxy="http://localhost:8019"),
        "tags": ["oai-gpt-4o"]
    },
    {
        "model": "gpt-4o",
        "api_key": get_api_key("LOCAL_API_KEY_GPT4O"),  # Replaced hardcoded key
        "base_url": "http://127.0.0.1:8888/v1/",
        "tags": ["local-gpt-4o"],
        "http_client": MyHttpClient(proxy="http://localhost:8019")
    },
    {
        "model": "gpt-4",
        "api_key": get_api_key("LOCAL_API_KEY_GPT4"),  # Replaced hardcoded key
        "base_url": "http://127.0.0.1:8888/v1/",
        "tags": ["local-gpt-4"],
        "http_client": MyHttpClient(proxy="http://localhost:8019")
    },
    {
        "model": "gemini-1.5-pro-latest",
        "api_key": get_api_key("GOOGLE_API_KEY_GEMINI_PRO"),  # Replaced hardcoded key
        "api_type": "google",
        "tags": ["gm-pro"],
        "http_client": MyHttpClient(proxy="http://localhost:8021"),
        "safety_settings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ]
    },
    {
        "model": "gemini-1.5-flash-latest",  # Corrected model name
        "api_key": get_api_key("GOOGLE_API_KEY_GEMINI_FLASH"),  # Replaced hardcoded key
        "api_type": "google",
        "tags": ["gm-flash"],  # Corrected tag
        "http_client": MyHttpClient(proxy="http://localhost:20171"),  # Ensure this proxy is correct for your setup
        "safety_settings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ]
    }
]

# To use these LLM configurations, please set the following environment variables:
# - OPENAI_API_KEY_GPT4O: Your OpenAI API key for GPT-4o.
# - LOCAL_API_KEY_GPT4O: Your API key for the local GPT-4o instance (e.g., "congliu-api-key").
# - LOCAL_API_KEY_GPT4: Your API key for the local GPT-4 instance (e.g., "congliu-api-key").
# - GOOGLE_API_KEY_GEMINI_PRO: Your Google API key for Gemini 1.5 Pro.
# - GOOGLE_API_KEY_GEMINI_FLASH: Your Google API key for Gemini 1.5 Flash.
#
# If an API key environment variable is not set for a model,
# that model's configuration might be skipped or cause errors when AutoGen attempts to use it.
# The 'http_client' with proxy settings is specific to this configuration;
# ensure your proxy servers are running at the specified addresses (e.g., localhost:8019).