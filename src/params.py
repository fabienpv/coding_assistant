import os

QWEN35_PARAMS = {
    "coding_reasoning": {
        "temperature": 0.6,
        "top_p": 0.95,
        "presence_penalty": 0.0,
        "extra_body": {
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": True}
        }
    },
    "general_reasoning": {
        "temperature": 1.0,
        "top_p": 0.95,
        "presence_penalty": 1.5,
        "extra_body": {
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": True}
        }
    },
    "general_instruct": {
        "temperature": 0.7,
        "top_p": 0.8,
        "presence_penalty": 1.5,
        "extra_body": {
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": False}
        }
    },
    "complex_instruct": {
        "temperature": 0.7,
        "top_p": 0.8,
        "presence_penalty": 1.5,
        "extra_body": {
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": False}
        }
    }
}

QWEN35_DEFAULT_MAX_TOKENS = {
    "coding_reasoning": 4096,
    "general_reasoning": 2048,
    "general_instruct": 512,
    "complex_instruct": 1024
}

LLAMA_SERVER_URL = "http://localhost:8080/v1"

MODEL_DIR = os.getenv('MODEL_DIR')

MODEL = f"{MODEL_DIR}\Qwen3.5-9B-Q8_0.gguf"