import os

MODE_PARAMS = {
    "coding_reasoning": {
        "temperature": 0.6,
        "top_p": 0.95,
        "presence_penalty": 0.0,
        "extra_body": {
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": True},
            "presence_penalty": 0.0
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
            "chat_template_kwargs": {"enable_thinking": True},
            "presence_penalty": 1.5
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
            "chat_template_kwargs": {"enable_thinking": False},
            "presence_penalty": 1.5
        }
    },
    "complex_instruct": {
        "temperature": 1.0,
        "top_p": 0.95,
        "presence_penalty": 1.5,
        "extra_body": {
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": False},
            "presence_penalty": 1.5
        }
    }
}

DEFAULT_MAX_TOKENS = {
    "coding_reasoning": {
        "reasoning": 2048,
        "response": 2048
    },
    "general_reasoning": {
        "reasoning": 1024,
        "response": 1024
    },
    "general_instruct": {
        "reasoning": 0,
        "response": 512
    },
    "complex_instruct": {
        "reasoning": 0,
        "response": 512
    }
}