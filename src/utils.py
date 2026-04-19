import streamlit as st
import json
import os
import cryptography.fernet
from openai import OpenAI

from src import prompts
import src.params as _params_
from src.models import get_model_client, prepare_image


from typing import Any, Generator


class SecureChatHistory:
    def __init__(self, file_path="chat_history.txt"):
        self.file_path = file_path
        self.key = cryptography.fernet.Fernet(os.environ.get('FERNET_KEY').encode('ascii'))
        self.__history = {}
        self._load()

    def __getitem__(self, key: str):
        return self.__history[key]
    
    def __setitem__(self, key: str, value: list):
        if "key" not in self.list_conversations:
            self.__history[key] = []

    def _load(self):
        """Decrypt and load history from JSON file."""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'rb') as f_:
                    encrypted_data = f_.read()
                    decrypted_content = self.key.decrypt(encrypted_data)
                    if decrypted_content:
                        self.__history = json.loads(decrypted_content.decode('utf-8'))
                    else:
                        self.__history = {}
            except Exception as e:
                st.error(f"Error loading history: {e}")
                self.__history = {}
        else:
            self.__history = {}

    def __save(self):
        """Encrypt and save history to JSON file."""
        try:
            # Convert history to JSON string
            data_str = json.dumps(self.__history)
            # Encrypt
            encrypted_data = self.key.encrypt(data_str.encode('utf-8'))
            # Save to file
            with open(self.file_path, 'wb') as f_:
                f_.write(encrypted_data)
        except Exception as e:
            st.error(f"Error saving history: {e}")

    def add_message(self, conversation_name: str, message: dict[str: str]):
        if (not len(message) == 2 or
            not "role" in message or
            not "content" in message or
            message["role"] not in ["user", "system", "assistant"] or
            not type(message["content"]) is str
        ):
            raise Exception(f"In SecureChatHistory.add_message, invalid message format: {message}")
        """Add a message to the history."""
        self.__history[conversation_name].append(message)
        self.__save()

    def get_last_message(self, conversation_name: str):
        """Add a message to the history."""
        return self.__history[conversation_name][-1]

    def get_conversation(key: str) -> list[dict[str, str]]:
        return self.__getitem__(key)
    
    def delete_conversation(conversation_name: str):
        if conversation_name in self.__history:
            del self.__history[conversation_name]

    def reset_history():
        self.__history = {}
        self.__save()

    @property
    def history(self) -> dict[str, list[dict[str, str]]]:
        """Return the decrypted history."""
        return self.__history
    
    @property
    def list_conversations(self) -> list[str]:
        """Return the decrypted history."""
        return list(self.__history.keys())


chat_history = None


def get_chat_history():
    global chat_history
    if chat_history is None:
        chat_history = SecureChatHistory()
    return chat_history


class ChatBot:
    def __init__(self, conversation_name: str = ""):
        self.__model_server = get_model_client()
        self.__conversation_name = None
        self.__conversation = None
        self.__chat_history = get_chat_history()
        self.__init_conversation(conversation_name)

    def __init_conversation(self, conversation_name: str):
        if conversation_name in self.__chat_history.list_conversations:
            self.__conversation_name = conversation_name
            self.__conversation = self.__chat_history[self.__conversation_name]
        else:
            self.__conversation_name = ""
    
    def get_response_stream(
            self, 
            query: str,
            max_reasoning_tokens: int,
            max_response_tokens: int,
            params: dict[str, Any],
            image: list[str] | str = None
        ) -> Generator[str, None, None]:
        """
        Generator that yields text word-by-word from the streaming response.
        """
        if type(image) is str:
            image = [image]

        # 1 - first message template with user query
        message = {
            "role": "user",
            "content": [{"type": "text", "text": query}]
        }
        
        # 2 - add images if any passed in
        if type(image) is list:
            message["content"] += [prepare_image(img_path) for img_path in image]

        # 3 - convert into a list, the format taken by completion()
        message = [message]

        # 4 - add system prompt to verify if response if fully generated (reasoning_mode)
        if max_reasoning_tokens:
            system_prompt = """Add <|end|> at the end of the generated answer."""
            message = [{
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            }] + message
   
        if not self.__conversation_name or not type(self.__conversation) is list:
            self.__conversation_name = self.conversation_naming(query)
            self.__chat_history[self.__conversation_name] = []
            self.__conversation = self.__chat_history[self.__conversation_name]

        # images are not saved
        self.__conversation.append({"role": "user", "content": query})
    
        if max_reasoning_tokens:
            response: dict[str, str] = self.__model_server.completion(
                messages=message,
                max_tokens=max_reasoning_tokens,
                gen_params=params,
                keep_reasoning=True
            )
            if not response["response"] or not "<|end|>" in response["response"]:
                system_prompt = """Add <|end|> at the end of the generated answer."""
                new_query = prompts.ANSWER_AFTER_REASONING.replace(
                    "$PLACEHOLDER1$", query
                ).replace(
                    "$PLACEHOLDER2$", response["reasoning"]
                )
                new_message = {
                    "role": "user",
                    "content": [{"type": "text", "text": new_query}]
                }
                if type(image) is list:
                    new_message["content"] += [prepare_image(img_path) for img_path in image]
                new_message = [new_message]
                final_response: str = self.__model_server.completion(
                    messages=new_message,
                    max_tokens=max_response_tokens,
                    gen_params=_params_.MODE_PARAMS["complex_instruct"],
                    keep_reasoning=False
                )
                response = {"reasoning": response["reasoning"], "response": final_response}
        
        else:
            response: dict[str, str] = self.__model_server.completion(
                messages=message,
                max_tokens=max_response_tokens,
                gen_params=params,
                keep_reasoning=True
            )
        return response

    def add_assistant_message(self, ai_response: str):
        if type(self.__conversation) is list and len(self.__conversation) > 0:
            message = {"role": "assistant", "content": ai_response}
            self.__chat_history.add_message(
                conversation_name=self.__conversation_name, 
                message=message
            )

    def conversation_naming(self, query: str) -> str:
        response = self.__model_server(
            prompt=prompts.CONVERSATION_NAMING.replace("$PLACEHOLDER$", query),
            max_tokens=36,
            system_prompt='Your role is to return a short summary and nothing else. Be concise.',
            gen_params=_params_.MODE_PARAMS["general_instruct"]
        )
        print("conversation_naming response:", response)
        return response[0]
    
    @property
    def conversation(self) -> list[dict[str, str]]:
        return self.__conversation
    
    @property
    def conversation_name(self) -> list[dict[str, str]]:
        return self.__conversation_name
