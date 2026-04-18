import streamlit as st
import json
import os
import cryptography.fernet
from openai import OpenAI
import pynvml

from src import prompts
import src.params as _params_

from typing import Any, Generator



class SecureChatHistory:
    def __init__(self, file_path="chat_history.json"):
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
                    print("decrypted_content", decrypted_content)
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


model_client = None


def get_model_client():
    global model_client
    if model_client is None:
        model_client = OpenAI(base_url=_params_.LLAMA_SERVER_URL, api_key="not-needed")
    return model_client


class ChatBot:
    def __init__(self, conversation_name: str = ""):
        self.__client = get_model_client()
        self.__model = _params_.MODEL
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
            max_tokens: int,
            params: dict[str, Any]
        ) -> Generator[str, None, None]:
        """
        Generator that yields text word-by-word from the streaming response.
        """
        message = {'role': 'user', 'content': query}

        if not self.__conversation_name or not type(self.__conversation) is list:
            self.__conversation_name = self.conversation_naming(query)
            self.__chat_history[self.__conversation_name] = []
            self.__conversation = self.__chat_history[self.__conversation_name]

        # self.__chat_history.add_message(self.__conversation_name, message)
        self.__conversation.append(message)
            
        try:
            response = self.__client.chat.completions.create(
                model=self.__model,
                max_tokens=max_tokens,
                messages=[message],
                stream=True,
                **params
            )
            full_response = ""
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    text_chunk = chunk.choices[0].delta.content
                    full_response += text_chunk
            print("FULL RESPONSE:", full_response)
            return full_response
                    # Note: Splitting by whitespace might break some markdown. Splitlines instead.
                    # for sentence in full_response.splitlines():
                    #     yield sentence
        except Exception as e:
            # yield f"[Error: {str(e)}]"
            return f"[Error: {str(e)}]"

    def add_assistant_message(self, ai_response: str):
        if type(self.__conversation) is list and len(self.__conversation) > 0:
            message = {"role": "assistant", "content": ai_response}
            self.__chat_history.add_message(
                conversation_name=self.__conversation_name, 
                message=message
            )

    def conversation_naming(self, query: str) -> str:
        params = _params_.QWEN35_PARAMS["general_instruct"]
        summary_message = [
            {
                'role': 'system',
                'content': [{
                    'type': 'text',
                    'text': 'Your role is to return a short summary. No thinking. No reasoning.'
                }]
            },
            {
                'role': 'user', 
                'content': [{
                    'type': 'text',
                    'text': prompts.CONVERSATION_NAMING.replace("$PLACEHOLDER$", query)
                }]
            }
        ]
        response = self.__client.chat.completions.create(
                model=self.__model,
                max_tokens=36,
                messages=summary_message,
                stream=False,
                **params
            )
        print("summary conv naming: ", response)
        return response.choices[0].message.content
    
    @property
    def conversation(self) -> list[dict[str, str]]:
        return self.__conversation
    
    @property
    def conversation_name(self) -> list[dict[str, str]]:
        return self.__conversation_name
