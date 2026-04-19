import os
import base64
from PIL import Image
from openai import OpenAI

from src.paths import LLAMA_SERVER_URL, QWEN35_9B, QWEN36_35B, GEMMA4_26B

import warnings

VERBOSE = True

os.environ["LLAMA_LOG_DISABLE"] = "1"


def prepare_image(img_path):
    with open(img_path, "rb") as f_:
        img_b64 = base64.b64encode(f_.read()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}


class ImageTextToTextPipeline:
    def __init__(self, server, model):
        self.__server = server
        self.__model = model
        self.__max_images = 3
        self.__thinking = "<|think|>"
        self.__no_thinking = "No thinking. No reasoning."

    def __call__(
        self,
        prompt: list[str] | str,
        max_tokens: int,
        gen_params: dict = {},
        image: list[str] | str | list[list[str]] = None,
        system_prompt: str = "",
        keep_reasoning: bool = False
    ) -> list[str] | list[dict[str, str]]:
        if VERBOSE:
            print("INPUT SYSTEM PROMPT:", system_prompt)
            print("INPUT PROMPT", prompt)
        if image is None:
            outputs = self.batch_text_generation(
                prompt=prompt,
                system_prompt=system_prompt,
                keep_reasoning=keep_reasoning,
                max_tokens=max_tokens,
                gen_params=gen_params
            )
        else:
            outputs = self.batch_image_text_to_text_generation(
                prompt=prompt,
                system_prompt=system_prompt,
                image=image,
                keep_reasoning=keep_reasoning,
                max_tokens=max_tokens,
                gen_params=gen_params
            )
        return outputs
    
    def completion(
        self,
        messages,
        max_tokens: int,
        gen_params: dict = {},
        keep_reasoning: bool = False,
        **kwargs
    ) -> dict[str, str] | str:
        stream = kwargs["stream"] if "stream" in kwargs else False
        response = self.__server.chat.completions.create(
            model=self.__model,
            max_tokens=max_tokens,
            messages=messages,
            stream=stream,
            **gen_params
        )
        print("RESPONSE:", response)
        content = ""
        try:
            reasoning_content = response.choices[0].message.reasoning_content
        except:
            reasoning_content = ""
        if response.choices[0].message.content:
            content = response.choices[0].message.content
        else:
            if reasoning_content and not keep_reasoning:
                content = reasoning_content
        if keep_reasoning:
            return {"reasoning": reasoning_content, "response": content}
        else:
            return content
        
    def batch_text_generation(
        self,
        prompt: list[str] | str,
        max_tokens: int,
        gen_params: dict = {},
        system_prompt: str = "",
        keep_reasoning: bool = False
    ) -> list[str] | list[dict[str, str]]:
        completions: list[str] | list[dict[str, str]] = []

        if type(prompt) is str:
            completion = self.text_generation(
                prompt=prompt,
                max_tokens=max_tokens,
                gen_params=gen_params,
                system_prompt=system_prompt,
                keep_reasoning=keep_reasoning
            )
            completions.append(completion)
        else:
            for p_ in prompt:
                completion = self.text_generation(
                    prompt=p_,
                    max_tokens=max_tokens,
                    gen_params=gen_params,
                    system_prompt=system_prompt,
                    keep_reasoning=keep_reasoning
                )
        return completions
    
    def text_generation(
        self,
        prompt: str,
        max_tokens: int,
        gen_params: dict = {},
        system_prompt: str = "",
        keep_reasoning: bool = False
    ) -> str | dict[str, str]:
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }]
        
        if "gemma-4" in self.__model:
            if self.__is_thinking_active(gen_params):
                system_prompt = self.__thinking + f" {system_prompt}".strip()
            else:
                system_prompt = self.__no_thinking + f" {system_prompt}".strip()
        
        if system_prompt:
            messages = [{
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            }]
        
        try:
            completion: str | dict[str, str] = self.completion(
                messages=messages,
                max_tokens=max_tokens,
                gen_params=gen_params,
                keep_reasoning=keep_reasoning
            )
        except:
            if keep_reasoning:
                completion = {"reasoning": "", "response": "response"}
            else:
                completion = ""
        return completion
    
    def batch_image_text_to_text_generation(
        self,
        prompt: list[str] | str,
        max_tokens: int,
        gen_params: dict = {},
        image: list[str] | str | list[list[str]] = None,
        system_prompt: str = "",
        keep_reasoning: bool = False
    ) -> list[str] | list[dict[str, str]]:
        completions: list[str] | list[dict[str, str]] = []

        if type(prompt) is str and type(image) is str:
            completion = self.image_text_to_text_generation(
                prompt=prompt,
                max_tokens=max_tokens,
                gen_params=gen_params,
                image=image,
                system_prompt=system_prompt,
                keep_reasoning=keep_reasoning
            )
            completions.append(completion)
        else:
            if type(prompt) is str:
                prompt = [prompt]
            if type(image) is str:
                image = [image]
            if len(image) != len(prompt):
                if len(image) == 1:
                    image = image * len(prompt)
                else:
                    raise Exception("Error in pipeline ImageTextToTextPipeline: "
                                    f"len(image) = {len(image)} but len(prompt) = {len(prompt)}")
            for (p_, im) in zip(prompt, image):
                completion = self.image_text_to_text_generation(
                    prompt=p_,
                    max_tokens=max_tokens,
                    gen_params=gen_params,
                    image=im,
                    system_prompt=system_prompt,
                    keep_reasoning=keep_reasoning
                )
                completions.append(completions)
        return completions
    
    def image_text_to_text_generation(
        self,
        prompt: list[str] | str,
        max_tokens: int,
        gen_params: dict = {},
        image: list[str] | str | list[list[str]] = None,
        system_prompt: str = "",
        keep_reasoning: bool = False
    ):
        if len(image) > self.__max_images:
            warnings.warn((
                f"Warning: More images {len(image)} "
                f"than allowed by the model settings {self.__max_images}. "
                f"Only the first {self.__max_images} are retained"
            ))
            image = image[:self.__max_images]

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}
            ] + [
                self.__prepare_image(img_path) for img_path in image
            ]
        }]

        if "gemma-4" in self.__model:
            if self.__is_thinking_active(gen_params):
                system_prompt = self.__thinking + f" {system_prompt}".strip()
            else:
                system_prompt = self.__no_thinking + f" {system_prompt}".strip()

        if system_prompt:
            messages = [{
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            }] + messages
        
        try:
            completion: str | dict[str, str] = self.completion(
                messages=messages,
                max_tokens=max_tokens,
                gen_params=gen_params,
                keep_reasoning=keep_reasoning
            )
        except:
            if keep_reasoning:
                completion = {"reasoning": "", "response": ""}
            else:
                completion = ""
        return completion
    
    @staticmethod
    def __is_thinking_active(gen_params: dict):
        if gen_params and "chat_template_kwargs" in gen_params:
            return gen_params["chat_template_kwargs"]["enable_thinking"]
        else:
            return False



class ModelClient:
    def __init__(self):
        self.__server = OpenAI(base_url=LLAMA_SERVER_URL, api_key="not-needed")
        self.__pipeline = ImageTextToTextPipeline(server=self.__server, model=QWEN36_35B)

    def __call__(
        self,
        prompt: str | list[str], 
        max_tokens: int, 
        **kwargs
    ) -> list[str] | list[dict[str, str]]:
        return self.__pipeline(prompt=prompt, max_tokens=max_tokens, **kwargs)
    
    def completion(
        self,
        messages,
        max_tokens: int,
        gen_params: dict = {},
        keep_reasoning: bool = False,
        **kwargs
    ) -> dict[str, str] | str:
        response = self.__pipeline.completion(
            messages=messages,
            max_tokens=max_tokens,
            gen_params=gen_params,
            keep_reasoning=keep_reasoning,
            **kwargs
        )
        return response
        
    
    def stream(
        self, 
        messages, 
        max_tokens: int,
        gen_params={},
        keep_reasoning: bool = False
    ):
        output = self.__pipeline.completion(
            self,
            messages=messages,
            max_tokens=max_tokens,
            gen_params=gen_params,
            keep_reasoning=keep_reasoning,
            stream=False
        )
        return output


model_client = None


def get_model_client():
    global model_client
    if not isinstance(model_client, ModelClient):
        model_client = ModelClient()
    return model_client


