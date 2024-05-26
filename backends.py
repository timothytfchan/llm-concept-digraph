"""
Language model backends.

Mostly copied from https://github.com/jprivera44/EscalAItion/, https://github.com/mukobi/welfare-diplomacy
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

from abc import ABC, abstractmethod
import time
from typing import List, Dict, Optional
from dataclasses import dataclass

# Ensure necessary imports are present; remove or update imports as needed
import google.generativeai as genai
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from anthropic._exceptions import APIError
import backoff
MAX_BACKOFF_TIME_DEFAULT = 600

# Load environment variables from .env file
load_dotenv()

from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
)
import torch

import together

def get_backend(model_name):
    if "gpt" in model_name:
        from backends import OpenAIBackend
        backend = OpenAIBackend(model_name)
        return backend
    elif "claude" in model_name:
        from backends import ClaudeCompletionBackend
        backend = ClaudeCompletionBackend(model_name)
        return backend
    elif "gemini" in model_name:
        from backends import GeminiBackend
        backend = GeminiBackend(model_name)
        return backend
    elif "llama" in model_name:
        from backends import LlamaBackend
        backend = LlamaBackend(model_name)
        return backend
    elif "mistral" in model_name:
        from backends import MistralBackend
        backend = MistralBackend(model_name)
        return backend
    else:
        raise ValueError(f"Model {model_name} not supported or not found.")

@dataclass
class BackendResponse:
    """Response data returned from a model."""

    completion: str
    completion_time_sec: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class LanguageModelBackend(ABC):
    """Abstract class for language model backends."""

    @abstractmethod
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        completion_preface: str = "",
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> BackendResponse:
        """
        Complete a prompt with the language model backend.

        Returns the completion as a string.
        """
        raise NotImplementedError("Subclass must implement abstract method")

class HuggingFaceCausalLMBackend(LanguageModelBackend):
    """HuggingFace chat completion backend (e.g. Llama2, Mistral, MPT)."""

    def __init__(
        self,
        model_name,
        local_llm_path,
        device,
        quantization,
        fourbit_16b_compute: bool,
        rope_scaling_dynamic: float,
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.max_tokens = 2000

        if quantization == 4:
            fourbit = True
            eightbit = False
        elif quantization == 8:
            eightbit = True
            fourbit = False
        else:
            fourbit = False
            eightbit = False
        if fourbit_16b_compute:
            bnb_4bit_compute_dtype = torch.bfloat16
        else:
            bnb_4bit_compute_dtype = torch.float32

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=fourbit,
            load_in_8bit=eightbit,
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
        )
        model_path = self.model_name
        if local_llm_path is not None:
            model_path = f"{local_llm_path}/{self.model_name}"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        rope_scaling = None
        if rope_scaling_dynamic > 1.0:
            rope_scaling = {"type": "dynamic", "factor": rope_scaling_dynamic}
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map=self.device,
            rope_scaling=rope_scaling,
        )

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        completion_preface: str = "",
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> BackendResponse:
        prompt = (
            f"<s>[INST] <<SYS>>{system_prompt}<</SYS>> {user_prompt} [/INST]"
            + completion_preface
        )
        start_time = time.time()

        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt", return_length=True)
            estimated_tokens = inputs.length.item()

            # Generate
            generate_ids = self.model.generate(
                inputs.input_ids.to(self.model.device),
                max_new_tokens=self.max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
            )
            output_ids = generate_ids.cpu()[:, inputs.input_ids.shape[1] :]
            completion = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )[0]
            completion_time_sec = time.time() - start_time
            return BackendResponse(
                completion=completion,
                completion_time_sec=completion_time_sec,
                prompt_tokens=estimated_tokens,
                completion_tokens=self.max_tokens,
                total_tokens=estimated_tokens,
            )

class OpenAIBackend(LanguageModelBackend):
    """OpenAI backend compatible with the OpenAI library version 1.0.0 or later."""

    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.max_tokens = 1000  # Adjust based on your needs
        # No longer instantiate OpenAI() as OpenAI client is directly used from openai package
          # Ensure the API key is loaded from .env
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))   
        

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = None  # You can customize max tokens or other parameters as needed
    ) -> BackendResponse:
        try:
            # Adjust the structure according to the new API requirements
            response = self.openai.chat.completions.create(model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},  # Adjust as needed
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens or self.max_tokens)  # Use specific max_tokens or default)

            # Extracting the completion text from the response
            completion = response.choices[0].message.content

            # Note: Adjust the following as necessary since the new API might not return these directly
            completion_time_sec = None  # Example placeholder, set to actual value if available

            # Create and return a BackendResponse object
            return BackendResponse(
                completion=completion,
                completion_time_sec=completion_time_sec,
                prompt_tokens=len(system_prompt.split() + user_prompt.split()),
                completion_tokens=len(completion.split()),
                total_tokens=None  # Set based on your requirements or calculations
            )

        except Exception as e:
            print(f"Error completing prompt with OpenAI: {str(e)}")
            raise



    def completions_with_backoff(self, **kwargs):
        """Exponential backoff for OpenAI API rate limit errors."""
        response = self.openai.chat.completions.create(**kwargs)
        assert response is not None, "OpenAI response is None"
        return response



class ClaudeCompletionBackend:
    """Claude completion backend (e.g. claude-2)."""
    def __init__(self, model_name):
        Anthropic_API = os.getenv('ANTHROPIC_API_KEY')
        self.anthropic = Anthropic(api_key=Anthropic_API)
        self.model_name = model_name
        self.max_tokens = 1000

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        completion_preface: str = "",
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> BackendResponse:
        if self.model_name.startswith("claude-3"):
            message = self.anthropic.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            completion = message.content
            estimated_prompt_tokens = self.anthropic.count_tokens(system_prompt + user_prompt)
            estimated_completion_tokens = self.anthropic.count_tokens(completion)
        else:
            prompt = (
                f"{HUMAN_PROMPT} {system_prompt}\n\n{user_prompt}{AI_PROMPT}"
                + completion_preface
            )
            estimated_prompt_tokens = self.anthropic.count_tokens(prompt)

            start_time = time.time()
            completion = self.completion_with_backoff(
                model=self.model_name,
                max_tokens_to_sample=self.max_tokens,
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
            )
            estimated_completion_tokens = int(len(completion.completion.split()) * 4 / 3)
            completion_time_sec = time.time() - start_time

        return BackendResponse(
            completion=completion,
            completion_time_sec=completion_time_sec,
            prompt_tokens=estimated_prompt_tokens,
            completion_tokens=estimated_completion_tokens,
            total_tokens=estimated_prompt_tokens + estimated_completion_tokens,
        )

    @backoff.on_exception(
        backoff.expo, APIError, max_time=MAX_BACKOFF_TIME_DEFAULT
    )
    def completion_with_backoff(self, **kwargs):
        """Exponential backoff for Claude API errors."""
        response = self.anthropic.completions.create(**kwargs)
        assert response is not None, "Anthropic response is None"
        return response

class GeminiBackend:
    """Gemini completion backend (e.g. gemini-pro)."""

    def __init__(self, model_name):
        generation_config = {
            "temperature": 1.0,
            "top_p": 1.0,
            "max_output_tokens": 1000,
        }

        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ]

        GOOGLE_API = os.getenv('GOOGLE_API_KEY')
        genai.configure(api_key=GOOGLE_API) #####please add your gemini key
        self.gemini= genai.GenerativeModel(model_name="gemini-pro",
                              generation_config=generation_config,
							  safety_settings=safety_settings)	
        self.model_name = model_name
        self.max_tokens = 1000

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        completion_preface: str = "",
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> BackendResponse:
        try:
            start_time = time.time()
            response = self.completions_with_backoff(
                contents=[
                    {"role": "user", "parts": [f'{system_prompt}\n\n{user_prompt}']},
                ]
            )
            completion = response.text  # type: ignore
            completion_time_sec = time.time() - start_time
            estimated_completion_tokens = int(len(completion.split()) * 4 / 3)
            estimated_prompt_tokens = int(len(system_prompt.split()+user_prompt.split()) * 4 / 3)
            
            return BackendResponse(
                completion=completion,
                completion_time_sec=completion_time_sec,
                prompt_tokens=estimated_prompt_tokens,
                completion_tokens=estimated_completion_tokens,
                total_tokens=estimated_prompt_tokens+estimated_completion_tokens,
            )

        except Exception as exc:  # pylint: disable=broad-except
            print(
                "Error completing prompt ending in\n%s\n\nException:\n%s",
                user_prompt[-300:],
                exc,
            )
            raise

    #@backoff.on_exception(
    #    backoff.expo, APIError, max_time=MAX_BACKOFF_TIME_DEFAULT
    #)
    def completions_with_backoff(self, **kwargs):
        response = self.gemini.generate_content(**kwargs)
        assert response is not None, "Gemini response is None"
        return response

class LlamaBackend:
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.max_tokens = 1000
        TOGETHER_API = os.getenv('TOGETHER_API_KEY')
        together.api_key = TOGETHER_API
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        completion_preface: str = "",
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> BackendResponse:
        prompt = (
            f"[INST] <<SYS>>{system_prompt}<</SYS>> {user_prompt} [/INST]"
            + completion_preface
        )
        start_time = time.time()
        completion = self.completion_with_backoff(
            model=self.model_name,
            max_tokens=self.max_tokens,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            stop=[
                    "[/INST]",
                    "</s>"
                ]
        )
        completion_time_sec = time.time() - start_time
        estimated_completion_tokens = int(len(completion.split()) * 4 / 3)
        estimated_prompt_tokens = int(len(system_prompt.split()+user_prompt.split()) * 4 / 3)
        
        return BackendResponse(
            completion=completion,
            completion_time_sec=completion_time_sec,
            prompt_tokens=estimated_prompt_tokens,
            completion_tokens=estimated_completion_tokens,
            total_tokens=estimated_prompt_tokens + estimated_completion_tokens,
        )

    def completion_with_backoff(self, **kwargs):
        """Exponential backoff for Together API errors."""
        response = together.Complete.create(**kwargs)
        assert response is not None, "Llama response is None"
        return response.output.choices[0].text
    
class MistralBackend:
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.max_tokens = 1000
        TOGETHER_API = os.getenv('TOGETHER_API_KEY')
        together.api_key = TOGETHER_API
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        completion_preface: str = "",
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> BackendResponse:
        prompt = (
            f"[INST] {system_prompt}\n\n{user_prompt} [/INST]"
            + completion_preface
        )
        start_time = time.time()
        completion = self.completion_with_backoff(
            model=self.model_name,
            max_tokens=self.max_tokens,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            stop=[
                    "[/INST]",
                    "</s>"
                ]
        )
        completion_time_sec = time.time() - start_time
        estimated_completion_tokens = int(len(completion.split()) * 4 / 3)
        estimated_prompt_tokens = int(len(system_prompt.split()+user_prompt.split()) * 4 / 3)
        
        return BackendResponse(
            completion=completion,
            completion_time_sec=completion_time_sec,
            prompt_tokens=estimated_prompt_tokens,
            completion_tokens=estimated_completion_tokens,
            total_tokens=estimated_prompt_tokens + estimated_completion_tokens,
        )

    def completion_with_backoff(self, **kwargs):
        """Exponential backoff for Together API errors."""
        response = together.Complete.create(**kwargs)
        assert response is not None, "Mistral response is None"
        return response.output.choices[0].text    
    
