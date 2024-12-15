from django.conf import settings

from plugins.summarizer.base import SummaryResult
from systems.plugins.index import BaseProvider
from utility.data import ensure_list, load_json
from utility.runtime import Runtime

import os
import math
import requests
import re


class DeepInfraRequestError(Exception):
    pass


class Provider(BaseProvider("summarizer", "di")):

    @classmethod
    def initialize(cls, instance, init):
        if not getattr(cls, "_tokenizer", None):
            cls._tokenizer = {}

        if instance.identifier not in cls._tokenizer:
            os.environ["HF_HOME"] = settings.MANAGER.hf_cache

            cls._tokenizer[instance.identifier] = cls._get_tokenizer(instance)

    @classmethod
    def _get_tokenizer(cls, instance):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            cls._get_model_name(), token=settings.HUGGINGFACE_TOKEN
        )

    def get_max_new_tokens(self):
        raise NotImplementedError(
            "Method get_max_new_tokens() must be implemented in DeepInfra plugin providers."
        )

    def implements_system_prompt(self):
        # Override in sub providers if needed
        return True

    def _get_prompt(self, text="", prompt="", persona="", output_format=""):
        max_context = self.get_max_context()
        prompt_tokens = self.get_token_count(prompt)
        sections = []
        messages = []
        format_messages = []

        if output_format:
            format_instruction = "Render all responses according to the following output format instructions: {}".format(
                output_format.strip()
            )
            format_response = "I will render all following responses according to the output format instructions provided."
            format_messages = [
                {
                    "role": "user",
                    "content": format_instruction,
                },
                {
                    "role": "assistant",
                    "content": format_response,
                },
            ]
            prompt_tokens += self.get_token_count(
                format_instruction
            ) + self.get_token_count(format_response)

        if not persona:
            persona = "You always produce factually correct information from any information given and you do not ask questions."

        system_prompt = "Use the following description as a persona for all instructions and questions: {}".format(
            persona.strip()
        )
        prompt_tokens += self.get_token_count(system_prompt)

        if self.implements_system_prompt():
            messages.append({"role": "system", "content": system_prompt})
        else:
            system_response = "I will use the provided persona when responding to instructions and answering questions."
            messages.extend(
                [
                    {"role": "user", "content": system_prompt},
                    {
                        "role": "assistant",
                        "content": system_response,
                    },
                ]
            )
            prompt_tokens += self.get_token_count(system_response)

        if text:
            for section in reversed(ensure_list(text)):
                if isinstance(section, dict):
                    temp_prompt_tokens = self.get_token_count(section["content"])

                    if (temp_prompt_tokens + prompt_tokens) > max_context:
                        break
                    else:
                        sections.append(section)
                        prompt_tokens += temp_prompt_tokens
                else:
                    section_prompt = "Reference the following text passage for processing instructions and answering questions: {}".format(
                        section.strip()
                    )
                    section_response = "I will reference the provided information when given instructions and answering questions."

                    temp_prompt_tokens = self.get_token_count(
                        section_prompt
                    ) + self.get_token_count(section_response)

                    if (temp_prompt_tokens + prompt_tokens) > max_context:
                        break
                    else:
                        sections.extend(
                            [
                                {
                                    "role": "user",
                                    "content": section_prompt,
                                },
                                {
                                    "role": "assistant",
                                    "content": section_response,
                                },
                            ]
                        )
                        prompt_tokens += temp_prompt_tokens

        messages.extend(reversed(sections))

        if output_format:
            messages.extend(format_messages)

        messages.append({"role": "user", "content": prompt})
        return messages

    def get_prompt_token_count(self, prompt="", persona="", output_format=""):
        token_count = 0
        token_padding = 10

        for message in ensure_list(
            self._get_prompt("", prompt, persona, output_format)
        ):
            if isinstance(message, dict):
                token_count += self.get_token_count(message["content"]) + token_padding
            else:
                token_count += self.get_token_count(message) + token_padding

        return token_count

    def _run_inference(self, messages, **config):
        response = requests.post(
            "https://api.deepinfra.com/v1/openai/chat/completions",
            headers={
                "Authorization": "Bearer {}".format(settings.DEEPINFRA_API_KEY),
                "Content-Type": "application/json",
            },
            timeout=600,
            json={**config, "messages": messages, "model": self._get_model_name()},
        )
        response_data = load_json(response.text)

        if response.status_code == 200 and len(response_data["choices"]):
            return response_data
        else:
            raise DeepInfraRequestError(
                "DeepInfra inference request failed with code {}: {}".format(
                    response.status_code, response_data
                )
            )

    def summarize(self, text="", prompt="", persona="", output_format="", **config):
        model_name = self._get_model_name()
        messages = self._get_prompt(
            text, prompt=prompt, persona=persona, output_format=output_format
        )
        if self.command.debug:
            self.command.data("DeepInfra {} prompt".format(model_name), messages)

        if re.search(r"\s*(JSON|json)\s*", output_format):
            config["response_format"] = {"type": "json_object"}

        results = self._run_inference(
            messages, max_tokens=self.get_max_new_tokens(), **config
        )
        if self.command.debug:
            self.command.data("DeepInfra {} results".format(model_name), results)

        return SummaryResult(
            text=results["choices"][0]["message"]["content"].strip(),
            prompt_tokens=results["usage"]["prompt_tokens"],
            output_tokens=results["usage"]["completion_tokens"],
            cost=results["usage"]["estimated_cost"],
        )
