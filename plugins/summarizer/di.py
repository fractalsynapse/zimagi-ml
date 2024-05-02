from django.conf import settings

from systems.plugins.index import BaseProvider
from utility.data import load_json

import os
import math
import requests


class DeepInfraRequestError(Exception):
    pass


class Provider(BaseProvider('summarizer', 'di')):

    @classmethod
    def initialize(cls, instance, init):
        if not getattr(cls, '_tokenizer', None):
            cls._tokenizer = {}

        if instance.identifier not in cls._tokenizer:
            os.environ['TRANSFORMERS_CACHE'] = settings.MANAGER.tr_model_cache
            os.environ['HF_HOME'] = settings.MANAGER.hf_cache

            cls._tokenizer[instance.identifier] = cls._get_tokenizer(instance)

    @classmethod
    def _get_tokenizer(cls, instance):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(
            cls._get_model_name(),
            use_auth_token = settings.HUGGINGFACE_TOKEN
        )


    def get_chunk_length(self):
        raise NotImplementedError("Method get_chunk_length() must be implemented in DeepInfra plugin providers.")

    def get_max_new_tokens(self):
        raise NotImplementedError("Method get_max_new_tokens() must be implemented in DeepInfra plugin providers.")


    def _run_inference(self, **config):
        response = requests.post(
            "https://api.deepinfra.com/v1/inference/{}".format(self._get_model_name()),
            headers = {
                'Authorization': "bearer {}".format(settings.DEEPINFRA_API_KEY),
                'Content-Type': 'application/json'
            },
            timeout = 600,
            json = config
        )
        response_data = load_json(response.text)

        if response.status_code == 200 and response_data['inference_status'].get('tokens_generated', None):
            return response_data['results']
        else:
            raise DeepInfraRequestError("DeepInfra inference request failed with code {}: {}".format(
                response.status_code,
                response_data
            ))


    def summarize(self, text, **config):
        prompt = self._get_prompt(text,
            prompt = config.get('prompt', ''),
            persona = config.get('persona', ''),
            output_format = config.get('format', '')
        )
        results = self._run_inference(
            input = prompt,
            max_new_tokens = self.get_max_new_tokens(),
            temperature = config.get('temperature', 0.1), # 0.01 - 100
            top_p = config.get('top_p', 0.9), # 0 - 1
            repetition_penalty = config.get('repetition_penalty', 0.9) # 0.01 - 5
        )
        return self._parse_summary_response(
            results[0]['generated_text'].strip()
        )
