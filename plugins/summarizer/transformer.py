from django.conf import settings

from systems.plugins.index import BaseProvider
from utility.data import get_identifier

import os


class Provider(BaseProvider('summarizer', 'transformer')):

    @classmethod
    def initialize(cls, instance, init):
        if not getattr(cls, '_tokenizer', None):
            cls._tokenizer = {}
        if not getattr(cls, '_pipeline', None):
            cls._pipeline = {}

        if instance.identifier not in cls._pipeline:
            os.environ['TRANSFORMERS_CACHE'] = settings.MANAGER.tr_model_cache
            os.environ['HF_HOME'] = settings.MANAGER.hf_cache

            cls._tokenizer[instance.identifier] = cls._get_tokenizer(instance)
            if init:
                cls._pipeline[instance.identifier] = cls._get_pipeline(instance)


    @classmethod
    def _get_model_name(cls):
        raise NotImplementedError("Method _get_model_name() must be implemented in sub classes")


    @classmethod
    def _get_tokenizer(cls, instance):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(
            cls._get_model_name(),
            use_auth_token = settings.HUGGINGFACE_TOKEN
        )

    @classmethod
    def _get_pipeline(cls, instance):
        from transformers import pipeline
        return pipeline('summarization',
            model = cls._get_model_name(),
            device = instance.field_device,
            use_auth_token = settings.HUGGINGFACE_TOKEN
        )

    def _get_identifier(self, init):
        return get_identifier([ super()._get_identifier(init), self._get_model_name() ])


    @property
    def pipeline(self):
        return self._pipeline[self.identifier]

    @property
    def tokenizer(self):
        return self._tokenizer[self.identifier]

    def get_token_count(self, text):
        return len(self.tokenizer(text)['input_ids']) if text.strip() else 0

    def get_chunk_length(self):
        return self.tokenizer.model_max_length
