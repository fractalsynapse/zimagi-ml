from django.conf import settings

from systems.plugins.index import BaseProvider

import re
import torch


class Provider(BaseProvider('summarizer', 'llama')):

    @classmethod
    def _get_pipeline(cls, instance):
        from transformers import pipeline
        return pipeline('text-generation',
            model = cls._get_model_name(),
            torch_dtype = torch.float16,
            device = instance.field_device,
            use_auth_token = settings.HUGGINGFACE_TOKEN
        )


    def get_chunk_length(self):
        return 3500


    def summarize(self, text, **config):
        prompt = self._get_prompt(text,
            prompt = config.get('prompt', ''),
            persona = config.get('persona', ''),
            output_format = config.get('format', '')
        )
        results = self.pipeline(
            prompt,
            do_sample = True,
            temperature = config.get('temperature', 0.1),
            top_k = config.get('top_k', 5),
            top_p = config.get('top_p', 0.9),
            num_return_sequences = 1,
            eos_token_id = self.tokenizer.eos_token_id,
            max_length = 4096,
        )
        return self._parse_summary_response(
            results[0]['generated_text'].split('[/INST]')[1].strip()
        )
