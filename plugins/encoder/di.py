from django.conf import settings

from systems.plugins.index import BaseProvider
from utility.data import load_json

import requests


class DeepInfraRequestError(Exception):
    pass


class Provider(BaseProvider('encoder', 'di')):

    @classmethod
    def initialize(cls, instance, init):
        pass


    def _run_inference(self, **config):
        response = requests.post(
            "https://api.deepinfra.com/v1/inference/sentence-transformers/{}".format(self._get_model_name()),
            headers = {
                'Authorization': "bearer {}".format(settings.DEEPINFRA_API_KEY),
                'Content-Type': 'application/json'
            },
            timeout = 600,
            json = config
        )
        response_data = load_json(response.text)

        if response.status_code == 200 and response_data['embeddings']:
            return response_data['embeddings']
        else:
            raise DeepInfraRequestError("DeepInfra inference request failed with code {}: {}".format(
                response.status_code,
                response_data
            ))


    def encode(self, sentences, **config):
        if not sentences:
            return []
        return self._run_inference(
            inputs = sentences
        )
