from django.conf import settings
from google.oauth2 import service_account
from google.cloud import aiplatform
from vertexai.preview.language_models import TextGenerationModel

from systems.plugins.index import BaseProvider
from utility.data import load_json


class Provider(BaseProvider('summarizer', 'vertex_ai')):

    @classmethod
    def initialize(cls, instance, init):
        if not getattr(cls, '_model', None):
            cls._model = {}

        if instance.identifier:
            credentials = load_json(settings.GOOGLE_SERVICE_CREDENTIALS)
            account = service_account.Credentials.from_service_account_info(credentials)

            aiplatform.init(
                location = settings.GOOGLE_VERTEX_AI_REGION,
                project = credentials['project_id'],
                staging_bucket = "gs://{}".format(settings.GOOGLE_VERTEX_AI_BUCKET),
                credentials = account
            )
            cls._model[instance.identifier] = TextGenerationModel.from_pretrained('text-bison@001')

    @property
    def model(self):
        return self._model[self.identifier]

    @property
    def tokenizer(self):
        return self._tokenizer[self.identifier]


    def get_chunk_length(self):
        return 8192

    def get_token_count(self, text):
        return len(text) / 10


    def _get_prompt(self, text, prompt = ''):
        return """
Summarize the following text between the !$!>>> and <<<!$! character sequences using everything after the ###+*$*$*+### character sequence as the instructions for summarization.
!$!>>>{text}<<<!$! ###+*$*$*+### {context}
        """.format(
            context = prompt.strip(),
            text = text.strip()
        )


    def summarize(self, text, **config):
        prompt = self._get_prompt(text, config.get('prompt', ''))
        response = self.model.predict(prompt,
            temperature = config.get('temperature', 0.1),
            top_k = config.get('top_k', 5),
            top_p = config.get('top_p', 0.2),
            max_output_tokens = 1024
        )
        print('======================')
        print('======================')
        print(prompt)
        print('======================')
        print(response.text)
        return response.text
