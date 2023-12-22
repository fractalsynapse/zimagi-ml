from systems.plugins.index import BaseProvider


class Provider(BaseProvider('summarizer', 'llama2_7b')):

    @classmethod
    def _get_model_name(cls):
        return 'meta-llama/Llama-2-7b-chat-hf'
