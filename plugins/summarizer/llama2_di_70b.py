from systems.plugins.index import BaseProvider


class Provider(BaseProvider('summarizer', 'llama2_di_70b')):

    @classmethod
    def _get_model_name(cls):
        return 'meta-llama/Llama-2-70b-chat-hf'
