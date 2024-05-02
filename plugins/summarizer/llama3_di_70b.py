from systems.plugins.index import BaseProvider


class Provider(BaseProvider('summarizer', 'llama3_di_70b')):

    @classmethod
    def _get_model_name(cls):
        return 'meta-llama/Meta-Llama-3-70B-Instruct'
