from systems.plugins.index import BaseProvider


class Provider(BaseProvider("summarizer", "llama3_di_70b")):

    @classmethod
    def _get_model_name(cls):
        return "meta-llama/Meta-Llama-3.1-70B-Instruct"

    def get_max_context(self):
        return 6000

    def get_chunk_length(self):
        return 5000

    def get_max_new_tokens(self):
        return 2000
