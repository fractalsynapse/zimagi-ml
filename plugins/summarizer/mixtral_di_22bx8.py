from systems.plugins.index import BaseProvider


class Provider(BaseProvider("summarizer", "mixtral_di_22bx8")):

    @classmethod
    def _get_model_name(cls):
        return "HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1"

    def get_max_context(self):
        return 30000

    def get_chunk_length(self):
        return 24000

    def get_max_new_tokens(self):
        return 2000
