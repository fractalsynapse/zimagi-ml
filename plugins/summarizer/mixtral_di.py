from systems.plugins.index import BaseProvider


class Provider(BaseProvider('summarizer', 'mixtral_di')):

    def get_chunk_length(self):
        return 23000

    def get_max_new_tokens(self):
        return 2000
