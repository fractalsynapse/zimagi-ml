from systems.plugins.index import BaseProvider


class Provider(BaseProvider('summarizer', 'mixtral_di_22bx8')):

    @classmethod
    def _get_model_name(cls):
        return 'HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1'
