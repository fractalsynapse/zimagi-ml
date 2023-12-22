from systems.plugins.index import BaseProvider


class Provider(BaseProvider('encoder', 'mpnet_di')):

    @classmethod
    def _get_model_name(cls):
        return 'all-mpnet-base-v2'

    def get_dimension(self):
        return 768
