from systems.commands.index import CommandMixin
from utility.data import Collection, ensure_list
from utility.web import WebParser
from utility.temp import temp_dir

import base64


class QdrantCommandMixin(CommandMixin('qdrant')):

    def qdrant(self, name, **options):
        return self.get_provider('qdrant_collection', name, **options)

    def get_qdrant_collections(self, names = None, **options):
        if names:
            return [
                self.qdrant(name, **options)
                for name in ensure_list(names)
            ]
        return [
            self.qdrant(name, **options)
            for name in list(self.manager.index.get_plugin_providers('qdrant_collection').keys())
        ]


    def get_embeddings(self, collection, *ids):
        qdrant = self.qdrant(collection)
        sentences = []
        embeddings = []

        for result in qdrant.get(*ids, fields = 'sentence', include_vectors = True):
            sentences.append(result.payload['sentence'])
            embeddings.append(result.vector)

        return Collection(
            sentences = sentences,
            embeddings = embeddings
        )

    def search_embeddings(self, collection, text, fields = None, limit = 10, filter_field = None, filter_ids = None):
        qdrant = self.qdrant(collection)
        text_info = self.generate_text_embeddings(text) if isinstance(text, str) else text
        if text_info is None:
            return []

        if fields is None:
            fields = []
        if filter_field:
            fields = [ filter_field, *fields ]

        options = {
            'limit': limit,
            'fields': [
                *fields,
                'sentence'
            ]
        }
        if filter_field and filter_ids:
            options['filter_field'] = filter_field
            options['filter_values'] = ensure_list(filter_ids)

        return qdrant.search(text_info.embeddings, **options)


    def parse_file_text(self, portal, api_doc_type, file_id):
        text = ''
        with temp_dir() as temp:
            file = portal.retrieve(api_doc_type, file_id)
            file_type = file['file'].split('.')[-1].lower()
            file_path = temp.save(base64.b64decode(file['content']), binary = True)
            try:
                parser = self.manager.get_provider('file_parser', file_type)
                file_text = parser.parse(file_path)
                if file_text:
                    text = file_text

            except ProviderError as e:
                pass
        return text

    def parse_web_text(self, webpage_url):
        text = ''
        parser = WebParser(webpage_url)
        if parser.text:
            text = parser.text
        return text
