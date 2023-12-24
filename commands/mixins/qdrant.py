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


    def create_snapshot(self, collection_name = None):
        def _create_snapshot(collection):
            collection.create_snapshot()
            self.success("Qdrant snapshot for {} successfully created".format(collection.name))

        results = self.run_list(
            self.get_qdrant_collections(collection_name),
            _create_snapshot
        )
        if results.aborted:
            self.error('Qdrant snapshot creation failed')

    def remove_snapshot(self, collection_name, snapshot_name):
        collection = self.qdrant(collection_name)
        if not collection.delete_snapshot(snapshot_name):
            self.warning("Qdrant snapshot {} not removed".format(snapshot_name))
        self.success("Qdrant snapshot {} successfully removed".format(snapshot_name))

    def clean_snapshots(self, collection_name = None, keep_num = 3):
        def _clean_snapshots(collection):
            collection.clean_snapshots(keep_num)
            self.success("Qdrant snapshots for {} successfully cleaned".format(collection.name))

        results = self.run_list(
            self.get_qdrant_collections(collection_name),
            _clean_snapshots
        )
        if results.aborted:
            self.error('Qdrant snapshot cleaning failed')

    def restore_snapshot(self, collection_name = None, snapshot_name = None):
        def _restore_snapshot(collection):
            collection.restore_snapshot()
            self.success("Latest Qdrant snapshot for {} successfully restored".format(collection.name))

        if snapshot_name:
            if not collection_name:
                self.error("Collection name required when specifying the snapshot name to restore")

            collection = self.qdrant(collection_name)
            if not collection.restore_snapshot(snapshot_name):
                self.error("Qdrant snapshot {} restore failed".format(snapshot_name))
            self.success("Qdrant snapshot {} successfully restored".format(snapshot_name))
        else:
            results = self.run_list(
                self.get_qdrant_collections(collection_name),
                _restore_snapshot
            )
            if results.aborted:
                self.error('Qdrant snapshot restoration failed')


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