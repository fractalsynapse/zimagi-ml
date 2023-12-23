from systems.commands.index import Command


class Snapshot(Command('qdrant.snapshot')):

    def exec(self):
        results = self.run_list(
            self.get_qdrant_collections(self.collection_name),
            self._create_snapshot
        )
        if results.aborted:
            self.error('Qdrant snapshot creation failed')

    def _create_snapshot(self, collection):
        collection.create_snapshot()
        self.success("Qdrant snapshot for {} successfully created".format(collection.name))
