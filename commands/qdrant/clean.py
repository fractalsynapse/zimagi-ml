from systems.commands.index import Command


class Clean(Command('qdrant.clean')):

    def exec(self):
        results = self.run_list(
            self.get_qdrant_collections(self.collection_name),
            self._clean_snapshots
        )
        if results.aborted:
            self.error('Qdrant snapshot cleaning failed')

    def _clean_snapshots(self, collection):
        collection.clean_snapshots(self.keep_num)
        self.success("Qdrant snapshots for {} successfully cleaned".format(collection.name))
