from systems.commands.index import Command


class Restore(Command('qdrant.restore')):

    def exec(self):
        if self.snapshot_name:
            if not self.collection_name:
                self.error("Collection name required when specifying the snapshot name to restore")

            collection = self.qdrant(self.collection_name)
            if not collection.restore_snapshot(self.snapshot_name):
                self.error("Qdrant snapshot {} restore failed".format(self.snapshot_name))
            self.success("Qdrant snapshot {} successfully restored".format(self.snapshot_name))
        else:
            results = self.run_list(
                self.get_qdrant_collections(self.collection_name),
                self._restore_snapshot
            )
            if results.aborted:
                self.error('Qdrant snapshot restoration failed')

    def _restore_snapshot(self, collection):
        collection.restore_snapshot()
        self.success("Latest Qdrant snapshot for {} successfully restored".format(collection.name))
