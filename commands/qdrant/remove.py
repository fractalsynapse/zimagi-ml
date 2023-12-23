from systems.commands.index import Command


class Remove(Command('qdrant.remove')):

    def exec(self):
        collection = self.qdrant(self.collection_name)
        if not collection.delete_snapshot(self.snapshot_name):
            self.warning("Qdrant snapshot {} not removed".format(self.snapshot_name))
        self.success("Qdrant snapshot {} successfully removed".format(self.snapshot_name))
