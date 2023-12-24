from systems.commands.index import Agent


class Qdrant(Agent('qdrant')):

    processes = (
        'qdrant_backup',
        'qdrant_clean'
    )

    def qdrant_backup(self):
        for package in self.listen('core:db:backup:init', state_key = 'qdrant'):
            self.create_snapshot()

    def qdrant_clean(self):
        for package in self.listen('core:db:clean', state_key = 'qdrant'):
            self.clean_snapshots(keep_num = package.message)
