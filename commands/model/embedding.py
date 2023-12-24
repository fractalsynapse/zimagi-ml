from systems.commands.index import Command


class Embedding(Command('model.embedding')):

  def exec(self):
    self.data('Sentence Embeddings',
        self.submit('agent:model:embedding', self.sentences),
        'embeddings'
    )