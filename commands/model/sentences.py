from systems.commands.index import Command


class Sentences(Command('model.sentences')):

  def exec(self):
    self.data('Sentences',
        self.submit('agent:model:sentence_parser', {
            'text': self.text,
            'config': self.model_config
        }),
        'sentences'
    )
