from systems.commands.index import Command
from utility.data import load_json


class Summary(Command('model.summary')):

  def exec(self):
    self.data('Text Summary',
        self.submit('agent:model:summary', {
            'text': self.text,
            'config': load_json(self.model_config) if isinstance(self.model_config, str) else self.model_config
        }),
        'summary'
    )
