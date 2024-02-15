from systems.commands.index import Agent


class Summarizer(Agent('model.summarizer')):

    def exec(self):
        self.summarizer = self.get_summarizer()

        channel = 'agent:model:summary'

        for package in self.listen(channel, state_key = 'model_summarizer'):
            text = package.message['text']
            config = package.message.get('config', {})

            try:
                self.data('Processing summary request', package.sender)
                response = self.profile(self._parse_model_summary, text, config)
                self.send(package.sender, response.result)

            except Exception as e:
                self.send(channel, package.message, package.sender)
                raise e

            self.send("{}:stats".format(channel), {
                'provider': self.summarizer.name,
                'time': response.time,
                'memory': response.memory,
                'length': len(text)
            })

    def _parse_model_summary(self, text, config):
        return self.summarizer.summarize(text, **config)
