from systems.commands.index import Agent

import statistics


class Encoder(Agent('model.encoder')):

    def exec(self):
        self.encoder = self.get_encoder()

        channel = 'agent:model:embedding'

        for package in self.listen(channel, state_key = 'model_encoder'):
            sentences = package.message
            sentence_lengths = [ len(sentence) for sentence in sentences ]

            try:
                self.data('Processing embedding request', package.sender)
                response = self.profile(self._parse_model_embeddings, sentences)

                self.send(package.sender, response.result)

            except Exception as e:
                self.send(channel, package.message, package.sender)
                raise e

            if sentence_lengths:
                self.send("{}:stats".format(channel), {
                    'provider': self.encoder.name,
                    'count': len(sentence_lengths),
                    'length_mean': round(statistics.mean(sentence_lengths), 1) if len(sentences) > 1 else sentence_lengths[0],
                    'length_sd': round(statistics.stdev(sentence_lengths), 1) if len(sentences) > 1 else 0,
                    'length_min': min(sentence_lengths),
                    'length_max': max(sentence_lengths),
                    'time': response.time,
                    'memory': response.memory
                })

    def _parse_model_embeddings(self, sentences):
        return self.encoder.encode(sentences)
