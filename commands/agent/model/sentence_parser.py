from systems.commands.index import Agent

import statistics


class SentenceParser(Agent('model.sentence_parser')):

    def exec(self):
        self.sentence_parser = self.get_sentence_parser()

        channel = 'agent:model:sentence_parser'

        for package in self.listen(channel, state_key = 'model_sentence_parser'):
            text = package.message

            try:
                self.data('Processing sentence parsing request', package.sender)
                response = self.profile(self._parse_model_sentences, text)
                sentence_lengths = [ len(sentence) for sentence in response.result ]

                self.send(package.sender, response.result)

            except Exception as e:
                self.send(channel, package.message, package.sender)
                raise e

            if sentence_lengths:
                self.send("{}:stats".format(channel), {
                    'provider': self.sentence_parser.name,
                    'length': len(text),
                    'sentence_count': len(response.result),
                    'sentence_length_mean': round(statistics.mean(sentence_lengths), 1) if len(sentence_lengths) > 1 else sentence_lengths[0],
                    'sentence_length_sd': round(statistics.stdev(sentence_lengths), 1) if len(sentence_lengths) > 1 else 0,
                    'sentence_length_min': min(sentence_lengths),
                    'sentence_length_max': max(sentence_lengths),
                    'time': response.time,
                    'memory': response.memory
                })

    def _parse_model_sentences(self, text):
        return self.sentence_parser.split(text)
