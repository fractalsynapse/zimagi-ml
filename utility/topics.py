from spacy.lang.en import stop_words

import re
import spacy


class TopicModel(object):

    text_max_length = 100000


    def __init__(self):
        self.spacy = spacy.load('en_core_web_lg')
        self.spacy.max_length = self.text_max_length


    def filtered_index(self, full_texts, context_texts):
        context_index = self.get_index(*context_texts)
        index = {}

        for topic, count in self.get_index(*full_texts).items():
            index[topic] = count

        return index


    def get_index(self, *texts):
        index = {}
        for text in texts:
            for topic in self.parse(text.strip()):
                if topic not in index:
                    index[topic] = 1
                else:
                    index[topic] += 1
        return { topic: count for topic, count in sorted(index.items(), key = lambda item: item[1], reverse = True) }


    def get_singular(self, text):
        parser = self.spacy(text)
        singular = []

        for token in parser:
            if not token.is_punct:
                singular.append(str(token.lemma_).strip().lower())

        return " ".join(singular).strip()


    def parse(self, text):
        if len(text) >= self.text_max_length:
            sentence_index = None
            for match in re.finditer(r'([^\.]\.|\?|\!)(?=\s+)', text[:self.text_max_length], re.MULTILINE):
                sentence_index = match.end()
            sentence_index = sentence_index if sentence_index else self.text_max_length

            topics = [
                *self._parse(text[:sentence_index].strip()),
                *self.parse(text[sentence_index:].strip())
            ]
        else:
            topics = self._parse(text)

        return topics

    def _parse(self, text):
        parser = self.spacy(text)
        topics = []

        for chunk in parser.noun_chunks:
            topic = []
            for index, word in enumerate(chunk):
                if (index > 0 or str(word) not in stop_words.STOP_WORDS) \
                    and not word.is_punct \
                    and len(str(word)) > 1 \
                    and not bool(re.search(r'([^\x00-\x7F]|\d+|\'|\"|\?|\(|\)|\[|\]|\||\=|\.)', str(word))) \
                    and word.pos_ not in ['PRON', 'ADP', 'ADV', 'VERB', 'DET', 'CCONJ', 'SCONJ']:
                    topic.append(str(word.lemma_).strip().lower())

            if topic and len(topic) < 4:
                topics.append(" ".join([ word.strip() for word in topic ]).strip())

            topics.append(str(chunk.root.lemma_).strip().lower())

        return topics
