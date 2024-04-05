from spacy.lang.en import stop_words

import re
import spacy


class TopicModel(object):

    def __init__(self):
        self.spacy = spacy.load('en_core_web_lg')
        self.spacy.max_length = 10000000


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


    def parse(self, text):
        parser = self.spacy(text)
        topics = []

        for chunk in parser.noun_chunks:
            if not bool(re.search(r'([^\x00-\x7F]|\d+|\'|\"|\(|\)|\[|\]|\.)', str(chunk))):
                topic = []
                for index, word in enumerate(chunk):
                    if (index > 0 or str(word) not in stop_words.STOP_WORDS) \
                        and not word.is_punct \
                        and len(str(word)) > 1 \
                        and word.pos_ not in ['PRON', 'ADP', 'ADJ', 'ADV', 'VERB', 'DET', 'CCONJ', 'SCONJ']:
                        topic.append(str(word.lemma_).strip().lower())

                if topic and len(topic) < 4:
                    topics.append(" ".join([ word.strip() for word in topic ]).strip())

        return topics
