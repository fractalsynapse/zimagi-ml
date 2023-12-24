from django.conf import settings

from systems.plugins.index import BaseProvider
from utility.data import get_identifier

import spacy
import re


class Provider(BaseProvider('sentence_parser', 'spacy')):

    @classmethod
    def initialize(cls, instance, init):
        if not getattr(cls, '_model', None):
            cls._model = {}

        if init and instance.identifier not in cls._model:
            cls._model[instance.identifier] = cls._get_model(instance)

    @classmethod
    def _get_model(cls, instance):
        return spacy.load(instance.field_model)

    def _get_identifier(self, init):
        return get_identifier([ super()._get_identifier(init), self.field_model ])


    @property
    def model(self):
        return self._model[self.identifier]


    def split(self, text):
        char_limit = 1000000

        def _get_sentences(inner_text):
            if len(inner_text) > char_limit:
                sentence_index = (inner_text[:char_limit].rindex('.') + 1)
                sentences = [
                    *_get_sentences(inner_text[:sentence_index].strip()),
                    *_get_sentences(inner_text[sentence_index:].strip())
                ]
            else:
                doc = self.model(inner_text)
                sentences = []

                for sentence in doc.sents:
                    if sentence[0].is_title:
                        noun_count = 0
                        pronoun_count = 0
                        verb_count = 0

                        for token in sentence:
                            if token.pos_ in [ 'NOUN', 'PROPN' ]:
                                noun_count += 1
                            elif token.pos_ in [ 'PRON' ]:
                                pronoun_count += 1
                            elif token.pos_ == 'VERB':
                                verb_count += 1

                        if noun_count > 0 or verb_count > 0 or pronoun_count > 0:
                            sentence = re.sub(r'\n+', ' ', str(sentence)).strip()
                            if len(sentence) < self.get_max_sentence_length():
                                sentences.append(sentence)
            return sentences
        return _get_sentences(text.strip())