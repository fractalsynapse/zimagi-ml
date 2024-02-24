from sklearn.metrics.pairwise import cosine_similarity
from django.conf import settings

from systems.models.index import Model
from utility.data import Collection

import time
import re
import math
import statistics


class BaseModelSummarizer(object):

    def __init__(self,
        command,
        instance,
        text_facade = None,
        document_facade = None
    ):
        self.command = command
        self.instance = instance
        self.summarizer = self.command.get_summarizer(init = False)

        self.text_facade = Model(text_facade).facade if text_facade else None
        self.instance_order = 'created'

        self.text_id_field = self.instance.facade.pk
        self.text_field = None
        self.text_filters = {}

        self.document_facade = Model(document_facade).facade if document_facade else None
        self.document_filters = {}

        self.embedding_collection = None
        self.embedding_id_field = self.instance.facade.pk


    def _get_text_chunks(self, text, prompt, max_chunks):
        max_token_count = self.summarizer.get_chunk_length()
        prompt_token_count = self.summarizer.get_token_count(prompt)
        token_count = prompt_token_count

        chunks = [""]
        chunk_index = 0

        if text.strip():
            for section_index, section in enumerate(self.command.parse_text_sections(text)):
                tokens = self.summarizer.get_token_count(section)
                if (token_count + tokens) > max_token_count:
                    chunk_index += 1
                    if not max_chunks or chunk_index < max_chunks:
                        chunks.append(section)
                        token_count = (prompt_token_count + tokens)
                    else:
                        break
                else:
                    token_count += tokens
                    chunks[chunk_index] = "{}\n\n{}".format(chunks[chunk_index], section)

        return chunks


    def _get_chunks(self, prompt, max_chunks, search_prompt = None, include_files = True, sentence_limit = 50):
        max_token_count = self.summarizer.get_chunk_length()
        prompt_token_count = self.summarizer.get_token_count(prompt)
        text_token_count = 0
        token_count = prompt_token_count

        documents = {}
        ranked_documents = {}
        document_index = {}
        document_scores = {}
        document_failed = {}

        chunks = [""]
        chunk_index = 0

        if not search_prompt:
            search_prompt = prompt

        # Get text chunks
        if self.text_facade and self.text_field:
            text_chunks = []
            text_tokens = []

            for text in self.text_facade.set_order(self.instance_order).field_values(self.text_field, **{
                "{}__isnull".format(self.text_field): False,
                self.text_id_field: self.instance.id,
                **self.text_filters
            }):
                if text.strip():
                    section = "The following is a {} for the {}\n\n{}".format(
                        self.text_field,
                        self.text_facade.name,
                        text
                    ).strip()

                    text_chunks.append(section)
                    text_tokens.append(self.summarizer.get_token_count(section))

            text_token_count = sum(text_tokens)
            max_text_token_count = (max_token_count / 3)
            while len(text_tokens) > 1 and text_token_count > max_text_token_count:
                text_chunks.pop()
                text_tokens.pop()
                text_token_count = sum(text_tokens)

            chunks[chunk_index] = "\n\n".join(text_chunks)

        # Find documents
        if include_files and self.document_facade and self.embedding_collection:
            total_sentences = 0
            document_sentences = {}

            search = self.command.generate_text_embeddings(search_prompt)
            document_rankings = self.command.search_embeddings(self.embedding_collection,
                search.embeddings,
                limit = sentence_limit,
                fields = [ 'document_id' ],
                filter_field = self.embedding_id_field,
                filter_ids = self.instance.id
            )
            for index, ranking in enumerate(document_rankings):
                for ranking_index, sentence_info in enumerate(ranking):
                    sentence = sentence_info.payload['sentence'].strip()
                    if 'document_id' in sentence_info.payload and len(re.split(r'\s+', sentence)) >= 5:
                        document_id = sentence_info.payload['document_id']

                        if document_id not in document_scores:
                            if document_id not in document_failed:
                                document_results = self.document_facade.filter(**{
                                    'id': document_id,
                                    **self.document_filters
                                })
                                if document_results:
                                    document = document_results[0]
                                    documents[document_id] = document
                                    document_scores[document_id] = sentence_info.score

                                    if document_id not in document_sentences:
                                        document_sentences[document_id] = [ sentence ]
                                    else:
                                        document_sentences[document_id].append(sentence)

                                    total_sentences += 1
                                else:
                                    document_failed[document_id] = True
                        else:
                            document_scores[document_id] += sentence_info.score
                            document_sentences[document_id].append(sentence)
                            total_sentences += 1

            for document_id, document_score in document_scores.items():
                document_scores[document_id] = (math.sqrt(
                    (document_score / total_sentences)
                    * (len(document_sentences[document_id]) / total_sentences)
                    ) * 100
                )

        # Get document chunks
        if documents:
            for document_id, score in sorted(document_scores.items(), key = lambda x:x[1], reverse = True):
                document = documents[document_id]
                if document_sentences[document_id] and document.get_index_key() not in document_index:
                    for section_index, section in enumerate(self.parse_sections(document, document_sentences[document_id])):
                        tokens = self.summarizer.get_token_count(section)

                        if (token_count + tokens) > max_token_count:
                            if not max_chunks or chunk_index < max_chunks:
                                chunk_index += 1
                                chunks.append(section)
                                token_count = tokens
                            else:
                                break
                        else:
                            token_count += tokens
                            chunks[chunk_index] = "{}\n\n{}".format(chunks[chunk_index], section)

                    ranked_documents["{:07.3f}:{}".format(score, document.id)] = document
                    document_index[document.get_index_key()] = True

        return chunks, ranked_documents


    def parse_sections(self, document, sentences, max_section_tokens = 6000, selectivity = -2):
        max_period_tokens = (max_section_tokens / 2)
        sentence_map = {}
        sections = []

        if document.text and not document.sentences:
            document.sentences = [
                re.sub(r'\s+', ' ', sentence)
                for sentence in self.command.parse_sentences(document.text, validate = False)
            ] if document.text else []
            document.save()

        for sentence in set(sentences):
            sentence = re.sub(r'\s+', ' ', sentence)
            sentence_tokens = self.summarizer.get_token_count(sentence)
            try:
                sentence_index = document.sentences.index(sentence)

                before_context = []
                before_tokens = 0
                after_context = []
                after_tokens = 0

                sentence_map[sentence_index] = sentence

                # Find relevant before
                for before_index in range((sentence_index - 1), -1, -1):
                    previous_sentence = document.sentences[before_index]
                    previous_tokens = self.summarizer.get_token_count(previous_sentence)
                    if (before_tokens + previous_tokens) > max_period_tokens:
                        break
                    before_context.append(previous_sentence)
                    before_tokens += previous_tokens
                    sentence_map[(sentence_index - before_index - 1)] = previous_sentence

                # Find relevant after
                for after_index in range((sentence_index + 1), len(document.sentences)):
                    next_sentence = document.sentences[after_index]
                    next_tokens = self.summarizer.get_token_count(next_sentence)
                    if (after_tokens + next_tokens) > max_period_tokens:
                        break
                    after_context.append(next_sentence)
                    after_tokens += next_tokens
                    sentence_map[(sentence_index + after_index + 1)] = next_sentence

                # Find similarity around context
                if False: # self.command.debug:
                    self.command.info('=' * self.command.display_width)
                    self.command.info('')
                    self.command.info('-------------------------------')
                    self.command.info('Sentence')
                    self.command.info('-------------------------------')
                    self.command.notice(sentence)

                # sentence_embeddings = self.command.generate_embeddings([ sentence ])

                # if len(before_context):
                #     previous_similarities = cosine_similarity(
                #         sentence_embeddings,
                #         self.command.generate_embeddings(before_context)
                #     )[0]
                #     cutoff_score = None
                #     if len(previous_similarities) > 1:
                #         cutoff_score = max(statistics.mean(previous_similarities) + (selectivity * statistics.stdev(previous_similarities)), 0)

                #     if False: # self.command.debug:
                #         self.command.info('')
                #         self.command.info('-------------------------------')
                #         self.command.info('Processing Previous Similarities')
                #         self.command.info('-------------------------------')
                #         self.command.data('Cutoff', cutoff_score)

                #     for index, sentence in enumerate(before_context):
                #         score = previous_similarities[index]
                #         if cutoff_score and score < cutoff_score:
                #             del before_context[index:]
                #             break

                #         if False: # self.command.debug:
                #             self.command.data(score, sentence)

                #         sentence_map[(sentence_index - index - 1)] = sentence

                # if len(after_context):
                #     next_similarities = cosine_similarity(
                #         sentence_embeddings,
                #         self.command.generate_embeddings(after_context)
                #     )[0]
                #     cutoff_score = None
                #     if len(next_similarities) > 1:
                #         cutoff_score = max(statistics.mean(next_similarities) + (selectivity * statistics.stdev(next_similarities)), 0)

                #     if False: # self.command.debug:
                #         self.command.info('')
                #         self.command.info('-------------------------------')
                #         self.command.info('Processing Next Similarities')
                #         self.command.info('-------------------------------')
                #         self.command.data('Cutoff', cutoff_score)

                #     for index, sentence in enumerate(after_context):
                #         score = next_similarities[index]
                #         if cutoff_score and score < cutoff_score:
                #             del after_context[index:]
                #             break

                #         if False: # self.command.debug:
                #             self.command.data(score, sentence)

                #         sentence_map[(sentence_index + index + 1)] = sentence

            except ValueError:
                pass

        previous_index = None
        section = []

        for sentence_index in sorted(sentence_map.keys()):
            sentence = sentence_map[sentence_index].strip()

            if previous_index is None or ((sentence_index - previous_index) > 1):
                # New section
                sections.append("\n".join(section))
                section = [ "The following is an excerpt from a document relevant to the query:\n\n", sentence ]
            else:
                # Continue section
                section.append(sentence)

            previous_index = sentence_index

        if section:
            sections.append("\n".join(section))

        return sections


    def generate(self, prompt, search_prompt = None, output_format = '', max_chunks = 2, include_files = True, sentence_limit = 50, **config):

        def generate_summary(info):
            _sub_prompt = "Extract and format the relevant information from the provided text for the following request: {}".format(prompt)
            _request_tokens = self.summarizer.get_token_count(info['text'])
            _summary_text = self.command.generate_summary(info['text'], prompt = _sub_prompt, **config)
            _response_tokens = self.summarizer.get_token_count(_summary_text)

            if self.command.debug:
                self.command.notice(
"""
================================
{}
................................
Request Tokens: {}
Response Tokens: {}
""".format(
                    _summary_text,
                    _request_tokens,
                    _response_tokens
                ))

            return {
                'index': info['index'],
                'text': _summary_text.strip(),
                'request_tokens': _request_tokens,
                'response_tokens': _response_tokens
            }

        def summarize(text = None):
            _summary_text = ''

            if text:
                _chunks = self._get_text_chunks(text, prompt, max_chunks)
                _documents = {}
            else:
                _chunks, _documents = self._get_chunks(prompt, max_chunks,
                    search_prompt = search_prompt,
                    include_files = include_files,
                    sentence_limit = sentence_limit
                )

            _chunks = [ { 'index': _index, 'text': _chunk } for _index, _chunk in enumerate(_chunks) ]
            _request_tokens = 0
            _response_tokens = 0

            if len(_chunks):
                if (not max_chunks or max_chunks > 1) and len(_chunks) > 1:
                    _results = self.command.run_list(_chunks, generate_summary)
                    _chunk_text = {}
                    for _chunk in _results.data:
                        _chunk_text[_chunk.result['index']] = _chunk.result['text']
                        _request_tokens += _chunk.result['request_tokens']
                        _response_tokens += _chunk.result['response_tokens']

                    _summary_text, _final_request_tokens, _final_response_tokens, _chunk_documents = summarize(
                        "\n\n".join([ _chunk_text[_index] for _index in sorted(_chunk_text.keys()) ])
                    )
                    _request_tokens += _final_request_tokens
                    _response_tokens += _final_response_tokens
                    _documents = { **_documents, **_chunk_documents }
                else:
                    _request_tokens += self.summarizer.get_token_count(_chunks[0]['text'])
                    _summary_text = self.command.generate_summary(_chunks[0]['text'],
                        prompt = prompt,
                        format = output_format,
                        **config
                    )
                    _response_tokens += self.summarizer.get_token_count(_summary_text)

                    if self.command.debug:
                        self.command.notice(
"""
**================================**
{}
**................................**
Request Tokens: {}
Response Tokens: {}
""".format(
                            _summary_text,
                            _request_tokens,
                            _response_tokens
                        ))

            return _summary_text.strip(), _request_tokens, _response_tokens, _documents

        start_time = time.time()
        summary_text, request_tokens, response_tokens, documents = summarize()
        token_count = (request_tokens + response_tokens)

        return Collection(
            text = summary_text,
            documents = documents,
            request_tokens = request_tokens,
            response_tokens = response_tokens,
            token_count = token_count,
            processing_time = (time.time() - start_time),
            processing_cost = (token_count * settings.SUMMARIZER_COST_PER_TOKEN)
        )
