from django.conf import settings

from systems.models.index import Model
from utility.data import Collection

import time


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

        self.document_facade = Model(document_facade).facade if document_facade else None

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


    def _get_chunks(self, prompt, max_chunks, search_prompt = None, include_files = True, sentence_limit = 20):
        max_token_count = self.summarizer.get_chunk_length()
        prompt_token_count = self.summarizer.get_token_count(prompt)
        token_count = prompt_token_count

        documents = {}
        ranked_documents = {}
        document_scores = {}
        document_failed = {}

        chunks = [""]
        chunk_index = 0

        if not search_prompt:
            search_prompt = prompt

        # Get text chunks
        if self.text_facade and self.text_field:
            for text in self.text_facade.set_order(self.instance_order).field_values(self.text_field, **{
                "{}__isnull".format(self.text_field): False,
                self.text_id_field: self.instance.id
            }):
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

        # Find documents
        if include_files and self.document_facade and self.embedding_collection:
            embeddings = self.command.generate_text_embeddings(search_prompt)
            document_rankings = self.command.search_embeddings(self.embedding_collection,
                embeddings.embeddings,
                limit = sentence_limit,
                fields = [ 'document_id' ],
                filter_field = self.embedding_id_field,
                filter_ids = self.instance.id
            )
            for index, ranking in enumerate(document_rankings):
                for ranking_index, sentence_info in enumerate(ranking):
                    sentence = sentence_info.payload['sentence']
                    document_id = sentence_info.payload['document_id']

                    if sentence_info.score >= 0.5:
                        if document_id not in document_scores:
                            if document_id not in document_failed:
                                document = self.document_facade.retrieve_by_id(document_id)
                                if document:
                                    documents[document_id] = document
                                    document_scores[document_id] = sentence_info.score
                                else:
                                    document_failed[document_id] = True
                        else:
                            document_scores[document_id] += sentence_info.score

        # Get document chunks
        if documents:
            for document_id, score in sorted(document_scores.items(), key = lambda x:x[1], reverse = True):
                document = documents[document_id]
                if document.text.strip():
                    for section_index, section in enumerate(self.command.parse_text_sections(document.text)):
                        tokens = self.summarizer.get_token_count(section)

                        if (token_count + tokens) > max_token_count:
                            chunk_index += 1
                            if not max_chunks or chunk_index < max_chunks:
                                chunks.append(section)
                                token_count = tokens
                            else:
                                break
                        else:
                            token_count += tokens
                            chunks[chunk_index] = "{}  {}".format(chunks[chunk_index], section)

                    ranked_documents["{:07.3f}:{}".format(score, document.id)] = document

        return chunks, ranked_documents


    def generate(self, prompt, search_prompt = None, output_format = '', max_chunks = 2, include_files = True, sentence_limit = 500, **config):

        def generate_summary(info):
            _sub_prompt = "Extract and format the relevant information from the provided text for the following request: {}".format(prompt)
            _request_tokens = self.summarizer.get_token_count(info['text'])
            _summary_text = self.command.generate_summary(info['text'], prompt = _sub_prompt, **config)
            _response_tokens = self.summarizer.get_token_count(_summary_text)

            if self.command.verbosity == 3:
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

                    if self.command.verbosity == 3:
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
