from django.conf import settings

from systems.models.index import Model
from utility.data import Collection, ensure_list
from utility.topics import TopicModel

import time
import re
import math
import statistics


class BaseModelSummarizer(object):

    def __init__(
        self,
        command,
        instance,
        text_facade=None,
        document_facade=None,
        provider=None,
        section_provider=None,
    ):
        self.command = command
        self.instance = instance

        self.provider = provider
        self.summarizer = self.command.get_summarizer(
            init=False, provider=self.provider
        )
        self.section_provider = section_provider if section_provider else provider
        self.section_summarizer = self.command.get_summarizer(
            init=False, provider=self.section_provider
        )

        self.topics = TopicModel()

        self.text_facade = Model(text_facade).facade if text_facade else None
        self.instance_order = "created"

        self.text_id_field = self.instance.facade.pk
        self.text_field = None
        self.text_filters = {}

        self.document_facade = (
            Model(document_facade).facade if document_facade else None
        )
        self.document_filters = {}
        self.document_id_field = None

        self.embedding_collection = None
        self.embedding_id_field = self.instance.facade.pk

    def _get_text_chunks(self, text, prompt, persona, output_format, max_chunks):
        max_token_count = self.summarizer.get_chunk_length()
        prompt_token_count = self.summarizer.get_prompt_token_count(
            text=text, prompt=prompt, persona=persona, output_format=output_format
        )
        token_count = prompt_token_count

        chunks = [""]
        chunk_index = 0

        if text.strip():
            for section_index, section in enumerate(
                self.command.parse_text_sections(text)
            ):
                tokens = self.summarizer.get_token_count(section)
                if (token_count + tokens) > max_token_count:
                    chunk_index += 1
                    if not max_chunks or chunk_index < max_chunks:
                        chunks.append(section)
                        token_count = prompt_token_count + tokens
                    else:
                        break
                else:
                    token_count += tokens
                    chunks[chunk_index] = "{}\n\n{}".format(
                        chunks[chunk_index], section
                    )

        return chunks

    def _get_chunks(
        self,
        prompt,
        max_chunks,
        search_prompt=None,
        user_prompt=None,
        include_files=True,
        sentence_limit=50,
        min_score=0,
    ):
        documents = {}
        ranked_documents = {}
        document_scores = {}
        document_failed = {}

        chunks = []

        if not search_prompt:
            search_prompt = prompt

        # Get text chunks
        if self.text_facade and self.text_field:
            for text in self.text_facade.set_order(self.instance_order).field_values(
                self.text_field,
                **{
                    "{}__isnull".format(self.text_field): False,
                    self.text_id_field: self.instance.id,
                    **self.text_filters,
                }
            ):
                if text.strip():
                    section = "The following is a {} for the {} to be used for summarization and answering questions:\n\n{}".format(
                        self.text_field, self.text_facade.name.replace("_", " "), text
                    ).strip()

                    chunks.append(section)
                    break

        # Find documents
        if include_files and self.document_facade and self.embedding_collection:
            total_sentences = 0
            document_sentences = {}
            document_indexes = {}

            search_topics = self.topics.parse(user_prompt if user_prompt else prompt)
            document_topic_scores = {}

            document_results = self.document_facade.filter(
                **{self.document_id_field: self.instance.id}
            )

            if self.command.debug:
                self.command.notice("Chunking documents")
                self.command.data("Document ID Field", self.document_id_field)
                self.command.data("Document Instance ID", self.instance.id)
                self.command.data("Document Results", document_results)
                self.command.data("Search Topics", search_topics)

            for document in document_results:
                topic_score = self.topics.get_topic_score(
                    search_topics, document.topics
                )

                if self.command.debug:
                    self.command.info(
                        "Document: {} ({})".format(document.id, document.name)
                    )
                    self.command.data("Document Topics", document.topics)
                    self.command.data("Document Topic Score", topic_score)

                if document.description:
                    description_topics = self.topics.parse(document.description)
                    topic_score += self.topics.get_topic_score(
                        search_topics, description_topics
                    )

                    if self.command.debug:
                        self.command.data("Description Topics", description_topics)
                        self.command.data("Document Updated Topic Score", topic_score)

                if topic_score:
                    document_topic_scores[document.id] = topic_score

            if self.command.debug:
                self.command.data("Document Topic Scores", document_topic_scores)

            if document_topic_scores:
                search = self.command.generate_text_embeddings(
                    search_prompt, validate=False
                )
                document_rankings = self.command.search_embeddings(
                    self.embedding_collection,
                    search.embeddings,
                    limit=sentence_limit,
                    fields=["order", "topics"],
                    filter_field=self.embedding_id_field,
                    filter_ids=list(document_topic_scores.keys()),
                    min_score=min_score,
                )
                for index, ranking in enumerate(document_rankings):
                    for ranking_index, sentence_info in enumerate(ranking):
                        sentence = sentence_info.payload["sentence"].strip()
                        document_id = sentence_info.payload[self.embedding_id_field]
                        topic_score = self.topics.get_topic_score(
                            search_topics, sentence_info.payload["topics"]
                        )

                        document_indexes["{}:{}".format(document_id, sentence)] = int(
                            sentence_info.payload["order"]
                        )

                        if self.command.debug:
                            self.command.data(
                                "Found Sentence ({})".format(sentence_info.score),
                                sentence,
                            )

                        if document_id not in document_scores:
                            if document_id not in document_failed:
                                document_results = self.document_facade.filter(
                                    **{"id": document_id, **self.document_filters}
                                )
                                if document_results:
                                    document = document_results[0]
                                    documents[document_id] = document
                                    document_scores[document_id] = (
                                        sentence_info.score * (1 + topic_score)
                                    )

                                    if document_id not in document_sentences:
                                        document_sentences[document_id] = [sentence]
                                    else:
                                        document_sentences[document_id].append(sentence)

                                    total_sentences += 1
                                else:
                                    document_failed[document_id] = True
                        else:
                            document_scores[document_id] += sentence_info.score * (
                                1 + topic_score
                            )
                            document_sentences[document_id].append(sentence)
                            total_sentences += 1

                for document_id, document_score in document_scores.items():
                    document_scores[document_id] = (
                        document_score  # >= 0
                        * (
                            len(document_sentences[document_id]) / total_sentences
                        )  # 0 - 1
                        * (document_topic_scores.get(document_id, 0) + 1)  # >= 0
                    )

            if self.command.debug:
                self.command.data("Document Scores", document_scores)

            if documents:
                for document_id, score in sorted(
                    document_scores.items(), key=lambda x: x[1], reverse=True
                ):
                    document = documents[document_id]
                    completed = False

                    if document_sentences[document_id]:
                        for section_index, section in enumerate(
                            self.parse_sections(
                                document,
                                document_sentences[document_id],
                                document_indexes,
                            )
                        ):
                            if len(chunks) < max_chunks:
                                section_info = {
                                    "text": section,
                                    "type": document.type,
                                    "id": document_id,
                                }
                                if self.command.debug:
                                    section_info["tokens"] = (
                                        self.section_summarizer.get_token_count(section)
                                    )
                                chunks.append(section_info)
                            else:
                                completed = True
                                break

                        ranked_documents[document.id] = {
                            "document": document,
                            "score": score,
                        }
                        if completed:
                            break

        return chunks, ranked_documents

    def parse_sections(self, document, sentences, indexes, max_section_tokens=3500):
        max_period_tokens = max_section_tokens / 2
        sentence_map = {}
        sections = []

        if document.text and not document.sentences:
            document.sentences = (
                self.command.parse_sentences(document.text, validate=False)
                if document.text
                else []
            )
            document.save()

        for sentence in set(sentences):
            sentence_tokens = self.section_summarizer.get_token_count(sentence)
            before_context = []
            before_tokens = 0
            after_context = []
            after_tokens = 0

            try:
                sentence_index = indexes["{}:{}".format(document.id, sentence)]
                sentence_map[sentence_index] = document.sentences[sentence_index]

            except ValueError as e:
                continue

            # Find relevant before
            for before_index in range((sentence_index - 1), -1, -1):
                previous_sentence = document.sentences[before_index]
                if previous_sentence:
                    previous_sentence = str(previous_sentence)
                    previous_tokens = self.section_summarizer.get_token_count(
                        previous_sentence
                    )
                    if (before_tokens + previous_tokens) > max_period_tokens:
                        break
                    before_context.append(previous_sentence)
                    before_tokens += previous_tokens
                    sentence_map[before_index] = previous_sentence

            # Find relevant after
            for after_index in range((sentence_index + 1), len(document.sentences)):
                next_sentence = document.sentences[after_index]
                if next_sentence:
                    next_sentence = str(next_sentence)
                    next_tokens = self.section_summarizer.get_token_count(next_sentence)
                    if (after_tokens + next_tokens) > max_period_tokens:
                        break
                    after_context.append(next_sentence)
                    after_tokens += next_tokens
                    sentence_map[after_index] = next_sentence

            # Find similarity around context
            if self.command.debug:
                self.command.info("=" * self.command.display_width)
                self.command.info("")
                self.command.info("-------------------------------")
                self.command.info("Sentence")
                self.command.info("-------------------------------")
                self.command.notice("{} ( {} )".format(sentence, sentence_index))
                self.command.notice(
                    "Tokens: {} / {}".format(before_tokens, after_tokens)
                )

        previous_index = None
        section = []

        for sentence_index in sorted(sentence_map.keys()):
            sentence = sentence_map[sentence_index].strip()

            if previous_index is None or ((sentence_index - previous_index) > 1):
                # New section
                if previous_index:
                    sections.append("\n".join(section))

                if document.description:
                    document_intro = "The following is an excerpt from {} '{}' with the following description: {}.\n\n".format(
                        document.type, document.name, document.description.strip(".!?")
                    )
                else:
                    document_intro = (
                        "The following is an excerpt from {} '{}'.\n\n".format(
                            document.type, document.name
                        )
                    )

                section = [
                    document_intro,
                    "Use this excerpt exclusively for summarization and answering questions:\n\n",
                    sentence,
                ]
            else:
                # Continue section
                section.append(sentence)

            previous_index = sentence_index

        if section:
            sections.append("\n".join(section))

        return sections

    def generate(
        self,
        prompt,
        search_prompt=None,
        user_prompt=None,
        output_format="",
        output_endings=None,
        max_chunks=10,
        include_files=True,
        sentence_limit=50,
        **config
    ):
        max_token_count = self.summarizer.get_chunk_length()
        persona = config.get("persona", "")

        if output_endings is None:
            output_endings = [".", "?", "!"]

        def generate_summary(info):
            chunk = info["chunk"]
            chunk_text = chunk["text"] if isinstance(chunk, dict) else chunk
            _sub_prompt = """
Extract only the relevant information from the provided text for the following request: {}

If there is no relevant information in the provided text
return only the phrase: No information available.
""".format(
                prompt
            )
            if self.command.debug:
                self.command.data(
                    "Prompt tokens",
                    self.section_summarizer.get_token_count(_sub_prompt),
                )
                self.command.data(
                    "Chunk tokens", self.section_summarizer.get_token_count(chunk_text)
                )

            _summary = self.command.generate_summary(
                chunk_text, prompt=_sub_prompt, provider=self.section_provider, **config
            )
            if self.command.debug:
                self.command.notice(
                    """
================================
{}
................................
Request Tokens: {}
Response Tokens: {}
Summary Cost: ${}
""".format(
                        _summary.text,
                        _summary.request_tokens,
                        _summary.response_tokens,
                        _summary.cost,
                    )
                )

            summary_text = _summary.text.strip()
            if re.search(r"^No information available", summary_text):
                summary_text = ""

            return {
                "index": info["index"],
                "type": chunk["type"] if isinstance(chunk, dict) else "text",
                "id": chunk["id"] if isinstance(chunk, dict) else None,
                "text": summary_text,
                "request_tokens": _summary.request_tokens,
                "response_tokens": _summary.response_tokens,
                "cost": _summary.cost,
            }

        def summarize(text=None):
            _summary_text = ""

            if text is not None:
                _chunks = self._get_text_chunks(
                    text, prompt, persona, output_format, max_chunks
                )
                _documents = {}

                if self.command.debug:
                    self.command.info("Text chunks:")
                    for chunk in _chunks:
                        self.command.data(
                            "Chunk Tokens", self.summarizer.get_token_count(chunk)
                        )
            else:
                _chunks, _documents = self._get_chunks(
                    prompt,
                    max_chunks,
                    search_prompt=search_prompt,
                    user_prompt=user_prompt,
                    include_files=include_files,
                    sentence_limit=sentence_limit,
                )

            if self.command.debug:
                self.command.data("Summary Chunks", _chunks)
                self.command.info("Summary Documents")
                for document_id, info in _documents.items():
                    self.command.data(document_id, info["score"])

            _chunks = [
                {"index": _index, "chunk": _chunk}
                for _index, _chunk in enumerate(_chunks)
            ]
            _request_tokens = 0
            _response_tokens = 0
            _processing_cost = 0

            if len(_chunks):
                if len(_chunks) > 1:
                    _results = self.command.run_list(_chunks, generate_summary)
                    _chunk_text = {}
                    _chunk_tokens = 0

                    for _chunk in _results.data:
                        _request_tokens += _chunk.result["request_tokens"]
                        _response_tokens += _chunk.result["response_tokens"]
                        _processing_cost += _chunk.result["cost"]

                        if _chunk.result["text"]:
                            _text_tokens = self.summarizer.get_token_count(
                                _chunk.result["text"]
                            )
                            if (_chunk_tokens + _text_tokens) <= max_token_count:
                                _chunk_text[_chunk.result["index"]] = _chunk.result[
                                    "text"
                                ]
                                _chunk_tokens += _text_tokens
                        else:
                            if self.command.debug:
                                self.command.data("Removing Document", _chunk.result)

                            _documents.pop(_chunk.result["id"], None)

                    _summary_input_text = "\n\n".join(
                        [_chunk_text[_index] for _index in sorted(_chunk_text.keys())]
                    )

                    if self.command.debug:
                        self.command.data("Summary Input Text", _summary_input_text)

                    (
                        _summary_text,
                        _final_request_tokens,
                        _final_response_tokens,
                        _final_cost,
                        _chunk_documents,
                    ) = summarize(_summary_input_text)

                    _request_tokens += _final_request_tokens
                    _response_tokens += _final_response_tokens
                    _processing_cost += _final_cost
                else:
                    _chunk = _chunks[0]["chunk"]
                    _summary = self.command.generate_summary(
                        _chunk["text"] if isinstance(_chunk, dict) else _chunk,
                        prompt=prompt,
                        output_format=output_format,
                        endings=output_endings,
                        provider=self.provider,
                        **config
                    )
                    _summary_text = _summary.text
                    _request_tokens += _summary.request_tokens
                    _response_tokens += _summary.response_tokens
                    _processing_cost += _summary.cost

                    if self.command.debug:
                        self.command.notice(
                            """
**================================**
{}
**................................**
Request Tokens: {}
Response Tokens: {}
Summary Cost: ${}
""".format(
                                _summary_text,
                                _request_tokens,
                                _response_tokens,
                                _processing_cost,
                            )
                        )

            return (
                _summary_text,
                _response_tokens,
                _response_tokens,
                _processing_cost,
                _documents,
            )

        start_time = time.time()
        summary_text, request_tokens, response_tokens, cost, documents = summarize()

        return Collection(
            text=summary_text,
            documents=documents,
            request_tokens=request_tokens,
            response_tokens=response_tokens,
            token_count=(request_tokens + response_tokens),
            processing_time=(time.time() - start_time),
            processing_cost=cost,
        )
