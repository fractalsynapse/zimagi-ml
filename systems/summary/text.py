from django.conf import settings

from utility.data import Collection, ensure_list

import time


class TextSummarizer(object):

    def __init__(self, command, text, provider=None, no_info_message=None):
        self.command = command
        self.text = text if text else text
        self.no_info_message = (
            no_info_message if no_info_message else "No information was found"
        )

        self.provider = provider
        self.summarizer = self.command.get_summarizer(
            init=False, provider=self.provider
        )

    def _get_chunks(self, text, prompt, output_format="", persona=""):
        max_token_count = self.summarizer.get_chunk_length()
        prompt_token_count = self.summarizer.get_prompt_token_count(
            prompt, persona, output_format
        )
        token_count = prompt_token_count

        chunks = [""]
        chunk_index = 0

        if text:
            for section_index, section in enumerate(
                self.command.parse_text_sections(text)
            ):
                tokens = self.summarizer.get_token_count(section)
                if (token_count + tokens) > max_token_count:
                    chunk_index += 1
                    chunks.append(section)
                    token_count = prompt_token_count + tokens
                else:
                    token_count += tokens
                    chunks[chunk_index] = "{}\n\n{}".format(
                        chunks[chunk_index], section
                    )

        return chunks

    def generate(self, prompt, output_format="", output_endings=None, **config):
        persona = config.get("persona", "")

        if output_endings is None:
            output_endings = [".", "?", "!"]

        def generate_summary(info):
            _summary = self.command.generate_summary(
                info["text"], prompt=prompt, provider=self.provider, **config
            )
            if self.command.debug and self.command.verbosity > 2:
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

            return {
                "index": info["index"],
                "text": _summary.text.strip(),
                "request_tokens": _summary.request_tokens,
                "response_tokens": _summary.response_tokens,
                "cost": _summary.cost,
            }

        def summarize(text):
            _summary_text = ""
            _chunks = [
                {"index": _index, "text": _chunk}
                for _index, _chunk in enumerate(
                    self._get_chunks(
                        text,
                        prompt=prompt,
                        output_format=output_format,
                        persona=persona,
                    )
                )
            ]

            _request_tokens = 0
            _response_tokens = 0
            _processing_cost = 0

            if len(_chunks):
                if len(_chunks) > 1:
                    _results = self.command.run_list(_chunks, generate_summary)
                    _chunk_text = {}
                    for _chunk in _results.data:
                        _chunk_text[_chunk.result["index"]] = _chunk.result["text"]
                        _request_tokens += _chunk.result["request_tokens"]
                        _response_tokens += _chunk.result["response_tokens"]
                        _processing_cost += _chunk.result["cost"]

                    (
                        _summary_text,
                        _final_request_tokens,
                        _final_response_tokens,
                        _final_cost,
                    ) = summarize(
                        "\n\n".join(
                            [
                                _chunk_text[_index]
                                for _index in sorted(_chunk_text.keys())
                            ]
                        )
                    )
                    _request_tokens += _final_request_tokens
                    _response_tokens += _final_response_tokens
                    _processing_cost += _final_cost
                else:
                    _summary = self.command.generate_summary(
                        _chunks[0]["text"],
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

                    if self.command.debug and self.command.verbosity > 2:
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
                _summary_text.strip(),
                _request_tokens,
                _response_tokens,
                _processing_cost,
            )

        start_time = time.time()

        if self.text is not None:
            summary_text, request_tokens, response_tokens, cost = summarize(self.text)
        else:
            summary_text = self.no_info_message
            request_tokens = 0
            response_tokens = 0
            cost = 0

        return Collection(
            text=summary_text,
            request_tokens=request_tokens,
            response_tokens=response_tokens,
            token_count=(request_tokens + response_tokens),
            processing_time=(time.time() - start_time),
            processing_cost=cost,
        )
