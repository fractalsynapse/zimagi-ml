from django.conf import settings

from utility.data import Collection, ensure_list

import time


class TextSummarizer(object):

    def __init__(self, command, text, provider = None, no_info_message = None):
        self.command = command
        self.text = text.strip() if text else text
        self.no_info_message = no_info_message if no_info_message else 'No information was found'

        self.provider = provider
        self.summarizer = self.command.get_summarizer(init = False, provider = self.provider)


    def _get_chunks(self, text, prompt, max_chunks = 0):
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


    def generate(self, prompt, output_format = '', output_endings = None, **config):
        if output_endings is None:
            output_endings = [ '.', '?', '!' ]

        max_chunks = config.pop('max_chunks', 0)

        def generate_summary(info):
            _summary_text, _request_tokens, _response_tokens = self.command.generate_summary(
                info['text'],
                prompt = prompt,
                provider = self.provider,
                **config
            )
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

        def summarize(text):
            _summary_text = ''
            _chunks = [ { 'index': _index, 'text': _chunk } for _index, _chunk in enumerate(self._get_chunks(
                text,
                prompt,
                max_chunks = max_chunks
            )) ]

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

                    _summary_text, _final_request_tokens, _final_response_tokens = summarize(
                        "\n\n".join([ _chunk_text[_index] for _index in sorted(_chunk_text.keys()) ])
                    )
                    _request_tokens += _final_request_tokens
                    _response_tokens += _final_response_tokens
                else:
                    _summary_text, _sub_request_tokens, _sub_response_tokens = self.command.generate_summary(
                        _chunks[0]['text'],
                        prompt = prompt,
                        format = output_format,
                        endings = output_endings,
                        provider = self.provider,
                        **config
                    )
                    _request_tokens += _sub_request_tokens
                    _response_tokens += _sub_response_tokens

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

            return _summary_text.strip(), _request_tokens, _response_tokens

        start_time = time.time()

        if self.text is not None:
            summary_text, request_tokens, response_tokens = summarize(self.text)
        else:
            summary_text = self.no_info_message
            request_tokens = 0
            response_tokens = 0

        token_count = (request_tokens + response_tokens)

        return Collection(
            text = summary_text,
            request_tokens = request_tokens,
            response_tokens = response_tokens,
            token_count = token_count,
            processing_time = (time.time() - start_time),
            processing_cost = (token_count * settings.SUMMARIZER_COST_PER_TOKEN)
        )
