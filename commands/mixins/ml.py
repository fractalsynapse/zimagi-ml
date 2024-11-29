from django.conf import settings

from systems.commands.index import CommandMixin
from utility.data import Collection, get_identifier, dump_json, ensure_list

import billiard as multiprocessing
import re


def check_ending(text, endings):
    if not endings:
        return True

    text = text.strip()
    for ending in endings:
        if text.endswith(ending):
            return True
    return False


class MLCommandMixin(CommandMixin('ml')):

    provider_lock = multiprocessing.Lock()


    def get_sentence_parser(self, **options):
        with self.provider_lock:
            if not getattr(self, 'providers', None):
                self.providers = []

            provider = self._get_model_provider('sentence_parser', settings.SENTENCE_PARSER_PROVIDERS, options)
            self._add_model(
                provider.provider_type,
                provider.name,
                provider.field_device
            )
            self.providers.append(provider)
        return provider

    def parse_sentences(self, text, **config):
        if not text:
            return []

        sentences = []
        for section in self.parse_text_sections(text.encode('ascii', 'ignore').decode().replace("\x00", '')):
            section_sentences = self.submit('agent:model:sentence_parser', {
                'text': section,
                'config': config
            })
            if section_sentences:
                sentences.extend(section_sentences)

        return sentences


    def get_encoder(self, **options):
        with self.provider_lock:
            if not getattr(self, 'providers', None):
                self.providers = []

            provider = self._get_model_provider('encoder', settings.ENCODER_PROVIDERS, options)
            self._add_model(
                provider.provider_type,
                provider.name,
                provider.field_device
            )
            self.providers.append(provider)
        return provider

    def generate_embeddings(self, sentences, **config):
        if not sentences:
            return []

        section_length = 125
        sections = [
            sentences[index * section_length:(index + 1) * section_length]
            for index in range((len(sentences) + section_length - 1) // section_length)
        ]
        embeddings = []

        for section in sections:
            embeddings.extend(self.submit('agent:model:embedding', {
                'sentences': section,
                'config': config
            }))

        return embeddings

    def generate_text_embeddings(self, text, **config):
        sentences = self.parse_sentences(text, **config)
        text_data = None

        if sentences:
            text_data = Collection(
                sentences = sentences,
                embeddings = self.generate_embeddings(sentences, **config)
            )
        return text_data


    def get_summarizer(self, **options):
        provider = options.get('provider', None)
        if not provider:
            provider = settings.SUMMARIZER_PROVIDERS

        with self.provider_lock:
            if not getattr(self, 'providers', None):
                self.providers = []

            provider = self._get_model_provider('summarizer', ensure_list(provider), options)
            self._add_model(
                provider.provider_type,
                provider.name,
                provider.field_device
            )
            self.providers.append(provider)

        return provider

    def generate_summary(self, text, **config):
        summary_provider = config.get('provider', None)
        summarizer = self.get_summarizer(init = False, provider = summary_provider)

        summary_channel = "agent:model:{}".format(summary_provider if summary_provider else 'summary')
        summary_persona = config.get('persona', '')
        summary_prompt = config.get('prompt', '')
        summary_format = config.get('output_format', '')
        summary_endings = ensure_list(config.get('endings', [ '.', '?', '!' ]))
        summary_retries = config.get('retries', 5)
        summary_config = {
            key: value for key, value in config.items()
            if key not in [ 'persona', 'prompt', 'output_format', 'retries', 'endings' ]
        }
        summary_id = get_identifier([
            text, summary_prompt, summary_persona, summary_format, summary_config
        ])

        def generate():
            summary = self._summary.get_or_create(summary_id)
            summary.text = ensure_list(text)
            summary.persona = summary_persona
            summary.prompt = summary_prompt
            summary.format = summary_format
            summary.endings = summary_endings
            summary.config = summary_config

            request_tokens = 0
            response_tokens = 0
            processing_cost = 0

            if self.debug and self.verbosity == 3:
                self.notice('Generating summary')
                self.info("\n")
                self.info(text)
                self.info("\n")
                self.info(dump_json(config, indent = 2))
                self.info("\n")
                self.info('-' * self.display_width)

            for index in range(summary_retries):
                result = self.submit(summary_channel, {
                    'text': text,
                    'config': config
                })
                request_tokens += result['prompt_tokens']
                response_tokens += result['output_tokens']
                processing_cost += result['cost']

                if result and check_ending(result['text'], summary_endings):
                    response_text = result['text']
                    break
                else:
                    response_text = 'Summary generation was unsuccessful. Please retry with a modified question.'

            if self.debug and self.verbosity == 3:
                self.info("\n")
                self.info(response_text)

            summary.result = response_text
            summary.request_tokens = request_tokens
            summary.response_tokens = response_tokens
            summary.cost = processing_cost
            summary.save()

            return Collection(
                text = response_text,
                request_tokens = request_tokens,
                response_tokens = response_tokens,
                total_tokens = (request_tokens + response_tokens),
                cost = processing_cost
            )

        return self.run_exclusive("ml:{}:{}".format(summary_channel, summary_id), generate)


    def exec_summary(self, provider = None):
        summarizer = self.get_summarizer(provider = provider)
        summary_key = provider if provider else 'summary'
        summary_channel = "agent:model:{}".format(summary_key)

        def parse_model_summary(text, config):
            return summarizer.summarize(text, **config).export()

        for package in self.listen(summary_channel, state_key = "model_{}".format(summary_key)):
            text = package.message['text']
            config = package.message.get('config', {})

            try:
                self.data("Processing {} request".format(summary_key), package.sender)
                response = self.profile(parse_model_summary, text, config)
                self.send(package.sender, response.result)

            except Exception as e:
                self.send(summary_channel, package.message, package.sender)
                raise e

            self.send("{}:stats".format(summary_channel), {
                'provider': summarizer.name,
                'time': response.time,
                'memory': response.memory
            })


    def shutdown(self):
        super().shutdown()
        if getattr(self, 'providers', None):
            with self.provider_lock:
                for provider in self.providers:
                    self._remove_model(
                        provider.provider_type,
                        provider.name,
                        provider.field_device
                    )


    def _get_model_provider(self, provider_type, providers, options):
        def get_provider():
            for provider_info in providers:
                if isinstance(provider_info, str):
                    provider_info = ( provider_info, 0 )
                if len(provider_info) == 1:
                    provider_info.append(0)

                provider_name = provider_info[0]

                for device_index in range(1, len(provider_info)):
                    max_count = provider_info[device_index]

                    if isinstance(max_count, str) and ':' in max_count:
                        max_count_components = max_count.split(':')
                        device_index = int(max_count_components[0])
                        max_count = int(max_count_components[1])
                    else:
                        device_index = (device_index - 1)
                        max_count = int(max_count)

                    device_name = "cuda:{}".format(device_index)
                    current_count = self._get_model_count(provider_type, provider_name, device_name) if max_count else 0

                    if not max_count or current_count < max_count:
                        return self.get_provider(
                            provider_type,
                            provider_name,
                            device = device_name,
                            **options
                        )
            return None

        return self.run_exclusive("{}_model_index".format(provider_type), get_provider)


    def _get_model_state_name(self, type, name, device):
        return "ml.{}.{}{}".format(type, name, ".{}".format(device) if device else '')

    def _get_model_count(self, type, name, device):
        name = self._get_model_state_name(type, name, device)

        def retrieve():
            return len(self.get_state(name, []))

        return self.run_exclusive(name, retrieve)

    def _add_model(self, type, name, device):
        name = self._get_model_state_name(type, name, device)

        def update():
            commands = self.get_state(name, [])
            if self.log_entry.name not in commands:
                commands.append(self.log_entry.name)
            return self.set_state(name, commands)

        return self.run_exclusive(name, update)

    def _remove_model(self, type, name, device):
        name = self._get_model_state_name(type, name, device)

        def update():
            commands = self.get_state(name, [])
            if self.log_entry.name in commands:
                commands.remove(self.log_entry.name)
            return self.set_state(name, commands)

        return self.run_exclusive(name, update)


    def parse_text_sections(self, text, cutoff_section_len = 10000):
        sections = []
        section = ''

        for text_item in ensure_list(text):
            for chunk in re.split(r'\n\n+', text_item.strip()):
                if chunk.strip():
                    combined_section = "{}\n{}".format(section, chunk).strip()
                    if len(combined_section) >= cutoff_section_len:
                        sections.append(combined_section)
                        section = ''
                    else:
                        section = combined_section

        if section:
            sections.append(section)

        return sections
