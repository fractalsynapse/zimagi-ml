from django.conf import settings

from systems.commands.index import CommandMixin
from utility.data import Collection
from utility.web import WebParser

import billiard as multiprocessing
import re


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

    def parse_sentences(self, text):
        if not text:
            return []
        return self.submit('agent:model:sentence_parser', text)


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

    def generate_embeddings(self, sentences):
        if not sentences:
            return []
        return self.submit('agent:model:embedding', sentences)

    def generate_text_embeddings(self, text):
        text_data = Collection(sentences = [], embeddings = [])
        has_data = False

        for section in self.parse_text_sections(text.encode('ascii', 'ignore').decode().replace("\x00", '')):
            sentences = self.parse_sentences(section)
            if sentences:
                text_data.sentences.extend(sentences)
                text_data.embeddings.extend(self.generate_embeddings(sentences))
            has_data = True

        return text_data if has_data else None


    def get_summarizer(self, **options):
        with self.provider_lock:
            if not getattr(self, 'providers', None):
                self.providers = []

            provider = self._get_model_provider('summarizer', settings.SUMMARIZER_PROVIDERS, options)
            self._add_model(
                provider.provider_type,
                provider.name,
                provider.field_device
            )
            self.providers.append(provider)
        return provider

    def generate_summary(self, text, **config):
        if not text:
            return None
        return self.submit('agent:model:summary', {
                'text': text,
                'config': config
            }
        )


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

        for chunk in re.split(r'\n\n+', text.strip()):
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


    def parse_web_text(self, webpage_url):
        text = ''
        parser = WebParser(webpage_url)
        if parser.text:
            text = parser.text
        return text
