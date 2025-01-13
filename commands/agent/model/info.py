from systems.commands.index import Agent
from utility.data import ensure_list

import statistics


class Info(Agent("model.info")):

    def exec(self):
        channel = "agent:model:info"

        for package in self.listen(channel, state_key="model_info"):
            model_provider = package.message["model"]
            model_fields = ensure_list(package.message["fields"])
            config = package.message.get("config", {})

            try:
                self.data("Processing token counting request", package.sender)
                response = self.profile(
                    self._get_model_info, model_provider, model_fields, config
                )
                self.send(package.sender, response.result)

            except Exception as e:
                self.send(channel, package.message, package.sender)
                raise e

            self.send(
                "{}:stats".format(channel),
                {
                    "provider": model_provider,
                    "fields": model_fields,
                    "time": response.time,
                    "memory": response.memory,
                },
            )

    def _get_model_info(self, provider, fields, config):
        summarizer = self.get_summarizer(provider=provider)
        info = {}

        if "token_count" in fields and "texts" in config:
            token_counts = []
            for text in ensure_list(config["texts"]):
                token_counts.append(summarizer.get_token_count(text))
            info["token_count"] = token_counts

        if "max_tokens" in fields:
            info["max_tokens"] = summarizer.get_chunk_length()

        if "prompt_tokens" in fields and "prompt" in config:
            info["prompt_tokens"] = summarizer.get_prompt_token_count(
                config["prompt"],
                config.get("persona", ""),
                config.get("format", ""),
            )

        return info
