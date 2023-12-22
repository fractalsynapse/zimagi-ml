from systems.plugins.index import ProviderMixin


class MixtralSummaryMixin(ProviderMixin('mixtral_summary')):

    def _get_prompt(self, text, prompt = '', persona = ''):
        return """
<s>[INST] <<SYS>>
{persona}
You always produce factually correct information from the text provided.
You do not ask questions.
<</SYS>>

Consider the following text for further questions and instructions:
{text}
[/INST]
I will refer to the provided text when following instructions
</s>
<s>[INST]
{instruction}
[/INST]
        """.format(
            persona = persona.strip(),
            instruction = prompt.strip(),
            text = text.strip()
        )

    def _parse_summary_response(self, summary):
        return summary.strip().strip("\'\"")
