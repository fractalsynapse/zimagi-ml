from systems.plugins.index import ProviderMixin


class LlamaSummaryMixin(ProviderMixin('llama_summary')):

    def _get_prompt(self, text, prompt = '', persona = '', output_format = ''):
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

Do not include external links.
{output_format}
[/INST]
        """.format(
            persona = persona.strip(),
            instruction = prompt.strip(),
            text = text.strip(),
            output_format = output_format.strip()
        ).strip()

    def _parse_summary_response(self, summary):
        return summary.strip()