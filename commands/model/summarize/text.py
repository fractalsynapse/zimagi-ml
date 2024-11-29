from systems.commands.index import Command
from systems.summary.text import TextSummarizer


class Text(Command('model.summarize.text')):

    def exec(self):
        summary = TextSummarizer(self, self.text, provider = 'mixtral_di_7bx8').generate(
            max_chunks = self.max_chunks,
            persona = self.persona.strip(),
            prompt = self.instruction.strip(),
            temperature = self.temperature,
            top_p = self.top_p,
            repetition_penalty = self.repetition_penalty
        )
        self.success(summary.text)
        summary.delete('text')
        self.data('Stats', summary.export())
