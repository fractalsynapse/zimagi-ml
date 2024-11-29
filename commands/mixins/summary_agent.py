from systems.commands.index import CommandMixin


class SummaryAgentCommandMixin(CommandMixin('summary_agent')):

    def exec(self):
        self.exec_summary(self.model_provider)
