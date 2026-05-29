"""Template agent — copy this file to implement a custom Brain OS specialist."""

from __future__ import annotations

from brain_os.agents.base_agent import BaseAgent


class TemplateAgent(BaseAgent):
    name = "template"
    role = "Template"
    description = "Copy and rename to add a custom specialist agent."
    knowledge_categories: tuple[str, ...] = ("general",)

    async def handle(self, query: str, context: dict) -> str:
        return await self.run(query, context)
