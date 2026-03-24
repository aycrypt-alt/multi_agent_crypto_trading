"""
Agent Registry — Manages agent lifecycle and discovery.

Supports registering thousands of agents, grouped by category.
Provides lookup by name, type, or tag.
"""

import logging
from collections import defaultdict

from .agent_base import Agent, AgentState

logger = logging.getLogger(__name__)


class AgentRegistry:
    """Central registry for all agents in the system."""

    def __init__(self):
        self._agents: dict[str, Agent] = {}           # id -> agent
        self._by_name: dict[str, Agent] = {}           # name -> agent
        self._by_category: dict[str, list[Agent]] = defaultdict(list)
        self._tags: dict[str, set[str]] = defaultdict(set)  # tag -> set of ids

    def register(self, agent: Agent, category: str = "general", tags: list[str] | None = None):
        self._agents[agent.id] = agent
        self._by_name[agent.name] = agent
        self._by_category[category].append(agent)
        for tag in (tags or []):
            self._tags[tag].add(agent.id)
        logger.info(f"Registered agent: {agent.name} [{category}]")

    def unregister(self, agent_id: str):
        agent = self._agents.pop(agent_id, None)
        if agent:
            self._by_name.pop(agent.name, None)
            for cat_agents in self._by_category.values():
                cat_agents[:] = [a for a in cat_agents if a.id != agent_id]
            for tag_set in self._tags.values():
                tag_set.discard(agent_id)

    def get(self, agent_id: str) -> Agent | None:
        return self._agents.get(agent_id)

    def get_by_name(self, name: str) -> Agent | None:
        return self._by_name.get(name)

    def get_by_category(self, category: str) -> list[Agent]:
        return self._by_category.get(category, [])

    def get_by_tag(self, tag: str) -> list[Agent]:
        return [self._agents[aid] for aid in self._tags.get(tag, set()) if aid in self._agents]

    def get_active(self) -> list[Agent]:
        return [a for a in self._agents.values() if a.state == AgentState.RUNNING]

    @property
    def count(self) -> int:
        return len(self._agents)

    def get_summary(self) -> dict:
        by_state = defaultdict(int)
        for a in self._agents.values():
            by_state[a.state.value] += 1
        return {
            "total_agents": self.count,
            "by_state": dict(by_state),
            "categories": {k: len(v) for k, v in self._by_category.items()},
        }
