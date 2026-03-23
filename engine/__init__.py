"""
DeepResearch Engine — Level 2-3 Components

Level 2: Generative Mutations (add/remove/replace code)
Level 2.5: Curriculum Learning (progressive goals)
Level 3: Autonomous Engineer (spec → research → build → optimize)
"""

from engine.mutations import MutationManager, MutationProposal, MutationResult
from engine.mutations import MUTATION_TYPES, FeatureDiscovery
from engine.curriculum import CurriculumRunner
from engine.autonomous import Orchestrator, DomainResearcher, Architect, Bootstrapper, Component

__all__ = [
    "MutationManager", "MutationProposal", "MutationResult",
    "MUTATION_TYPES", "FeatureDiscovery",
    "CurriculumRunner",
    "Orchestrator", "DomainResearcher", "Architect", "Bootstrapper", "Component",
]
