"""
DeepResearch Engine — Level 1-3 Components

Level 1:   strategy.py (Thompson Sampling, annealing, population)
Level 1.5: engine/knowledge.py (external domain knowledge acquisition)
Level 2:   engine/mutations.py (generative mutations, safety rails)
Level 2.5: engine/curriculum.py (progressive goals)
Level 3:   engine/autonomous.py (spec → research → build → optimize)
"""

from engine.mutations import MutationManager, MutationProposal, MutationResult
from engine.mutations import MUTATION_TYPES, FeatureDiscovery
from engine.curriculum import CurriculumRunner
from engine.autonomous import Orchestrator, DomainResearcher, Architect, Bootstrapper, Component, ReportGenerator
from engine.knowledge import KnowledgeAcquisition, SearchStrategy, SourceManager, TechniqueLibrary

__all__ = [
    "MutationManager", "MutationProposal", "MutationResult",
    "MUTATION_TYPES", "FeatureDiscovery",
    "CurriculumRunner",
    "Orchestrator", "DomainResearcher", "Architect", "Bootstrapper", "Component", "ReportGenerator",
    "KnowledgeAcquisition", "SearchStrategy", "SourceManager", "TechniqueLibrary",
]
