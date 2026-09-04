"""Adaptive enterprise RAG control plane.

The package deliberately keeps benchmark labels out of routing and online
control decisions. Gold documents and answer facts are only consumed by the
separate offline attribution/evaluation utilities.
"""

from .budget import BudgetDecision, DynamicEvidenceBudget
from .features import AdaptiveRouter, RouterDecision, RouterFeatures
from .requirements import Requirement, RequirementPlan, RequirementMapper

__all__ = [
    "AdaptiveRouter",
    "BudgetDecision",
    "DynamicEvidenceBudget",
    "Requirement",
    "RequirementMapper",
    "RequirementPlan",
    "RouterDecision",
    "RouterFeatures",
]
