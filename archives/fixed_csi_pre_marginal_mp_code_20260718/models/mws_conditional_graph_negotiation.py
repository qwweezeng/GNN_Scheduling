"""Compatibility layer for the former MWS-CGN research name.

The canonical implementation is :mod:`models.marginal_mp`.  Historical names
remain importable so archived experiments and result reproduction do not break.
New code must use the MarginalMP names directly.
"""

from models.marginal_mp import (  # noqa: F401
    DirectLinkMarginalMP,
    GATAssistedMarginalMP,
    GATHardActionMarginalMP,
    GATAssistedJoinInitializedMarginalMP,
    JoinInitializedMarginalMP,
    MarginalMPScheduler,
    exact_link_toggle_marginals,
)


MWSConditionalGraphNegotiationScheduler = MarginalMPScheduler
PureInitializedMWSCGN = JoinInitializedMarginalMP
GATExpectedRateInitializedMWSCGN = GATAssistedJoinInitializedMarginalMP
GATHardSeedInitializedMWSCGN = GATHardActionMarginalMP
ConditionalMWSMessagePassingScheduler = JoinInitializedMarginalMP
exact_active_set_toggle_margins = exact_link_toggle_marginals
exact_physical_toggle_margins = exact_link_toggle_marginals


__all__ = [
    "MWSConditionalGraphNegotiationScheduler",
    "PureInitializedMWSCGN",
    "GATExpectedRateInitializedMWSCGN",
    "GATHardSeedInitializedMWSCGN",
    "ConditionalMWSMessagePassingScheduler",
    "exact_active_set_toggle_margins",
    "exact_physical_toggle_margins",
]
