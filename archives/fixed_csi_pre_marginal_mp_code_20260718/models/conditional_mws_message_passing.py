"""Compatibility imports for two former MarginalMP research names."""

from models.mws_conditional_graph_negotiation import (  # noqa: F401
    ConditionalMWSMessagePassingScheduler,
    GATExpectedRateInitializedMWSCGN,
    GATHardSeedInitializedMWSCGN,
    MWSConditionalGraphNegotiationScheduler,
    PureInitializedMWSCGN,
    exact_active_set_toggle_margins,
    exact_physical_toggle_margins,
)

__all__ = [
    "MWSConditionalGraphNegotiationScheduler",
    "PureInitializedMWSCGN",
    "GATExpectedRateInitializedMWSCGN",
    "GATHardSeedInitializedMWSCGN",
    "ConditionalMWSMessagePassingScheduler",
    "exact_active_set_toggle_margins",
    "exact_physical_toggle_margins",
]
