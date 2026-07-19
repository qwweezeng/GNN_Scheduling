"""Compatibility checks for the MarginalMP naming migration."""

from models.marginal_mp import (
    DirectLinkMarginalMP,
    GATAssistedMarginalMP,
    JoinInitializedMarginalMP,
    MarginalMPScheduler,
    exact_link_toggle_marginals,
)
from models.mws_conditional_graph_negotiation import (
    GATExpectedRateInitializedMWSCGN,
    MWSConditionalGraphNegotiationScheduler,
    PureInitializedMWSCGN,
    exact_active_set_toggle_margins,
)


def test_legacy_names_resolve_to_canonical_marginal_mp():
    assert PureInitializedMWSCGN is JoinInitializedMarginalMP
    assert MWSConditionalGraphNegotiationScheduler is MarginalMPScheduler
    assert exact_active_set_toggle_margins is exact_link_toggle_marginals


def test_canonical_marginal_mp_default_budget_is_four_rounds():
    class _Initializer:
        def schedule_with_state(self, state, queues):
            raise AssertionError("not called by this configuration test")

    scheduler = MarginalMPScheduler(_Initializer())
    assert scheduler.rounds == 4
