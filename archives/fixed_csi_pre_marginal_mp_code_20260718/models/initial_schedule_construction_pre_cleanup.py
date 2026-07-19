"""Canonical initial-schedule constructors for MarginalMP experiments.

The public names in this module describe two independent design choices:

* ``DirectLink`` versus ``GATAssisted`` specifies how the queue-weighted
  link-quality score used by the initial LocalMax stage is obtained.
* ``LocalMax`` versus ``Join`` specifies whether the one-shot LocalMax skeleton
  is returned directly or receives one additional add-only reuse round.

Low-level implementations remain in ``models.local_physical_reuse`` so that
historical experiments keep working.  New paper code should import this file.
"""

from models.local_physical_reuse import (
    GATExpectedRateLocalPhysicalReuseScheduler,
    GATHardSeedPhysicalReuseScheduler,
    PurePhysicalReuseScheduler,
    QueueChannelLocalMaxScheduler,
)


class DirectLinkLocalMaxScheduler(QueueChannelLocalMaxScheduler):
    """Queue-weighted direct-link quality followed by one LocalMax round."""

    model_version = "direct_link_localmax_v1"

    def __init__(self, channel_model):
        super().__init__(channel_model, priority_mode="solo_rate")


class GATAssistedLocalMaxScheduler(QueueChannelLocalMaxScheduler):
    """GAT-assisted link quality followed by one LocalMax round."""

    model_version = "gat_assisted_localmax_v1"

    def __init__(self, channel_model):
        super().__init__(channel_model, priority_mode="gat_expected_rate")


class DirectLinkJoinScheduler(PurePhysicalReuseScheduler):
    """Direct-Link LocalMax followed by one add-only Join round."""

    model_version = "direct_link_join_v1"


class GATAssistedJoinScheduler(GATExpectedRateLocalPhysicalReuseScheduler):
    """GAT-Assisted LocalMax followed by one add-only Join round."""

    model_version = "gat_assisted_join_v1"


class GATHardActionJoinScheduler(GATHardSeedPhysicalReuseScheduler):
    """Conference-GAT hard action followed by one add-only Join audit."""

    model_version = "gat_hard_action_join_audit_v1"


__all__ = [
    "DirectLinkLocalMaxScheduler",
    "GATAssistedLocalMaxScheduler",
    "DirectLinkJoinScheduler",
    "GATAssistedJoinScheduler",
    "GATHardActionJoinScheduler",
]
