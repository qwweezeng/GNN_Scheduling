"""Canonical initial-schedule constructors for MarginalMP experiments.

The public names in this module describe two independent design choices:

* ``DirectLink`` versus ``GATAssisted`` specifies how the queue-weighted
  link-quality score used by the initial LocalMax stage is obtained.
* ``Direct-Link Join`` is retained only as the single internal structural
  baseline.  It is not an initializer for MarginalMP.

Low-level implementations remain in ``models.local_physical_reuse`` so that
historical experiments keep working.  New paper code should import this file.
"""

from models.local_physical_reuse import (
    PurePhysicalReuseScheduler,
    QueueChannelLocalMaxScheduler,
)


class DirectLinkLocalMaxScheduler(QueueChannelLocalMaxScheduler):
    """Queue-weighted direct-link quality followed by one LocalMax round."""

    model_version = "direct_link_localmax_v1"

    def __init__(self):
        super().__init__(None, priority_mode="solo_rate")


class GATAssistedLocalMaxScheduler(QueueChannelLocalMaxScheduler):
    """GAT-assisted link quality followed by one LocalMax round."""

    model_version = "gat_assisted_localmax_v1"

    def __init__(self, channel_model):
        super().__init__(channel_model, priority_mode="gat_expected_rate")


class DirectLinkJoinScheduler(PurePhysicalReuseScheduler):
    """Direct-Link LocalMax followed by one add-only Join round."""

    model_version = "direct_link_join_v1"


__all__ = [
    "DirectLinkLocalMaxScheduler",
    "GATAssistedLocalMaxScheduler",
    "DirectLinkJoinScheduler",
]
