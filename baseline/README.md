# Baseline layout

This directory has two deliberately separate layers.

## Project adapters used by current experiments

`queue_schedulers.py` provides schedulers with the common interface

```python
schedule_with_state(state, queues)
```

The active queue adapters are:

- `HardConflictLocalGreedyScheduler`: Zhao-style distributed LGS adapted to
  the project's symmetrized hard conflict graph;
- `CentralizedMWSGreedyScheduler`: global-CSI add/delete greedy reference,
  which is strong but is not a mathematical upper bound.
- `RandomNonemptyScheduler`: independent random activation of nonempty links;
- `WeightedFPLinQScheduler`: queue-weighted centralized FPLinQ adapter.  It
  fails loudly when the retained third-party environment is unavailable.

The internal `Direct-Link Join` baseline is defined next to the proposed model
in `models/initial_schedule_construction.py`; it is not a third-party baseline.

`distgcn_adapter.py` currently builds the symmetric hard graph and
$Q_iC_i^{solo}$ weights expected by the third-party DistGCN problem.  DistGCN
results are not publication-ready until TensorFlow checkpoint execution and
link-index mapping are verified end to end; Hard-Conflict LGS must not be
renamed as DistGCN.

## Third-party source trees

`distgcn-main/`, `Spatial_Deep_Learning_for_Wireless_Scheduling-master/`, and
the other imported repositories are preserved as external reference code.
Current experiments do not edit their internals; project-specific input and
output conversion belongs in an adapter or in `queue_schedulers.py`.

The authoritative method definitions are documented in `docs/SYSTEM_MODEL.md`.
The adaptation, legacy dependency, checkpoint, retraining, fairness, and
deadline-priority decisions for every baseline are consolidated in
`docs/EXPERIMENTS.md` Section 7.
