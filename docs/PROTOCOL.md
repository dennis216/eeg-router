# Original H4 protocol

- Source namespace: `v3-groupkfold-confirmatory-v1`; five patient-held-out folds and seeds 17, 42, 3407.
- Candidate events are sorted by onset; a group uses the earliest onset as anchor with a 30-second maximum difference. This is not transitive chaining.
- When multiple events from one detector occur within a group, the highest-peak member is retained.
- Router scores candidate members, selects the highest score (peak probability breaks ties) and applies the validation-selected retention threshold.
- Emitted intervals start at the chosen member's onset and end at the group's maximum offset.
- Evaluation starts at the original metadata-defined first candidate end. Replacing this start or the group end changes the protocol.
- The original matching algorithm maximizes total interval overlap, then filters assignments at one-second overlap. It differs from maximum-cardinality-first matching.
- False alarms per 24 h use effective background hours. Conversion to FA/h divides by 24.
- Alarm minutes/24 h in the supplement use the union of predicted intervals outside original seizure intervals, divided by effective monitoring time, including overshoot of matched events.
- Main summary is an unweighted mean of 15 fold-seed metrics. Aggregate TP/truth division and patient means are separate summaries.
- Full oracle uses retrospective truth to select available actions and abstain. Forced-selection oracle emits an existing member for every group.

The diagnostic supplement's LL/LO/OL/OO conditions fix learned/oracle retained groups and learned/oracle member choice respectively. Their implementation remains subject to `STATUS.md`.
