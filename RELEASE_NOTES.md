# v1.3.1 -- Corrected bounded instance-support evaluation

This release accompanies *Bounded Instance-Support Evaluation for Finite
Crop--Weed Review Queues* and supersedes v1.3.0 for manuscript citation.

The evidence layer preserves the analyses introduced in v1.3.0 and makes two
linked corrections:

- human--mask binary correspondence is evaluated on the 444 regions for which
  both operators reached a determinate yes/no consensus; the 74 disputed or
  uncertain regions remain visible in the complete three-by-two contingency
  table and do not enter binary denominators; and
- one-to-one support and set coverage are defined as threshold-conditioned
  geometric graph endpoints at fixed queue capacity, not as lower or upper
  bounds on human plant recovery.

The corrected human--mask values are precision 0.7880, recall 0.8272, F1
0.8071, Jaccard 0.6766, and accuracy 0.7320. The release updates the analysis
script, compact verifier, manuscript, supplement, Figure 1, and source data in
one versioned layer. All other registered numerical endpoints are unchanged.

The archive also contains the 27-contract sensitivity grid, four ranker-source
schemes, crossed seed-by-image uncertainty, standard AP/AR and queue-specific
diagnostics, aggregate operator agreement, the locked CWFID endpoint, five
figures with source data, and preserved failed runs.

The release does not redistribute source imagery, dataset masks, public model
weights, private infrastructure records, credentials, internal addresses,
operator identities, or item-level operator responses. Full model training
requires the cited public datasets and registered upstream weights.
