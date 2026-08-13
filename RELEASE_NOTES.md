# v1.3.0 -- Bounded instance-support evaluation

This release accompanies *Bounded Instance-Support Evaluation for Finite
Crop--Weed Review Queues*.

It includes:

- a 27-contract sensitivity grid crossing instance coverage, candidate
  labelled coverage, and same-role purity;
- commissioned-only, original-only, union, and reciprocal ranker-source
  schemes at an equal 471-candidate test burden;
- ranker-free proposal ceilings, per-tile queue oracles, set coverage, maximum
  matching, and merge-capacity accounting;
- five U-Net seeds and three seeds each for DeepLabV3+ and Mask R-CNN, with
  crossed seed-by-image uncertainty and date-block sensitivity;
- COCO-style AP/AR and explicitly labelled queue-specific PQ-style diagnostics
  on the same predicted-weed regions;
- aggregate two-operator nominal and ordered agreement, confidence summaries,
  full contingencies, and human--mask precision, recall, F1, and Jaccard;
- the unchanged locked 60-image CWFID zero-candidate endpoint plus continuous
  post-lock score diagnostics;
- Figures 1--5 in PDF, SVG, TIFF, and PNG formats with source-data and
  text-bound audits;
- a rewritten manuscript, supplement, cover letter, and highlights; and
- a standard-library verifier for the v1.3.0 evidence layer.

Two superseded standard-metric attempts remain in `failures/`. The release does
not redistribute source imagery, dataset masks, public model weights, private
infrastructure records, credentials, internal addresses, operator identities,
or item-level operator responses. Full model training requires the cited
public datasets and registered upstream weights.
