# EEG Benchmarking Framework Plan

## Objective

Build a reproducible MATLAB benchmarking framework to compare 8 EEG motor imagery pipelines across all 109 PhysioNet EEGMMIDB subjects. Each configuration must use identical data handling, trial extraction, normalization, evaluation protocol, and HO settings while varying only:

- feature extraction: `csp` vs `riemannian`
- classifier: `svm` vs `rf`
- filter type: `common` vs `butterworth`

## Experiment Matrix

1. CSP + SVM + HO + common band-pass
2. Riemannian + SVM + HO + common band-pass
3. CSP + RF + HO + common band-pass
4. Riemannian + RF + HO + common band-pass
5. CSP + SVM + HO + Butterworth band-pass
6. Riemannian + SVM + HO + Butterworth band-pass
7. CSP + RF + HO + Butterworth band-pass
8. Riemannian + RF + HO + Butterworth band-pass

## Architecture Decisions

### 1. Config-driven framework

Use one master runner with a MATLAB config struct:

- `cfg.feature_type`
- `cfg.classifier`
- `cfg.filter_type`
- `cfg.subject_ids`
- `cfg.seed`
- `cfg.ho`
- `cfg.eval`

Reason:
This avoids 8 divergent scripts and keeps fairness constraints enforceable.

### 2. Shared data ingestion

Use one PhysioNet subject loader for all experiments:

- download `R04`, `R08`, `R12`
- extract `T1` and `T2`
- convert to balanced left/right 4-second trials
- retry on failure
- redownload corrupt cached EDF files

Reason:
All pipelines must start from identical subject data.

### 3. Shared preprocessing stage

Apply filtering after converting data to `channels x time x trials`.

Reason:
This makes filter comparisons fair and keeps CSP and Riemannian pipelines aligned.

### 4. Separate feature extractors

- `extract_csp_features.m`
- `extract_riemannian_features.m`

Reason:
Feature extraction is the intended experimental variable and should be isolated cleanly.

### 5. HO wrapper matched to final classifier

HO fitness will use the same classifier family as the final model:

- SVM cases -> SVM fitness
- RF cases -> RF fitness

Reason:
This is required for a clean experimental comparison.

### 6. Resume-safe result storage

Each case writes one per-subject CSV with upsert logic.

Reason:
Full 109-subject x 8-case execution is a long batch job and must recover after interruption.

### 7. Staged execution

- Stage A: smoke test on 3 subjects
- Stage B: pilot on 10 subjects
- Stage C: full 109-subject run

Reason:
Large MATLAB experiments fail expensively if not validated early.

## Folder Structure

Planned output layout:

- `benchmark_framework/`
  - `run_benchmark_stage.m`
  - `build_case_configs.m`
  - `functions/`
- `benchmark_results/`
  - `stage_smoke/`
  - `stage_pilot/`
  - `stage_full/`
  - `reports/`
  - `figures/`

## Evaluation Protocol

Keep fixed across all cases:

- subject set
- trial extraction
- label mapping
- normalization rules
- random seed
- train/test protocol
- HO iterations and population size
- 8-30 Hz target band

Per-subject metrics:

- accuracy
- precision
- recall
- F1-score
- selected feature count
- HO best score

Overall metrics:

- mean accuracy
- std accuracy
- median accuracy
- best subject
- worst subject
- subject counts above 70%, 80%, 90%

## Statistical Comparison

Planned comparisons:

- CSP vs Riemannian under same classifier and filter
- SVM vs RF under same feature extractor and filter
- common vs Butterworth under same feature extractor and classifier

Primary outputs:

- overall summary CSV
- Excel workbook
- pairwise difference tables
- interpretation notes in Markdown and PDF

## Assumptions

1. PhysioNet EEGMMIDB runs `R04`, `R08`, `R12` are the intended left/right imagery source.
2. `T1` and `T2` labels map to the two target imagery classes consistently for this project.
3. MATLAB R2026a has the needed toolboxes for:
   - `edfread`, `edfinfo`
   - signal processing functions
   - `fitcsvm`
   - `fitcensemble`
4. Using 4-second windows after event onset is acceptable for all 109 subjects.
5. The "common band-pass" definition can be represented by a stable `designfilt('bandpassiir', ...)` implementation unless you later specify a different exact design.
6. Riemannian pipelines should use the improved tangent-space mapping to avoid invalid geometry.

## Risks

1. Runtime risk:
   8 cases x 109 subjects is a large computation and may require long MATLAB sessions.

2. Data inconsistency risk:
   Some PhysioNet EDF files may download partially or be corrupted in cache.

3. HO cost risk:
   Riemannian feature spaces are large; without dimensionality control they may become slow or unstable.

4. Metric instability risk:
   Subject-level trial counts are small, so holdout metrics can vary. This must be documented.

5. Report generation risk:
   Producing both Markdown and PDF from MATLAB results may require a two-step export path.

## Implementation Phases

### Phase 1

- scaffold framework
- implement shared loader, logger, checkpointing, configs

### Phase 2

- implement filters
- implement CSP and Riemannian feature extractors
- implement HO wrapper with classifier-matched fitness

### Phase 3

- implement subject execution loop
- write per-case CSVs
- add figures and summaries

### Phase 4

- run smoke test
- fix issues

### Phase 5

- run pilot
- validate runtime and outputs

### Phase 6

- run full 109-subject batch
- generate final workbook, summary CSV, Markdown report, and PDF
