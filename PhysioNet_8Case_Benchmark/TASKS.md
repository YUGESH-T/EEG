# EEG Benchmarking Tasks

## Phase 1: Planning and scaffolding

- [ ] Create framework directory structure
- [ ] Create shared configuration builder
- [ ] Create case naming convention
- [ ] Create benchmark result directory convention
- [ ] Create shared logging and checkpoint helpers

## Phase 2: Shared data pipeline

- [ ] Implement PhysioNet downloader with retry
- [ ] Implement EDF cache validation and redownload
- [ ] Implement subject loader for `R04`, `R08`, `R12`
- [ ] Implement balanced left/right imagery trial extraction
- [ ] Implement subject-level data normalization hooks

## Phase 3: Filtering

- [ ] Implement common band-pass filter mode
- [ ] Implement Butterworth band-pass filter mode
- [ ] Ensure filtering stage is identical across all pipelines

## Phase 4: Feature extraction

- [ ] Implement CSP feature extraction
- [ ] Implement Riemannian covariance extraction
- [ ] Implement correct tangent-space mapping
- [ ] Implement optional ANOVA preselection for Riemannian cases

## Phase 5: HO optimization

- [ ] Implement reusable HO optimizer
- [ ] Implement SVM fitness mode
- [ ] Implement RF fitness mode
- [ ] Implement binary feature mask thresholding
- [ ] Implement zero-feature fallback handling

## Phase 6: Classifiers and evaluation

- [ ] Implement SVM training/evaluation
- [ ] Implement RF training/evaluation
- [ ] Implement per-subject metric computation
- [ ] Implement selected feature count reporting
- [ ] Implement convergence tracking

## Phase 7: Experiment orchestration

- [ ] Implement smoke test stage on 3 subjects
- [ ] Implement pilot stage on 10 subjects
- [ ] Implement full stage on 109 subjects
- [ ] Implement resume-safe case execution

## Phase 8: Reporting outputs

- [ ] Write per-case CSV files
- [ ] Write master summary CSV
- [ ] Build Excel workbook
- [ ] Save confusion matrix figures
- [ ] Save convergence plots
- [ ] Compute overall summary statistics
- [ ] Compute pairwise comparisons

## Phase 9: Final documentation

- [ ] Generate final Markdown report
- [ ] Export PDF report
- [ ] Write interpretation notes
- [ ] Summarize best-performing case
- [ ] Summarize limitations and risks
