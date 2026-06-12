function results = run_csp_catboost(mode, cfg_override)
% RUN_CSP_CATBOOST
% Isolated experimental branch: preprocessing -> CSP -> CatBoost
%
% This pipeline is a clean, minimal branch that:
%   - Uses ONLY the existing preprocessing (Notch, Butterworth BP, CAR, Epoching)
%   - Uses ONLY CSP feature extraction (with configurable component counts)
%   - Uses ONLY CatBoost classification (via Python bridge)
%   - Completely bypasses: HO, PCA, feature fusion, temporal, spectral,
%     Riemannian, SVM, and ensemble logic
%
% Usage:
%   results = run_csp_catboost('smoke');               % PhysioNet S001-S003
%   results = run_csp_catboost('pilot10');             % PhysioNet S001-S010
%   results = run_csp_catboost('physionet109');        % PhysioNet all 109
%   results = run_csp_catboost('cho_smoke');           % Cho2017 S01
%   results = run_csp_catboost('cho_pilot5');          % Cho2017 S01-S05
%   results = run_csp_catboost('cho_full52');          % Cho2017 all 52
%
% CSP component sweep:
%   cfg.features.csp_components = 14;
%   results = run_csp_catboost('smoke', cfg);

if nargin < 1, mode = 'smoke'; end
if nargin < 2, cfg_override = struct(); end

cfg = merge_config(default_config(), cfg_override);
rng(cfg.seed, 'twister');

mode = lower(string(mode));

fbcsp_label = 'OFF';
if isfield(cfg.features, 'use_fbcsp') && cfg.features.use_fbcsp
    fbcsp_label = sprintf('ON (%d bands)', size(cfg.features.fbcsp_bands, 1));
end
fprintf('\n======================================================\n');
fprintf('  CSP + CatBoost Pipeline (Isolated Branch)\n');
fprintf('  Mode: %s | CSP components: %d | FBCSP: %s\n', ...
    mode, cfg.features.csp_components, fbcsp_label);
fprintf('======================================================\n\n');

if startsWith(mode, "cho")
    results = run_cho2017_batch(mode, cfg);
else
    results = run_physionet_batch(mode, cfg);
end

end

%% ======================================================================
%  DEFAULT CONFIG
%  ======================================================================

function cfg = default_config()

    cfg = struct();
    cfg.pipeline_type = 'csp_catboost';
    cfg.seed = 42;
    cfg.fs = 160;

    project_root = fileparts(fileparts(mfilename('fullpath')));

    % Dataset
    cfg.dataset.cache_dir = fullfile(project_root, 'physionet_cache');
    cfg.dataset.cho_dir = fullfile(project_root, 'gigadb');
    cfg.dataset.run_ids = [4 8 12];
    cfg.dataset.subjects = 1:109;
    cfg.dataset.min_balanced_trials = 10;
    cfg.dataset.download_if_missing = false;
    cfg.dataset.channel_indices = [];
    cfg.channel_mode = 'all';

    % Results
    cfg.results.output_dir = fullfile( ...
        fileparts(mfilename('fullpath')), 'results_csp_catboost');
    cfg.results.csv_file = fullfile( ...
        cfg.results.output_dir, 'csp_catboost_results.csv');
    cfg.results.mat_file = fullfile( ...
        cfg.results.output_dir, 'csp_catboost_results.mat');
    cfg.results.resume = true;

    % Preprocessing
    cfg.preprocessing.notch_freqs = 50;
    cfg.preprocessing.notch_quality_factor = 35;
    cfg.preprocessing.bandpass = [4 32];
    cfg.preprocessing.butterworth_order = 4;
    cfg.preprocessing.tmin = 0.5;
    cfg.preprocessing.tmax = 3.5;

    % CSP Features — the only feature extraction used
    cfg.features.csp_components = 14;  % Higher count per guide recommendation
    cfg.features.use_fbcsp = false;    % Set true to enable 3-band FBCSP
    cfg.features.fbcsp_bands = [8 12; 12 20; 20 30];  % mu, low-beta, high-beta
    cfg.features.csp_regularization = 1e-3;  % Diagonal regularization for CSP covariance
    cfg.features.use_feature_selection = false;  % Set true to rank-and-select top-K
    cfg.features.feature_selection_method = 'mutual_information';  % 'fisher' or 'mutual_information'
    cfg.features.top_k_features = 6;  % Number of features to keep after selection
    cfg.features.use_spectral = false; % Extract spectral features (bandpower)
    cfg.features.use_temporal = false; % Extract temporal features (mean, var, RMS, ZCR, Hjorth)
    cfg.features.use_ho = false;       % Extract higher-order features (skewness, kurtosis)

    % Classification (CatBoost only)
    cfg.classification.cv_folds = 5;
    cfg.classification.catboost_iterations = 200;
    cfg.classification.catboost_depth = 6;
    cfg.classification.catboost_learning_rate = 0.08;
    cfg.classification.catboost_verbose = false;

end

%% ======================================================================
%  PHYSIONET BATCH RUNNER
%  ======================================================================

function results = run_physionet_batch(mode, cfg)

    switch mode
        case "smoke",        subject_ids = 1:3;
        case "pilot10",      subject_ids = 1:10;
        case "physionet1",   subject_ids = 1;
        case "physionet109", subject_ids = cfg.dataset.subjects;
        otherwise
            error('Unknown PhysioNet mode: %s', mode);
    end

    ensure_directory(cfg.results.output_dir);
    completed = existing_completed_subjects(cfg.results.csv_file);

    fprintf('Dataset: PhysioNet | Subjects: %d | CSP components: %d\n', ...
        numel(subject_ids), cfg.features.csp_components);

    for subject_id = subject_ids

        subject_name = sprintf('s%03d', subject_id);

        if cfg.results.resume && any(completed == subject_id)
            fprintf('Skipping %s (already completed)\n', subject_name);
            continue;
        end

        row = init_result_row(subject_id, subject_name, 'physionet', cfg);

        try
            fprintf('\nProcessing %s\n', subject_name);
            [raw_eeg, events, subject_info] = ...
                load_physionet_subject(subject_id, cfg);
            subject_cfg = cfg;
            subject_cfg.fs = subject_info.fs;

            subj_results = run_subject_case(raw_eeg, events, subject_cfg);

            row.status = "OK";
            row.n_trials = subject_info.n_trials;
            row.n_channels = size(raw_eeg, 1);
            row.accuracy = subj_results.overall.accuracy;
            row.precision_val = subj_results.overall.precision;
            row.recall_val = subj_results.overall.recall;
            row.f1 = subj_results.overall.f1;
            row.csp_feature_dim = subj_results.csp_feature_dim;
        catch ME
            row.status = "ERROR";
            row.error_message = string(ME.message);
            fprintf('  ERROR: %s\n', ME.message);
        end

        append_result_row(cfg.results.csv_file, row);
    end

    results = finalize_results(mode, cfg);
end

%% ======================================================================
%  CHO2017 BATCH RUNNER
%  ======================================================================

function results = run_cho2017_batch(mode, cfg)

    cfg.fs = 512;
    cfg.preprocessing.bandpass = [8 30];

    switch mode
        case "cho_smoke",   subject_ids = 1;
        case "cho_pilot5",  subject_ids = 1:5;
        case "cho_full52",  subject_ids = 1:52;
        otherwise
            error('Unknown Cho2017 mode: %s', mode);
    end

    % Adjust result paths for Cho dataset
    cfg.results.csv_file = fullfile( ...
        cfg.results.output_dir, 'csp_catboost_cho2017_results.csv');
    cfg.results.mat_file = fullfile( ...
        cfg.results.output_dir, 'csp_catboost_cho2017_results.mat');

    ensure_directory(cfg.results.output_dir);
    completed = existing_completed_subjects(cfg.results.csv_file);

    fprintf('Dataset: Cho2017 | Subjects: %d | CSP components: %d\n', ...
        numel(subject_ids), cfg.features.csp_components);

    for subject_id = subject_ids

        subject_name = sprintf('cho_s%02d', subject_id);

        if cfg.results.resume && any(completed == subject_id)
            fprintf('Skipping %s (already completed)\n', subject_name);
            continue;
        end

        row = init_result_row(subject_id, subject_name, 'cho2017', cfg);

        try
            fprintf('\nProcessing %s\n', subject_name);
            mat_file = fullfile( ...
                cfg.dataset.cho_dir, sprintf('s%02d.mat', subject_id));
            if ~exist(mat_file, 'file')
                error('Cho2017 file not found: %s', mat_file);
            end

            loaded = load(mat_file);
            eeg_struct = loaded.eeg;

            trial_len = 3584;
            left_trials = reshape( ...
                eeg_struct.imagery_left, 68, trial_len, eeg_struct.n_imagery_trials);
            right_trials = reshape( ...
                eeg_struct.imagery_right, 68, trial_len, eeg_struct.n_imagery_trials);

            all_epochs = cat(3, left_trials, right_trials);
            all_labels = [ ...
                ones(eeg_struct.n_imagery_trials, 1); ...
                2 * ones(eeg_struct.n_imagery_trials, 1)];

            fs_cho = eeg_struct.srate;
            start_sample = round((0.5 + 2.0) * fs_cho);
            end_sample = round((4.0 + 2.0) * fs_cho);
            all_epochs = double(all_epochs(:, start_sample:end_sample, :));

            subject_cfg = cfg;
            subject_cfg.fs = fs_cho;

            subj_results = run_subject_case_from_epochs( ...
                all_epochs, all_labels, subject_cfg);

            row.status = "OK";
            row.n_trials = numel(all_labels);
            row.n_channels = size(all_epochs, 1);
            row.accuracy = subj_results.overall.accuracy;
            row.precision_val = subj_results.overall.precision;
            row.recall_val = subj_results.overall.recall;
            row.f1 = subj_results.overall.f1;
            row.csp_feature_dim = subj_results.csp_feature_dim;
        catch ME
            row.status = "ERROR";
            row.error_message = string(ME.message);
            fprintf('  ERROR: %s\n', ME.message);
        end

        append_result_row(cfg.results.csv_file, row);
    end

    results = finalize_results(mode, cfg);
end

%% ======================================================================
%  CORE SUBJECT PIPELINE (from raw EEG)
%  ======================================================================

function results = run_subject_case(raw_eeg, events, cfg)
% Preprocessing -> Epoching -> CSP -> CatBoost
% NO HO. NO PCA. NO fusion. NO spectral/temporal/riemannian.

    [epochs, labels] = preprocess_pipeline(raw_eeg, events, cfg);
    results = run_subject_case_from_epochs(epochs, labels, cfg);
end

function results = run_subject_case_from_epochs(epochs, labels, cfg)
% Core cross-validation loop: CSP features -> CatBoost directly.

    labels = labels(:);
    valid_mask = labels > 0;
    epochs = epochs(:, :, valid_mask);
    labels = labels(valid_mask);

    cv = cvpartition(labels, 'KFold', cfg.classification.cv_folds);
    fold_metrics = struct([]);
    all_true = [];
    all_pred = [];
    csp_feature_dim = 0;

    for fold_idx = 1:cv.NumTestSets
        fprintf('  Fold %d/%d\n', fold_idx, cv.NumTestSets);

        train_idx = training(cv, fold_idx);
        test_idx = test(cv, fold_idx);
        X_train = epochs(:, :, train_idx);
        y_train = labels(train_idx);
        X_test = epochs(:, :, test_idx);
        y_test = labels(test_idx);

        % ---- CSP Feature Extraction (Base path) ----
        if isfield(cfg.features, 'use_fbcsp') && cfg.features.use_fbcsp
            % FBCSP path remains unchanged for now; returns concatenated log-variance
            [csp_train, csp_test] = extract_minimal_fbcsp_features( ...
                X_train, y_train, X_test, cfg);
            Z_train = []; % FBCSP raw signal extraction not supported yet
            Z_test = [];
        else
            [csp_train, csp_test, Z_train, Z_test] = extract_plain_csp_features( ...
                X_train, y_train, X_test, cfg);
        end

        % ---- Secondary Feature Extraction ----
        train_blocks = {csp_train};
        test_blocks = {csp_test};
        
        active_features_str = "CSP";

        if isfield(cfg.features, 'use_spectral') && cfg.features.use_spectral && ~isempty(Z_train)
            train_blocks{end+1} = extract_spectral_features(Z_train, cfg.fs);
            test_blocks{end+1}  = extract_spectral_features(Z_test, cfg.fs);
            active_features_str = active_features_str + " + Spectral";
        end
        if isfield(cfg.features, 'use_temporal') && cfg.features.use_temporal && ~isempty(Z_train)
            train_blocks{end+1} = extract_temporal_features(Z_train);
            test_blocks{end+1}  = extract_temporal_features(Z_test);
            active_features_str = active_features_str + " + Temporal";
        end
        if isfield(cfg.features, 'use_ho') && cfg.features.use_ho && ~isempty(Z_train)
            train_blocks{end+1} = extract_ho_features(Z_train);
            test_blocks{end+1}  = extract_ho_features(Z_test);
            active_features_str = active_features_str + " + HO";
        end

        % ---- Feature Fusion ----
        fused_train = fuse_feature_blocks(train_blocks);
        fused_test  = fuse_feature_blocks(test_blocks);

        if fold_idx == 1
            fprintf('    Active Features: %s\n', active_features_str);
            fprintf('    Fused Feature Matrix Size: [%d, %d]\n', size(fused_train, 1), size(fused_train, 2));
        end

        % Z-score standardize
        mu = mean(fused_train, 1);
        sigma = std(fused_train, [], 1) + 1e-8;
        X_train_final = (fused_train - mu) ./ sigma;
        X_test_final = (fused_test - mu) ./ sigma;

        % ---- Optional Feature Selection (rank + subset, no projection) ----
        if isfield(cfg.features, 'use_feature_selection') && cfg.features.use_feature_selection
            [X_train_final, X_test_final, ~] = select_top_k_features( ...
                X_train_final, y_train, X_test_final, ...
                cfg.features.feature_selection_method, ...
                cfg.features.top_k_features);
        end

        csp_feature_dim = size(X_train_final, 2);

        % ---- CatBoost Classification (ONLY classifier) ----
        y_pred = classify_with_catboost( ...
            X_train_final, y_train, X_test_final, cfg);

        metrics = compute_metrics(y_test, y_pred);

        fold_metrics(fold_idx).accuracy = metrics.accuracy;
        fold_metrics(fold_idx).precision = metrics.precision;
        fold_metrics(fold_idx).recall = metrics.recall;
        fold_metrics(fold_idx).f1 = metrics.f1;

        all_true = [all_true; y_test]; %#ok<AGROW>
        all_pred = [all_pred; y_pred(:)]; %#ok<AGROW>

        fprintf('    Acc: %.2f%% | CSP dims: %d\n', ...
            metrics.accuracy * 100, csp_feature_dim);
    end

    results.overall = compute_metrics(all_true, all_pred);
    results.fold_metrics = fold_metrics;
    results.csp_feature_dim = csp_feature_dim;
    results.true_labels = all_true;
    results.predicted_labels = all_pred;

    fprintf('  Subject Accuracy: %.2f%% | F1: %.4f\n', ...
        results.overall.accuracy * 100, results.overall.f1);
end

%% ======================================================================
%  PREPROCESSING (reused from validated pipeline, standalone copy)
%  ======================================================================

function [epochs, labels] = preprocess_pipeline(raw_eeg, events, cfg)

    fs = cfg.fs;
    data = apply_channel_mode(double(raw_eeg), cfg);

    % Notch filter
    for notch_freq = cfg.preprocessing.notch_freqs
        if notch_freq > 0 && notch_freq < fs / 2
            bandwidth = notch_freq / cfg.preprocessing.notch_quality_factor;
            low_freq = max(notch_freq - bandwidth / 2, 0.1);
            high_freq = min(notch_freq + bandwidth / 2, fs / 2 - 0.1);
            [b, a] = butter(2, [low_freq high_freq] / (fs / 2), 'stop');
            data = filtfilt(b, a, data')';
        end
    end

    % Butterworth bandpass
    [b, a] = butter( ...
        cfg.preprocessing.butterworth_order, ...
        cfg.preprocessing.bandpass / (fs / 2), 'bandpass');
    data = filtfilt(b, a, data')';

    % CAR (Common Average Reference — dedicated helper)
    data = apply_car_reference(data);

    % Epoching
    tmin_samples = round(cfg.preprocessing.tmin * fs);
    tmax_samples = round(cfg.preprocessing.tmax * fs);
    epoch_len = tmax_samples - tmin_samples;
    n_events = size(events, 1);
    n_channels = size(data, 1);

    epochs = zeros(n_channels, epoch_len, n_events);
    labels = zeros(n_events, 1);
    keep = false(n_events, 1);

    for event_idx = 1:n_events
        start_idx = events(event_idx, 1) + tmin_samples;
        stop_idx = start_idx + epoch_len - 1;
        if start_idx >= 1 && stop_idx <= size(data, 2)
            epoch = data(:, start_idx:stop_idx);
            epoch = epoch - mean(epoch, 2);
            epochs(:, :, event_idx) = epoch;
            labels(event_idx) = events(event_idx, 2);
            keep(event_idx) = true;
        end
    end

    epochs = epochs(:, :, keep);
    labels = labels(keep);
end

function data = apply_channel_mode(data, cfg)
    if ~isfield(cfg, 'channel_mode') || ...
            lower(string(cfg.channel_mode)) == "all"
        return;
    end

    channel_labels = configured_channel_labels(size(data, 1));
    normalized = lower(regexprep(string(channel_labels(:)), '[^A-Za-z0-9]', ''));

    switch lower(string(cfg.channel_mode))
        case "sensorimotor"
            targets = lower(["FC5","FC3","FC1","FCz","FC2","FC4","FC6", ...
                "C5","C3","C1","Cz","C2","C4","C6", ...
                "CP5","CP3","CP1","CPz","CP2","CP4","CP6"]);
        case "motor_extended"
            targets = lower(["FC3","FC1","FCz","FC2","FC4", ...
                "C3","C1","Cz","C2","C4", ...
                "CP3","CP1","CPz","CP2","CP4"]);
        otherwise
            return;
    end

    idx = [];
    for k = 1:numel(targets)
        hit = find(normalized == targets(k), 1);
        if ~isempty(hit), idx(end+1) = hit; end %#ok<AGROW>
    end
    data = data(sort(idx), :);
end

function labels = configured_channel_labels(n)
    standard64 = ["FC5","FC3","FC1","FCz","FC2","FC4","FC6", ...
        "C5","C3","C1","Cz","C2","C4","C6", ...
        "CP5","CP3","CP1","CPz","CP2","CP4","CP6", ...
        "FP1","FPz","FP2","AF7","AF3","AFz","AF4","AF8", ...
        "F7","F5","F3","F1","Fz","F2","F4","F6","F8", ...
        "FT7","FT8","T7","T8","T9","T10","TP7","TP8", ...
        "P7","P5","P3","P1","Pz","P2","P4","P6","P8", ...
        "PO7","PO3","POz","PO4","PO8","O1","Oz","O2","Iz"];
    if n == 64,     labels = standard64;
    elseif n == 68, labels = [standard64, "AUX1","AUX2","AUX3","AUX4"];
    else,           labels = arrayfun(@(x) sprintf("Ch%d",x), 1:n);
    end
end

%% ======================================================================
%  CSP FEATURE EXTRACTION (plain single-band CSP)
%  ======================================================================

function [train_features, test_features, Z_train, Z_test] = extract_plain_csp_features( ...
        X_train, y_train, X_test, cfg)

    fs = cfg.fs;
    lambda = cfg.features.csp_regularization;

    n_components = min(cfg.features.csp_components, ...
        2 * floor(size(X_train,1)/2));

    labels = unique(y_train);

    if numel(labels) ~= 2
        error('CSP requires exactly two classes.');
    end

    fprintf('    Regularized CSP lambda: %.6f\n', lambda);

    % SINGLE BANDPASS ONLY
    [b,a] = butter(4, [8 30] / (fs/2), 'bandpass');

    X_train_f = filter_epochs(X_train, b, a);
    X_test_f  = filter_epochs(X_test, b, a);

    % SINGLE CSP
    W = compute_csp_projection( ...
        X_train_f, y_train, n_components, labels, lambda);

    [train_features, Z_train] = project_csp(X_train_f, W);
    [test_features, Z_test]   = project_csp(X_test_f, W);

    fprintf('    Plain CSP features: %d\n', ...
        size(train_features,2));
end

function X_f = filter_epochs(X, b, a)
    X_f = zeros(size(X));
    for e = 1:size(X, 3)
        X_f(:, :, e) = filtfilt(b, a, X(:, :, e)')';
    end
end

function W = compute_csp_projection(X, y, n_components, labels, lambda)
    cov_1 = avg_norm_cov(X(:, :, y == labels(1)), lambda);
    cov_2 = avg_norm_cov(X(:, :, y == labels(2)), lambda);
    global_cov = (cov_1 + cov_2) / 2;
    beta = 0.15;
    cov_1 = (1 - beta) * cov_1 + beta * global_cov;
    cov_2 = (1 - beta) * cov_2 + beta * global_cov;

    composite = cov_1 + cov_2;

    % Fallback: detect singular/unstable composite before eigendecomposition
    if rcond(composite) < eps
        warning('compute_csp_projection:instability', ...
            'Covariance instability detected; applying additional diagonal stabilization.');
        composite = composite + lambda * 10 * eye(size(composite, 1));
    end

    [V, D] = eig(cov_1, composite);
    d = diag(D);

    % Guard against NaN/Inf eigenvalues from near-singular matrices
    if any(~isfinite(d))
        warning('compute_csp_projection:nanEig', ...
            'Non-finite eigenvalues detected; clamping to [0,1].');
        d(~isfinite(d)) = 0.5;
        D = diag(d);
    end

    [~, order] = sort(d, 'descend');
    V = V(:, order);
    half = n_components / 2;
    W = [V(:, 1:half), V(:, end-half+1:end)];
end

function cov_avg = avg_norm_cov(X, lambda)
    n_ch = size(X, 1);
    cov_avg = zeros(n_ch);
    n_epochs = size(X, 3);
    for e = 1:n_epochs
        trial = squeeze(X(:, :, e));
        trial = trial - mean(trial, 2);
        c = (trial * trial') / size(trial, 2);
        alpha = 0.12;
        target = trace(c) / n_ch * eye(n_ch);
        c = (1 - alpha) * c + alpha * target;
        c = c + lambda * eye(n_ch);
        c = (c + c') / 2;
        cov_avg = cov_avg + c / max(trace(c), eps);
    end
    cov_avg = cov_avg / max(n_epochs, 1);
    cov_avg = (cov_avg + cov_avg') / 2;
end

function [features, Z_all] = project_csp(X, W)
    n_epochs = size(X, 3);
    n_components = size(W, 2);
    n_time = size(X, 2);
    features = zeros(n_epochs, n_components);
    Z_all = zeros(n_components, n_time, n_epochs);
    for e = 1:n_epochs
        Z = W' * X(:, :, e);
        Z_all(:, :, e) = Z;
        v = var(Z, 0, 2);
        features(e, :) = log(v / sum(v) + eps)';
    end
end

function scores = fisher_scores(features, labels, class_labels)
    scores = zeros(1, size(features, 2));
    for f = 1:size(features, 2)
        c1 = features(labels == class_labels(1), f);
        c2 = features(labels == class_labels(2), f);
        scores(f) = (mean(c1) - mean(c2))^2 / (var(c1) + var(c2) + eps);
    end
end

%% ======================================================================
%  CATBOOST CLASSIFIER (Python bridge — reuses existing pattern)
%  ======================================================================

function y_pred = classify_with_catboost(X_train, y_train, X_test, cfg)

    try
        catboost_mod = py.importlib.import_module('catboost');
        np = py.importlib.import_module('numpy');

        label_offset = min(y_train);
        y_zero = y_train(:) - label_offset;

        model = catboost_mod.CatBoostClassifier(pyargs( ...
            'iterations', int32(cfg.classification.catboost_iterations), ...
            'depth', int32(cfg.classification.catboost_depth), ...
            'learning_rate', cfg.classification.catboost_learning_rate, ...
            'loss_function', 'Logloss', ...
            'verbose', cfg.classification.catboost_verbose));

        model.fit(np.array(X_train), np.array(y_zero));

        py_pred = model.predict(np.array(X_test));
        y_pred = double(py_pred);
        y_pred = y_pred(:) + label_offset;

    catch ME
        fprintf('    CatBoost Python bridge failed: %s\n', ME.message);
        fprintf('    Falling back to MATLAB AdaBoostM1 ensemble.\n');
        t = templateTree('MaxNumSplits', 20);
        try
            mdl = fitcensemble(X_train, y_train, ...
                'Method', 'AdaBoostM1', ...
                'Learners', t, ...
                'NumLearningCycles', min(cfg.classification.catboost_iterations, 100));
        catch
            mdl = fitctree(X_train, y_train, 'MaxNumSplits', 20);
        end
        y_pred = predict(mdl, X_test);
    end

    y_pred = y_pred(:);
end

%% ======================================================================
%  PHYSIONET DATA LOADER (standalone copy)
%  ======================================================================

function [raw_eeg, events, subject_info] = load_physionet_subject(subject_id, cfg)

    subject_code = sprintf('S%03d', subject_id);
    subject_dir = fullfile(cfg.dataset.cache_dir, subject_code);

    if ~exist(subject_dir, 'dir')
        error('Subject cache folder not found: %s', subject_dir);
    end

    raw_runs = {};
    events = [];
    sample_offset = 0;
    fs = cfg.fs;
    samples_per_trial = round((cfg.preprocessing.tmax - cfg.preprocessing.tmin) * fs);
    left_count = 0;
    right_count = 0;

    for run_id = cfg.dataset.run_ids
        file_name = sprintf('%sR%02d.edf', subject_code, run_id);
        local_path = fullfile(subject_dir, file_name);

        if ~exist(local_path, 'file')
            if cfg.dataset.download_if_missing
                url = sprintf( ...
                    'https://physionet.org/files/eegmmidb/1.0.0/%s/%s', ...
                    subject_code, file_name);
                websave(local_path, url);
            else
                error('Missing cached EDF file: %s', local_path);
            end
        end

        info = edfinfo(local_path);
        timetable_data = edfread(local_path);
        run_raw = timetable_to_matrix(timetable_data);

        if ~isempty(cfg.dataset.channel_indices)
            run_raw = run_raw(cfg.dataset.channel_indices, :);
        end

        raw_runs{end + 1} = run_raw; %#ok<AGROW>
        annotations = info.Annotations;

        for ann_idx = 1:height(annotations)
            label = string(annotations.Annotations(ann_idx));
            if label ~= "T1" && label ~= "T2", continue; end
            onset_sec = seconds(annotations.Properties.RowTimes(ann_idx));
            start_idx = sample_offset + floor(onset_sec * fs) + 1;
            stop_idx = start_idx + samples_per_trial - 1;
            if stop_idx > sample_offset + size(run_raw, 2), continue; end
            class_label = 1;
            if label == "T2", class_label = 2; right_count = right_count + 1;
            else, left_count = left_count + 1;
            end
            events = [events; start_idx, class_label]; %#ok<AGROW>
        end
        sample_offset = sample_offset + size(run_raw, 2);
    end

    balanced = min(left_count, right_count);
    if balanced < cfg.dataset.min_balanced_trials
        error('Insufficient balanced trials for %s: left=%d, right=%d', ...
            subject_code, left_count, right_count);
    end

    raw_eeg = cell2mat(raw_runs);
    events = balance_events(events, balanced);

    subject_info.subject_id = subject_id;
    subject_info.subject_name = lower(subject_code);
    subject_info.fs = fs;
    subject_info.n_trials = size(events, 1);
end

function data = timetable_to_matrix(tt)
    n_channels = width(tt);
    n_records = height(tt);
    first_cell = tt{1, 1};
    spr = numel(first_cell{1});
    data = zeros(n_channels, n_records * spr);
    for ch = 1:n_channels
        signal = zeros(n_records * spr, 1);
        cursor = 1;
        for rec = 1:n_records
            chunk = tt{rec, ch}{1};
            signal(cursor:cursor + numel(chunk) - 1) = chunk;
            cursor = cursor + numel(chunk);
        end
        data(ch, :) = signal.';
    end
end

function events_out = balance_events(events, n_per_class)
    c1 = events(events(:, 2) == 1, :);
    c2 = events(events(:, 2) == 2, :);
    events_out = [c1(1:n_per_class, :); c2(1:n_per_class, :)];
    [~, order] = sort(events_out(:, 1));
    events_out = events_out(order, :);
end

%% ======================================================================
%  METRICS
%  ======================================================================

function metrics = compute_metrics(y_true, y_pred)
    y_true = y_true(:);
    y_pred = y_pred(:);
    labels = unique([y_true; y_pred]);
    pos = labels(end);
    tp = sum(y_true == pos & y_pred == pos);
    tn = sum(y_true ~= pos & y_pred ~= pos);
    fp = sum(y_true ~= pos & y_pred == pos);
    fn = sum(y_true == pos & y_pred ~= pos);
    metrics.accuracy = (tp + tn) / numel(y_true);
    metrics.precision = tp / max(tp + fp, eps);
    metrics.recall = tp / max(tp + fn, eps);
    metrics.f1 = 2 * metrics.precision * metrics.recall / ...
        max(metrics.precision + metrics.recall, eps);
    metrics.confusion_matrix = [tn, fp; fn, tp];
end

%% ======================================================================
%  RESULT MANAGEMENT
%  ======================================================================

function row = init_result_row(subject_id, subject_name, dataset_name, cfg)
    row = table( ...
        subject_id, ...
        string(subject_name), ...
        string(dataset_name), ...
        string("csp_catboost"), ...
        "PENDING", ...
        0, 0, ...
        cfg.features.csp_components, ...
        0, ...
        NaN, NaN, NaN, NaN, ...
        string(""), ...
        'VariableNames', ...
        {'subject_id', 'subject_name', 'dataset', 'pipeline', ...
         'status', 'n_trials', 'n_channels', ...
         'csp_components', 'csp_feature_dim', ...
         'accuracy', 'precision_val', 'recall_val', 'f1', ...
         'error_message'});
end

function completed = existing_completed_subjects(csv_file)
    completed = [];
    if exist(csv_file, 'file')
        rows = readtable(csv_file, 'TextType', 'string');
        completed = rows.subject_id(rows.status == "OK");
    end
end

function append_result_row(csv_file, row)
    if exist(csv_file, 'file')
        existing = readtable(csv_file, 'TextType', 'string');
        existing(existing.subject_id == row.subject_id, :) = [];
        rows = [existing; row];
    else
        rows = row;
    end
    writetable(rows, csv_file);
end

function results = finalize_results(mode, cfg)
    results = struct();
    results.mode = mode;
    results.csv_file = cfg.results.csv_file;
    if exist(cfg.results.csv_file, 'file')
        all_rows = readtable(cfg.results.csv_file, 'TextType', 'string');
        ok_rows = all_rows(all_rows.status == "OK", :);
        results.rows = all_rows;
        if ~isempty(ok_rows)
            results.summary = table( ...
                mean(ok_rows.accuracy), std(ok_rows.accuracy), ...
                median(ok_rows.accuracy), mean(ok_rows.f1), ...
                height(ok_rows), ...
                'VariableNames', ...
                {'MeanAccuracy','StdAccuracy','MedianAccuracy','MeanF1','N'});
            fprintf('\n--- CSP+CatBoost Summary ---\n');
            disp(results.summary);
        end
    end
    try
        save(cfg.results.mat_file, 'results');
    catch ME
        fprintf('Warning: Could not save MAT file: %s\n', ME.message);
    end
end

function ensure_directory(folder_path)
    if ~exist(folder_path, 'dir'), mkdir(folder_path); end
end

%% ======================================================================
%  CONFIG MERGE
%  ======================================================================

function cfg = merge_config(base, override)
    cfg = base;
    fields = fieldnames(override);
    for k = 1:numel(fields)
        f = fields{k};
        if isstruct(override.(f)) && isfield(cfg, f) && isstruct(cfg.(f))
            cfg.(f) = merge_config(cfg.(f), override.(f));
        else
            cfg.(f) = override.(f);
        end
    end
end
