function results = run_gigadb_hybrid_pipeline(mode)

if nargin < 1
    mode = 'gigadb_smoke';
end

cfg = default_config();

cfg.fs = 512;

cfg.preprocessing.bandpass = [8 30];

cfg.preprocessing.use_temporal_augmentation = false;

rng(cfg.seed, 'twister');

fprintf('GigaDB Hybrid Pipeline Started\n');

switch lower(mode)

    case 'gigadb_smoke'
        subject_files = {'s01.mat'};

    case 'gigadb_pilot5'
        subject_files = arrayfun( ...
            @(x) sprintf('s%02d.mat', x), ...
            1:5, ...
            'UniformOutput', false);

    case 'gigadb_pilot10'
        subject_files = arrayfun( ...
            @(x) sprintf('s%02d.mat', x), ...
            1:10, ...
            'UniformOutput', false);

    case 'gigadb_full52'
        subject_files = arrayfun( ...
            @(x) sprintf('s%02d.mat', x), ...
            1:52, ...
            'UniformOutput', false);

    otherwise
        error('Unknown mode');
end

all_results = [];

for subject_idx = 1:length(subject_files)

    mat_file = subject_files{subject_idx};

    fprintf('\n====================================\n');
    fprintf('Processing %s\n', mat_file);
    fprintf('====================================\n');

    load(mat_file);

    trial_len = 3584;

    left_trials = reshape( ...
        eeg.imagery_left, ...
        68, ...
        trial_len, ...
        eeg.n_imagery_trials);

    right_trials = reshape( ...
        eeg.imagery_right, ...
        68, ...
        trial_len, ...
        eeg.n_imagery_trials);

    epochs = cat(3, left_trials, right_trials);

    labels = [ ...
        ones(eeg.n_imagery_trials,1); ...
        2 * ones(eeg.n_imagery_trials,1)];

    fs = eeg.srate;

    start_sample = round((0.5 + 2.0) * fs);
    end_sample   = round((4.0 + 2.0) * fs);

    epochs = epochs(:, start_sample:end_sample, :);

    epochs = double(epochs);

    labels = labels(:);

    valid_mask = labels > 0;

    epochs = epochs(:, :, valid_mask);

    labels = labels(valid_mask);
% RUN_HYBRID_CSP_RIEMANNIAN_HO_CATBOOST
% Hybrid EEG MI classification pipeline



rng(cfg.seed, 'twister');

fprintf('Hybrid CSP-Riemannian-HO ensemble pipeline started.\n');

epochs = double(epochs);

labels = labels(:);

valid_mask = labels > 0;

epochs = epochs(:, :, valid_mask);

labels = labels(valid_mask);

labels = labels(:);

valid_mask = labels > 0;

epochs = epochs(:, :, valid_mask);
labels = labels(valid_mask);

cv = cvpartition(labels, 'KFold', cfg.classification.cv_folds);

fold_metrics = struct([]);

all_true = [];
all_pred = [];

for fold_idx = 1:cv.NumTestSets

    fprintf('Fold %d/%d\n', fold_idx, cv.NumTestSets);

    train_idx = training(cv, fold_idx);
    test_idx = test(cv, fold_idx);

    X_train = epochs(:, :, train_idx);
    y_train = labels(train_idx);

    X_test = epochs(:, :, test_idx);
    y_test = labels(test_idx);

    [X_train, X_test, channel_info] = ...
        select_adaptive_channels(X_train, y_train, X_test, cfg);

    [X_train, y_train, augmentation_info] = ...
        augment_training_epochs(X_train, y_train, cfg);

    [train_features, test_features, feature_names] = ...
        extract_all_feature_blocks(X_train, y_train, X_test, cfg);

    [X_train_fused, X_test_fused, fusion_model] = ...
        fuse_and_scale_features(train_features, test_features, cfg);

    [best_mask, ho_info] = ...
        binary_hippopotamus_optimizer(X_train_fused, y_train, cfg);

    X_train_selected = X_train_fused(:, best_mask);
    X_test_selected = X_test_fused(:, best_mask);

    [y_pred, classifier_info] = ...
        train_classifier_head( ...
            X_train_selected, ...
            y_train, ...
            X_test_selected, ...
            cfg);

    metrics = compute_metrics(y_test, y_pred);

    fold_metrics(fold_idx).accuracy = metrics.accuracy;
    fold_metrics(fold_idx).precision = metrics.precision;
    fold_metrics(fold_idx).recall = metrics.recall;
    fold_metrics(fold_idx).f1 = metrics.f1;

    fold_metrics(fold_idx).selected_feature_count = sum(best_mask);
    fold_metrics(fold_idx).total_feature_count = numel(best_mask);

    fold_metrics(fold_idx).ho_best_score = ho_info.best_score;

    fold_metrics(fold_idx).classifier_backend = ...
        classifier_info.backend;

    fold_metrics(fold_idx).feature_names = feature_names;
    fold_metrics(fold_idx).fusion_model = fusion_model;
    fold_metrics(fold_idx).selected_channels = ...
        channel_info.selected_channels;
    fold_metrics(fold_idx).channel_scores = ...
        channel_info.channel_scores;
    fold_metrics(fold_idx).augmentation_windows = ...
        augmentation_info.windows_sec;
    fold_metrics(fold_idx).augmented_training_trials = ...
        augmentation_info.n_augmented_trials;

    all_true = [all_true; y_test];
    all_pred = [all_pred; y_pred(:)];

    fprintf( ...
        '  Accuracy: %.2f%% | Selected features: %d/%d | Classifier: %s\n', ...
        metrics.accuracy * 100, ...
        sum(best_mask), ...
        numel(best_mask), ...
        classifier_info.backend);

end

overall = compute_metrics(all_true, all_pred);

results = struct();

results.pipeline = ...
    'Butterworth -> CAR -> Epoching -> CSP + Riemannian + Spectral + Hjorth -> PCA -> HO -> LogReg/SVM Ensemble';

results.config = cfg;
results.fold_metrics = fold_metrics;

results.overall = overall;

results.true_labels = all_true;
results.predicted_labels = all_pred;

fprintf('\nOverall Accuracy: %.2f%%\n', overall.accuracy * 100);
fprintf('Overall F1-score: %.4f\n', overall.f1);

results.subject = mat_file;

all_results = [all_results; results];

end

results = all_results;

end

function cfg = default_config()

cfg = struct();

cfg.seed = 42;

cfg.fs = 512;

cfg.synthetic.n_channels = 8;
cfg.synthetic.n_trials = 24;
cfg.synthetic.trial_duration_sec = 4;

project_root = fileparts(fileparts(mfilename('fullpath')));

cfg.dataset.cache_dir = ...
    fullfile(project_root, 'physionet_cache');

cfg.dataset.run_ids = [4 8 12];

cfg.dataset.subjects = 1:109;

cfg.dataset.min_balanced_trials = 10;

cfg.dataset.download_if_missing = false;

% Motor cortex channels only
cfg.dataset.channel_indices = [6 8 10 12 14 16 18 ...
     44 46 48 50 52 54 56];

cfg.results.output_dir = ...
    fullfile(fileparts(mfilename('fullpath')), ...
    'results_109_subjects');

cfg.results.csv_file = ...
    fullfile(cfg.results.output_dir, ...
    'hybrid_109_subject_results_v2.csv');

cfg.results.mat_file = ...
    fullfile(cfg.results.output_dir, ...
    'hybrid_109_subject_results_v2.mat');

cfg.results.resume = true;

cfg.preprocessing.notch_freqs = 50;

cfg.preprocessing.notch_quality_factor = 35;

cfg.preprocessing.bandpass = [8 30];

cfg.preprocessing.butterworth_order = 4;

cfg.preprocessing.tmin = 0;
cfg.preprocessing.tmax = 4;
cfg.preprocessing.temporal_augmentation_windows = [
    0 3;
    0.5 3.5
];
cfg.preprocessing.use_temporal_augmentation = false;

% ICA disabled for stability/speed
cfg.preprocessing.use_ica = false;

cfg.preprocessing.ica_component_z_threshold = 3;

cfg.features.csp_bands = [
    8 10;
    10 12;
    12 14;
    14 16;
    16 20;
    20 24;
    24 28;
    28 30
];

cfg.features.csp_components = 6;
cfg.features.adaptive_csp_band_count = 7;
cfg.features.adaptive_channel_count = 8;
cfg.features.min_adaptive_channel_count = 6;

cfg.features.spectral_bands = ...
    [4 8
    8 12
    12 16
    16 20
    20 24
    24 28
    28 32];

cfg.features.riemannian_epsilon = 1e-6;

cfg.features.sample_entropy_m = 2;
cfg.features.sample_entropy_r = 0.2;

cfg.features.sample_entropy_stride = 8;
cfg.features.sample_entropy_max_points = 80;
cfg.features.use_blockwise_pca = true;
cfg.features.block_pca_variance = 95;
cfg.features.max_block_pca_components = 12;
cfg.features.use_final_pca = true;
cfg.features.final_pca_variance = 95;
cfg.features.max_final_pca_components = 40;

cfg.optimization.population_size = 10;

cfg.optimization.iterations = 20;

cfg.optimization.threshold = 0.5;

cfg.optimization.mutation_rate = 0.08;

cfg.optimization.cv_folds = 3;

cfg.optimization.feature_penalty = 0.015;

cfg.classification.cv_folds = 3;

cfg.classification.decision_head = 'ensemble';
cfg.classification.ho_decision_head = 'ensemble';
cfg.classification.ensemble_backend = ...
    'logreg_linear_svm';
cfg.classification.use_weighted_ensemble = false;
cfg.classification.use_weighted_ensemble_during_ho = false;
cfg.classification.inner_weight_folds = 3;
cfg.classification.include_rbf_svm = false;
cfg.classification.include_boosted_tree = false;
cfg.classification.enable_python_catboost = false;

cfg.classification.mlp_hidden_dims = [64 32];
cfg.classification.mlp_dropout = 0.2;
cfg.classification.mlp_initial_learn_rate = 1e-4;
cfg.classification.mlp_max_epochs = 80;
cfg.classification.mlp_mini_batch_size = 32;
cfg.classification.mlp_validation_ratio = 0.2;
cfg.classification.mlp_validation_patience = 8;
cfg.classification.mlp_execution_environment = 'cpu';

cfg.classification.num_learning_cycles = 80;

cfg.classification.max_num_splits = 20;

cfg.classification.learning_rate = 0.08;

cfg.classification.depth = 6;

cfg.classification.verbose = false;

end


function [epochs, labels] = preprocess_pipeline(raw_eeg, events, cfg)

fs = cfg.fs;

data = double(raw_eeg);

for notch_freq = cfg.preprocessing.notch_freqs

    if notch_freq > 0 && notch_freq < fs / 2

        [b, a] = design_notch_filter( ...
            notch_freq, ...
            fs, ...
            cfg.preprocessing.notch_quality_factor);

        data = filtfilt(b, a, data')';

    end

end

[b, a] = butter( ...
    cfg.preprocessing.butterworth_order, ...
    cfg.preprocessing.bandpass / (fs / 2), ...
    'bandpass');

data = filtfilt(b, a, data')';

% CAR
data = data - mean(data, 1);

if cfg.preprocessing.use_ica

    data = suppress_artifacts_with_ica_or_pca(data, cfg);

end

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

function clean_data = suppress_artifacts_with_ica_or_pca(data, cfg)
    clean_data = data;

    if exist('runica', 'file') == 2
        try
            [weights, sphere] = runica(data, 'verbose', 'off');
            activations = weights * sphere * data;
            component_power = var(activations, 0, 2);
            artifact_mask = zscore(component_power) > cfg.preprocessing.ica_component_z_threshold;
            activations(artifact_mask, :) = 0;
            clean_data = pinv(weights * sphere) * activations;
            return;
        catch
        end
    end

    [coeff, score, latent, ~, ~, mu] = pca(data');
    component_power = latent / sum(latent);
    artifact_mask = zscore(component_power) > cfg.preprocessing.ica_component_z_threshold;
    if any(artifact_mask)
        score(:, artifact_mask) = 0;
        clean_data = (score * coeff')' + mu';
    end
end

function [b, a] = design_notch_filter(notch_freq, fs, quality_factor)
    bandwidth = notch_freq / quality_factor;
    low_freq = max(notch_freq - bandwidth / 2, 0.1);
    high_freq = min(notch_freq + bandwidth / 2, fs / 2 - 0.1);
    [b, a] = butter(2, [low_freq high_freq] / (fs / 2), 'stop');
end

function [X_train_selected, X_test_selected, info] = ...
    select_adaptive_channels(X_train, y_train, X_test, cfg)

    n_channels = size(X_train, 1);

    info = struct();
    info.channel_scores = zeros(1, n_channels);
    info.selected_channels = 1:n_channels;

    if n_channels <= cfg.features.min_adaptive_channel_count
        X_train_selected = X_train;
        X_test_selected = X_test;
        return;
    end

    labels = unique(y_train);

    if numel(labels) ~= 2
        X_train_selected = X_train;
        X_test_selected = X_test;
        return;
    end

    channel_scores = zeros(1, n_channels);

    for channel_idx = 1:n_channels

        channel_epoch_power = squeeze( ...
            log(mean(X_train(channel_idx, :, :) .^ 2, 2) + eps));

        class_1 = channel_epoch_power(y_train == labels(1));
        class_2 = channel_epoch_power(y_train == labels(2));

        numerator = (mean(class_1) - mean(class_2)) ^ 2;
        denominator = var(class_1) + var(class_2) + eps;

        channel_scores(channel_idx) = numerator / denominator;

    end

    [~, ranked_channels] = sort(channel_scores, 'descend');

    n_selected = min(n_channels, ...
        max(cfg.features.min_adaptive_channel_count, ...
            cfg.features.adaptive_channel_count));

    selected_channels = sort(ranked_channels(1:n_selected));

    X_train_selected = X_train(selected_channels, :, :);
    X_test_selected = X_test(selected_channels, :, :);

    info.channel_scores = channel_scores;
    info.selected_channels = selected_channels;

    fprintf('  Adaptive channels: kept %d/%d\n', ...
        numel(selected_channels), n_channels);

end

function [X_augmented, y_augmented, info] = ...
    augment_training_epochs(X_train, y_train, cfg)

    info = struct();
    info.windows_sec = cfg.preprocessing.temporal_augmentation_windows;
    info.n_original_trials = size(X_train, 3);
    info.n_augmented_trials = size(X_train, 3);

    if ~cfg.preprocessing.use_temporal_augmentation
        X_augmented = X_train;
        y_augmented = y_train;
        return;
    end

    fs = cfg.fs;
    windows_sec = cfg.preprocessing.temporal_augmentation_windows;
    n_channels = size(X_train, 1);
    n_epochs = size(X_train, 3);
    epoch_samples = size(X_train, 2);

    crop_ranges = zeros(0, 2);
    crop_len = [];

    for window_idx = 1:size(windows_sec, 1)

        start_sample = round( ...
            (windows_sec(window_idx, 1) - cfg.preprocessing.tmin) * fs) + 1;
        stop_sample = round( ...
            (windows_sec(window_idx, 2) - cfg.preprocessing.tmin) * fs);

        if start_sample < 1 || stop_sample > epoch_samples || ...
           stop_sample <= start_sample
            continue;
        end

        current_len = stop_sample - start_sample + 1;

        if isempty(crop_len)
            crop_len = current_len;
        elseif current_len ~= crop_len
            continue;
        end

        crop_ranges(end + 1, :) = [start_sample, stop_sample]; %#ok<AGROW>

    end

    if isempty(crop_ranges)
        X_augmented = X_train;
        y_augmented = y_train;
        info.windows_sec = zeros(0, 2);
        return;
    end

    n_windows = size(crop_ranges, 1);
    X_augmented = zeros(n_channels, crop_len, n_epochs * n_windows);
    y_augmented = zeros(n_epochs * n_windows, 1);

    write_idx = 1;

    for epoch_idx = 1:n_epochs

        for window_idx = 1:n_windows

            range = crop_ranges(window_idx, 1):crop_ranges(window_idx, 2);

            X_augmented(:, :, write_idx) = X_train(:, range, epoch_idx);
            y_augmented(write_idx) = y_train(epoch_idx);

            write_idx = write_idx + 1;

        end

    end

    info.windows_sec = windows_sec(1:n_windows, :);
    info.n_augmented_trials = size(X_augmented, 3);

    fprintf('  Temporal augmentation: %d -> %d training trials\n', ...
        info.n_original_trials, info.n_augmented_trials);

end

function [train_blocks, test_blocks, feature_names] = extract_all_feature_blocks(X_train, y_train, X_test, cfg)
    train_blocks = {};
    test_blocks = {};
    feature_names = {};

    fprintf('  Extracting CSP features...\n');
    [csp_train, csp_test] = extract_csp_features(X_train, y_train, X_test, cfg);
    train_blocks{end + 1} = csp_train;
    test_blocks{end + 1} = csp_test;
    feature_names{end + 1} = 'CSP';

    fprintf('  Extracting Riemannian tangent-space features...\n');
    [riem_train, riem_test] = extract_riemannian_features(X_train, X_test, cfg);
    train_blocks{end + 1} = riem_train;
    test_blocks{end + 1} = riem_test;
    feature_names{end + 1} = 'Riemannian tangent space';

    fprintf('  Extracting spectral features...\n');
    spectral_train = extract_spectral_features(X_train, cfg);
    spectral_test = extract_spectral_features(X_test, cfg);
    train_blocks{end + 1} = spectral_train;
    test_blocks{end + 1} = spectral_test;
    feature_names{end + 1} = 'Differential entropy + band power';

    fprintf('  Extracting temporal features...\n');
    temporal_train = extract_temporal_features(X_train, cfg);
    temporal_test = extract_temporal_features(X_test, cfg);
    train_blocks{end + 1} = temporal_train;
    test_blocks{end + 1} = temporal_test;
    feature_names{end + 1} = 'Hjorth + sample entropy';
end

function [train_features, test_features] = extract_csp_features(X_train, y_train, X_test, cfg)

bands = cfg.features.csp_bands;

fs = cfg.fs;

n_components = min(cfg.features.csp_components, ...
    2 * floor(size(X_train,1)/2));

n_bands = size(bands, 1);

all_train_features = [];
all_test_features = [];
band_scores = zeros(1, n_bands);
band_feature_ranges = cell(1, n_bands);
filtered_train_bands = cell(1, n_bands);
filtered_test_bands = cell(1, n_bands);

labels = unique(y_train);

if numel(labels) ~= 2
    error('CSP requires exactly two classes.');
end



for band_idx = 1:n_bands
    [b, a] = butter(4, bands(band_idx, :) / (fs / 2), 'bandpass');
    filtered_train_bands{band_idx} = filter_epochs(X_train, b, a);
    filtered_test_bands{band_idx} = filter_epochs(X_test, b, a);
end


for band_idx = 1:n_bands

    X_train_f = filtered_train_bands{band_idx};
    X_test_f = filtered_test_bands{band_idx};

    [train_band_features, test_band_features] = ...
        compute_csp_feature_pair( ...
            X_train_f, ...
            y_train, ...
            X_test_f, ...
            n_components, ...
            labels);

    band_feature_scores = fisher_scores_for_features( ...
        train_band_features, y_train, labels);

    band_scores(band_idx) = mean(band_feature_scores);

    start_col = size(all_train_features, 2) + 1;
    stop_col = start_col + size(train_band_features, 2) - 1;
    band_feature_ranges{band_idx} = start_col:stop_col;

    all_train_features = ...
        [all_train_features, train_band_features];

    all_test_features = ...
        [all_test_features, test_band_features];

end

% Rank complete CSP bands so each fold keeps subject-specific frequencies
% instead of isolated components from every band.
[~, ranked_bands] = sort(band_scores, 'descend');

n_selected_bands = min(n_bands, ...
    max(3, cfg.features.adaptive_csp_band_count));

selected_bands = sort(ranked_bands(1:n_selected_bands));

selected_idx = [];

for idx = 1:numel(selected_bands)

    selected_idx = [selected_idx, ...
        band_feature_ranges{selected_bands(idx)}]; %#ok<AGROW>

end

train_features = all_train_features(:, selected_idx);

test_features = all_test_features(:, selected_idx);

fprintf('  Adaptive CSP bands: kept %d/%d\n', ...
    numel(selected_bands), n_bands);

end

function [train_features, test_features] = ...
    compute_csp_feature_pair( ...
    X_train_f, ...
    y_train, ...
    X_test_f, ...
    n_components, ...
    labels)

W = compute_csp_projection_matrix( ...
    X_train_f, ...
    y_train, ...
    n_components, ...
    labels);

train_features = ...
    project_csp_features(X_train_f, W);

test_features = ...
    project_csp_features(X_test_f, W);

end

function scores = fisher_scores_for_features( ...
    features, labels, class_labels)

scores = zeros(1, size(features, 2));

for feature_idx = 1:size(features, 2)

    feature_values = features(:, feature_idx);

    class_1 = ...
        feature_values(labels == class_labels(1));

    class_2 = ...
        feature_values(labels == class_labels(2));

    numerator = ...
        (mean(class_1) - mean(class_2)) ^ 2;

    denominator = ...
        var(class_1) + var(class_2) + eps;

    scores(feature_idx) = numerator / denominator;

end

end

function W = compute_csp_projection_matrix( ...
    X_train_f, y_train, n_components, labels)

cov_1 = average_normalized_covariance( ...
    X_train_f(:, :, y_train == labels(1)));

cov_2 = average_normalized_covariance( ...
    X_train_f(:, :, y_train == labels(2)));

[V, D] = eig(cov_1, cov_1 + cov_2);

[~, order] = sort(diag(D), 'descend');

V = V(:, order);

half_components = n_components / 2;

W = [ ...
    V(:, 1:half_components), ...
    V(:, end-half_components+1:end)];

end



function X_f = filter_epochs(X, b, a)
    X_f = zeros(size(X));
    for epoch_idx = 1:size(X, 3)
        X_f(:, :, epoch_idx) = filtfilt(b, a, X(:, :, epoch_idx)')';
    end
end

function cov_avg = average_normalized_covariance(X)

n_channels = size(X, 1);

cov_avg = zeros(n_channels, n_channels);

n_epochs = size(X, 3);

for epoch_idx = 1:n_epochs

    trial = squeeze(X(:, :, epoch_idx));

    trial = trial - mean(trial, 2);

    c = (trial * trial') / size(trial, 2);

    if isscalar(c)

        c = eye(size(trial,1));

    end

    % Stronger adaptive shrinkage
    alpha = 0.015;

    target = trace(c) / size(c,1) * eye(size(c));

    c = (1 - alpha) * c + alpha * target;

    % Numerical stabilization
    c = c + 1e-6 * eye(size(c));

    c = (c + c') / 2;

    cov_avg = cov_avg + ...
        c / max(trace(c), eps);

end

cov_avg = cov_avg / max(n_epochs, 1);

cov_avg = ...
    (cov_avg + cov_avg') / 2;

end

function c_shrink = shrink_covariance(c, alpha)

n = size(c, 1);

target = ...
    trace(c) / n * eye(n);

c_shrink = ...
    (1 - alpha) * c + ...
    alpha * target;

c_shrink = ...
    (c_shrink + c_shrink') / 2;

end

function features = project_csp_features(X, W)
    n_epochs = size(X, 3);
    n_components = size(W, 2);
    features = zeros(n_epochs, n_components);
    for epoch_idx = 1:n_epochs
        Z = W' * X(:, :, epoch_idx);
        variance = var(Z, 0, 2);
        features(epoch_idx, :) = log((variance / sum(variance)) + eps)';
    end
end

function [train_features, test_features] = extract_riemannian_features(X_train, X_test, cfg)
    train_cov = covariance_matrices(X_train, cfg.features.riemannian_epsilon);
    test_cov = covariance_matrices(X_test, cfg.features.riemannian_epsilon);
    reference_cov = riemannian_mean_spd(train_cov);
    train_features = tangent_space_features(train_cov, reference_cov);
    test_features = tangent_space_features(test_cov, reference_cov);
end

function covs = covariance_matrices(X, epsilon)
    n_channels = size(X, 1);
    n_epochs = size(X, 3);
    covs = zeros(n_channels, n_channels, n_epochs);
    for epoch_idx = 1:n_epochs
        trial = squeeze(X(:, :, epoch_idx));

        % Channel centering
        trial = trial - mean(trial, 2);

        % Proper covariance estimation
        c = (trial * trial') / size(trial, 2);

        % Adaptive shrinkage target
        alpha = 0.015;

        target = trace(c) / n_channels * eye(n_channels);

        c = (1 - alpha) * c + alpha * target;

        % Numerical stabilization
        c = c + epsilon * eye(n_channels);

        % Normalize
        covs(:, :, epoch_idx) = c / max(trace(c), eps);
    end
end

function mean_cov = riemannian_mean_spd(covs)
    mean_cov = mean(covs, 3);
    mean_cov = (mean_cov + mean_cov') / 2;
    for iter = 1:20
        sqrt_mean = sqrtm(mean_cov);
        inv_sqrt_mean = sqrt_mean \ eye(size(sqrt_mean));
        tangent_sum = zeros(size(mean_cov));
        for idx = 1:size(covs, 3)
            tangent_sum = tangent_sum + logm(inv_sqrt_mean * covs(:, :, idx) * inv_sqrt_mean);
        end
        tangent_avg = tangent_sum / size(covs, 3);
        mean_cov = sqrt_mean * expm(tangent_avg) * sqrt_mean;
        mean_cov = real((mean_cov + mean_cov') / 2);
        if norm(tangent_avg, 'fro') < 1e-6
            break;
        end
    end
end

function features = tangent_space_features(covs, reference_cov)
    n_channels = size(covs, 1);
    n_epochs = size(covs, 3);
    inv_sqrt_ref = sqrtm(reference_cov) \ eye(n_channels);
    upper_mask = triu(true(n_channels));
    offdiag_mask = triu(true(n_channels), 1);
    n_features = nnz(upper_mask);
    features = zeros(n_epochs, n_features);

    for epoch_idx = 1:n_epochs
        tangent = real(logm(inv_sqrt_ref * covs(:, :, epoch_idx) * inv_sqrt_ref));
        tangent(offdiag_mask) = sqrt(2) * tangent(offdiag_mask);
        features(epoch_idx, :) = tangent(upper_mask)';
    end
end

function features = extract_spectral_features(X, cfg)
    fs = cfg.fs;
    bands = cfg.features.spectral_bands;
    n_epochs = size(X, 3);
    n_channels = size(X, 1);
    n_bands = size(bands, 1);
    features = zeros(n_epochs, n_channels + n_channels * n_bands);

    for epoch_idx = 1:n_epochs
        epoch = squeeze(X(:, :, epoch_idx))';
        channel_var = var(epoch, 0, 1) + eps;
        differential_entropy = 0.5 * log(2 * pi * exp(1) * channel_var);
        band_power = zeros(1, n_channels * n_bands);
        write_idx = 1;
        for band_idx = 1:n_bands
            power_values = bandpower(epoch, fs, bands(band_idx, :));
            band_power(write_idx:write_idx + n_channels - 1) = log(power_values + eps);
            write_idx = write_idx + n_channels;
        end
        features(epoch_idx, :) = [differential_entropy, band_power];
    end
end

function features = extract_temporal_features(X, cfg)
    n_epochs = size(X, 3);
    n_channels = size(X, 1);
    features = zeros(n_epochs, n_channels * 4);

    for epoch_idx = 1:n_epochs
        epoch = squeeze(X(:, :, epoch_idx))';
        dx = diff(epoch, 1, 1);
        ddx = diff(dx, 1, 1);
        var_x = var(epoch, 0, 1) + eps;
        var_dx = var(dx, 0, 1) + eps;
        var_ddx = var(ddx, 0, 1) + eps;
        activity = var_x;
        mobility = sqrt(var_dx ./ var_x);
        complexity = sqrt(var_ddx ./ var_dx) ./ mobility;

        sample_entropy = zeros(1, n_channels);
        for channel_idx = 1:n_channels
            sample_entropy(channel_idx) = compute_sample_entropy(epoch(:, channel_idx), ...
                cfg.features.sample_entropy_m, cfg.features.sample_entropy_r, ...
                cfg.features.sample_entropy_stride, cfg.features.sample_entropy_max_points);
        end
        features(epoch_idx, :) = [activity, mobility, complexity, sample_entropy];
    end
end

function value = compute_sample_entropy(signal, m, r_factor, stride, max_points)
    signal = signal(:)';
    if stride > 1
        signal = signal(1:stride:end);
    end
    if numel(signal) > max_points
        sample_idx = round(linspace(1, numel(signal), max_points));
        signal = signal(sample_idx);
    end
    signal = signal - mean(signal);
    tolerance = r_factor * std(signal);
    if tolerance <= eps || numel(signal) <= m + 2
        value = 0;
        return;
    end
    count_m = count_template_matches(signal, m, tolerance);
    count_m1 = count_template_matches(signal, m + 1, tolerance);
    value = -log((count_m1 + eps) / (count_m + eps));
end

function count = count_template_matches(signal, m, tolerance)
    n_templates = numel(signal) - m + 1;
    count = 0;
    for i = 1:n_templates - 1
        template_i = signal(i:i + m - 1);
        for j = i + 1:n_templates
            template_j = signal(j:j + m - 1);
            if max(abs(template_i - template_j)) <= tolerance
                count = count + 1;
            end
        end
    end
end

function [X_train_fused, X_test_fused, model] = ...
    fuse_and_scale_features(train_blocks, test_blocks, cfg)
    n_blocks = numel(train_blocks);

    transformed_train_blocks = cell(1, n_blocks);
    transformed_test_blocks = cell(1, n_blocks);

    model = struct();
    model.blocks = cell(1, n_blocks);

    for block_idx = 1:n_blocks
        train_block = train_blocks{block_idx};
        test_block = test_blocks{block_idx};

        block_model = struct();
        block_model.mu = mean(train_block, 1);
        block_model.sigma = std(train_block, 0, 1);
        block_model.sigma(block_model.sigma == 0) = 1;

        train_block = ...
            (train_block - block_model.mu) ./ block_model.sigma;
        test_block = ...
            (test_block - block_model.mu) ./ block_model.sigma;

        if cfg.features.use_blockwise_pca && size(train_block, 2) > 1
            [coeff, score, ~, ~, explained, pca_mu] = pca(train_block);

            cumvar = cumsum(explained);
            n_comp = find( ...
                cumvar >= cfg.features.block_pca_variance, ...
                1);

            if isempty(n_comp)
                n_comp = size(score, 2);
            end

            n_comp = max(1, min(n_comp, size(score, 2)));
            n_comp = min(n_comp, cfg.features.max_block_pca_components);

            train_block = score(:, 1:n_comp);
            test_block = ...
                (test_block - pca_mu) * coeff(:, 1:n_comp);

            block_model.pca_coeff = coeff;
            block_model.pca_mu = pca_mu;
            block_model.pca_components = n_comp;
            block_model.explained = explained;
        else
            block_model.pca_coeff = [];
            block_model.pca_mu = zeros(1, size(train_block, 2));
            block_model.pca_components = size(train_block, 2);
            block_model.explained = [];
        end

        transformed_train_blocks{block_idx} = train_block;
        transformed_test_blocks{block_idx} = test_block;
        model.blocks{block_idx} = block_model;
    end

    X_train_fused = cell2mat(transformed_train_blocks);
    X_test_fused = cell2mat(transformed_test_blocks);

    model.final_mu = mean(X_train_fused, 1);
    model.final_sigma = std(X_train_fused, 0, 1);
    model.final_sigma(model.final_sigma == 0) = 1;

    X_train_fused = ...
        (X_train_fused - model.final_mu) ./ model.final_sigma;
    X_test_fused = ...
        (X_test_fused - model.final_mu) ./ model.final_sigma;

    if cfg.features.use_final_pca && size(X_train_fused, 2) > 1
        [coeff, score, ~, ~, explained, pca_mu] = pca(X_train_fused);

        cumvar = cumsum(explained);
        n_comp = find( ...
            cumvar >= cfg.features.final_pca_variance, ...
            1);

        if isempty(n_comp)
            n_comp = size(score, 2);
        end

        n_comp = max(1, min(n_comp, size(score, 2)));
        n_comp = min(n_comp, cfg.features.max_final_pca_components);

        X_train_fused = score(:, 1:n_comp);
        X_test_fused = ...
            (X_test_fused - pca_mu) * coeff(:, 1:n_comp);

        model.final_pca_coeff = coeff;
        model.final_pca_mu = pca_mu;
        model.final_pca_components = n_comp;
        model.final_pca_explained = explained;
    else
        model.final_pca_coeff = [];
        model.final_pca_mu = zeros(1, size(X_train_fused, 2));
        model.final_pca_components = size(X_train_fused, 2);
        model.final_pca_explained = [];
    end

    model.total_components = size(X_train_fused, 2);
end

function [best_mask, info] = binary_hippopotamus_optimizer(X, y, cfg)
    n_features = size(X, 2);
    pop_size = cfg.optimization.population_size;
    iterations = cfg.optimization.iterations;
    threshold = cfg.optimization.threshold;
    mutation_rate = cfg.optimization.mutation_rate;

    population = rand(pop_size, n_features);
    fitness = zeros(pop_size, 1);
    best_score = -Inf;
    best_position = population(1, :);
    convergence = zeros(iterations, 1);

    for iter = 1:iterations
        for idx = 1:pop_size
            current = population(idx, :);
            candidate = current;
            phase = rand();
            if phase < 0.5
                candidate = current + rand(1, n_features) .* (best_position - current);
            else
                random_peer = population(randi(pop_size), :);
                candidate = current + 0.5 * randn(1, n_features) .* (current - random_peer);
            end
            mutation_mask = rand(1, n_features) < mutation_rate;
            candidate(mutation_mask) = rand(1, sum(mutation_mask));
            candidate = min(max(candidate, 0), 1);
            candidate_mask = candidate > threshold;
            if ~any(candidate_mask)
                candidate_mask(randi(n_features)) = true;
                candidate(candidate_mask) = 1;
            end
            candidate_score = evaluate_feature_mask(candidate_mask, X, y, cfg);
            if candidate_score >= fitness(idx)
                population(idx, :) = candidate;
                fitness(idx) = candidate_score;
            end
            if fitness(idx) > best_score
                best_score = fitness(idx);
                best_position = population(idx, :);
            end
        end
        convergence(iter) = best_score;
    end

    best_mask = best_position > threshold;
    if ~any(best_mask)
        best_mask(:) = true;
    end
    info.best_score = best_score;
    info.convergence = convergence;
end

function score = evaluate_feature_mask(mask, X, y, cfg)

    if ~any(mask)
        score = -Inf;
        return;
    end

    class_labels = unique(y);

    class_counts = arrayfun(@(label) sum(y == label), class_labels);

    n_folds = min(cfg.optimization.cv_folds, ...
                  min(class_counts));

    if n_folds < 2
        score = 0;
        return;
    end

    cv = cvpartition(y, 'KFold', n_folds);

    accuracies = zeros(cv.NumTestSets, 1);

    f1_scores = zeros(cv.NumTestSets, 1);

    X_selected = X(:, mask);

    for fold_idx = 1:cv.NumTestSets

        train_idx = training(cv, fold_idx);

        test_idx = test(cv, fold_idx);

        [pred, ~] = train_classifier_head( ...
            X_selected(train_idx, :), ...
            y(train_idx), ...
            X_selected(test_idx, :), ...
            cfg, ...
            cfg.classification.ho_decision_head);

        y_true = y(test_idx);

        accuracies(fold_idx) = ...
            mean(pred(:) == y_true);

        tp = sum((pred(:) == 2) & (y_true == 2));

        fp = sum((pred(:) == 2) & (y_true == 1));

        fn = sum((pred(:) == 1) & (y_true == 2));

        precision = tp / (tp + fp + eps);

        recall = tp / (tp + fn + eps);

        f1_scores(fold_idx) = ...
            2 * (precision * recall) / ...
            (precision + recall + eps);

    end

    feature_ratio = sum(mask) / numel(mask);

    feature_penalty = cfg.optimization.feature_penalty;

    score = ...
        0.45 * mean(accuracies) + ...
        0.35 * mean(f1_scores) - ...
        feature_penalty * feature_ratio;

end 
function [pred, info] = train_classifier_head( ...
    X_train, y_train, X_test, cfg, decision_head)

if nargin < 5 || isempty(decision_head)
    decision_head = cfg.classification.decision_head;
end

info = struct();

mu = mean(X_train, 1);
sigma = std(X_train, [], 1) + 1e-8;

X_train = (X_train - mu) ./ sigma;
X_test = (X_test - mu) ./ sigma;

info.normalization_mu = mu;
info.normalization_sigma = sigma;
info.decision_head = char(string(decision_head));

switch lower(string(decision_head))
    case "mlp"
        [pred, mlp_info] = train_mlp_head_and_predict( ...
            X_train, y_train, X_test, cfg);
        info.backend = 'MLP-64-32';
        info.mlp = mlp_info;
    case "ensemble"
        [pred, ensemble_info] = train_ensemble_head_and_predict( ...
            X_train, y_train, X_test, cfg, ...
            cfg.classification.use_weighted_ensemble);
        info.backend = ensemble_info.backend;
        info.weights = ensemble_info.weights;
        info.models = ensemble_info.models;
    otherwise
        error('Unknown classifier decision head: %s', decision_head);
end

end

function [pred, info] = train_ensemble_head_and_predict( ...
    X_train, y_train, X_test, cfg, use_weighted_ensemble)

if nargin < 5
    use_weighted_ensemble = cfg.classification.use_weighted_ensemble;
end

info = struct();

model_names = {'logreg', 'linear_svm'};

if use_weighted_ensemble
    if cfg.classification.include_rbf_svm
        model_names{end + 1} = 'rbf_svm';
    end

    if cfg.classification.include_boosted_tree
        model_names{end + 1} = 'boosted_tree';
    end

    if isfield(cfg.classification, 'enable_python_catboost') && ...
            cfg.classification.enable_python_catboost
        model_names{end + 1} = 'python_catboost';
    end
end

all_pred = zeros(size(X_test, 1), numel(model_names));

for model_idx = 1:numel(model_names)
    all_pred(:, model_idx) = train_model_prediction( ...
        model_names{model_idx}, ...
        X_train, ...
        y_train, ...
        X_test, ...
        cfg);
end

if use_weighted_ensemble
    weights = estimate_validation_weights( ...
        X_train, ...
        y_train, ...
        cfg, ...
        model_names);
    pred = weighted_vote_predictions(all_pred, weights, unique(y_train));
    info.backend = sprintf( ...
        'Weighted-Ensemble-%s', ...
        strjoin(model_names, '-'));
else
    weights = ones(1, numel(model_names)) / numel(model_names);
    pred = mode(all_pred, 2);
    info.backend = sprintf( ...
        'Ensemble-%s', ...
        strjoin(model_names, '-'));
end

info.weights = weights;
info.models = model_names;

end

function [pred, info] = train_mlp_head_and_predict(X_train, y_train, X_test, cfg)

info = struct();

n_features = size(X_train, 2);
hidden_dims = cfg.classification.mlp_hidden_dims;
dropout_rate = cfg.classification.mlp_dropout;
mini_batch_size = min( ...
    cfg.classification.mlp_mini_batch_size, ...
    max(1, size(X_train, 1)));

Y_train_cat = categorical(y_train(:));

[X_fit, Y_fit_cat, X_val, Y_val_cat] = ...
    create_mlp_validation_split(X_train, Y_train_cat, cfg);

layers = [
    featureInputLayer(n_features, 'Normalization', 'none', 'Name', 'input')
    fullyConnectedLayer(hidden_dims(1), 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    dropoutLayer(dropout_rate, 'Name', 'drop1')
    fullyConnectedLayer(hidden_dims(2), 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(numel(categories(Y_train_cat)), 'Name', 'fc_out')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classoutput')
];

validation_frequency = max(1, floor(size(X_fit, 1) / mini_batch_size));

if isempty(X_val)
    options = trainingOptions('adam', ...
        'InitialLearnRate', cfg.classification.mlp_initial_learn_rate, ...
        'MaxEpochs', cfg.classification.mlp_max_epochs, ...
        'MiniBatchSize', mini_batch_size, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', false, ...
        'ExecutionEnvironment', ...
        cfg.classification.mlp_execution_environment);
else
    options = trainingOptions('adam', ...
        'InitialLearnRate', cfg.classification.mlp_initial_learn_rate, ...
        'MaxEpochs', cfg.classification.mlp_max_epochs, ...
        'MiniBatchSize', mini_batch_size, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', false, ...
        'ExecutionEnvironment', ...
        cfg.classification.mlp_execution_environment, ...
        'ValidationData', {X_val, Y_val_cat}, ...
        'ValidationFrequency', validation_frequency, ...
        'ValidationPatience', ...
        cfg.classification.mlp_validation_patience);
end

net = trainNetwork(X_fit, Y_fit_cat, layers, options);

scores = predict(net, X_test);
class_labels = str2double(categories(Y_train_cat));
[~, best_idx] = max(scores, [], 2);
pred = class_labels(best_idx);
pred = pred(:);

info.net = net;
info.hidden_dims = hidden_dims;
info.dropout = dropout_rate;
info.mini_batch_size = mini_batch_size;
info.validation_used = ~isempty(X_val);

end

function [X_fit, Y_fit_cat, X_val, Y_val_cat] = ...
    create_mlp_validation_split(X_train, Y_train_cat, cfg)

X_fit = X_train;
Y_fit_cat = Y_train_cat;
X_val = [];
Y_val_cat = categorical();

if size(X_train, 1) < 10 || numel(categories(Y_train_cat)) < 2
    return;
end

try
    holdout = cvpartition(double(Y_train_cat), ...
        'HoldOut', cfg.classification.mlp_validation_ratio);
catch
    return;
end

fit_idx = training(holdout);
val_idx = test(holdout);

if sum(fit_idx) < 2 || sum(val_idx) < 1
    return;
end

if numel(unique(Y_train_cat(fit_idx))) < 2 || ...
        numel(unique(Y_train_cat(val_idx))) < 2
    return;
end

X_fit = X_train(fit_idx, :);
Y_fit_cat = Y_train_cat(fit_idx);
X_val = X_train(val_idx, :);
Y_val_cat = Y_train_cat(val_idx);

end

function pred = train_model_prediction(model_name, X_train, y_train, X_test, cfg)

switch lower(string(model_name))
    case "logreg"
        pred = train_logistic_and_predict(X_train, y_train, X_test);
    case "linear_svm"
        pred = train_svm_and_predict(X_train, y_train, X_test);
    case "rbf_svm"
        pred = train_rbf_svm_and_predict(X_train, y_train, X_test);
    case "boosted_tree"
        pred = train_boosted_tree_fallback(X_train, y_train, X_test, cfg);
    case "python_catboost"
        pred = train_python_catboost(X_train, y_train, X_test, cfg);
    otherwise
        error('Unknown model in ensemble: %s', model_name);
end

pred = pred(:);

end

function pred = train_logistic_and_predict(X_train, y_train, X_test)

log_model = fitclinear( ...
    X_train, ...
    y_train, ...
    'Learner', 'logistic', ...
    'Regularization', 'ridge', ...
    'Lambda', 1e-3);

pred = predict(log_model, X_test);

end

function pred = train_svm_and_predict(X_train, y_train, X_test)

svm_model = fitcsvm( ...
    X_train, ...
    y_train, ...
    'KernelFunction', 'linear', ...
    'Standardize', false);

pred = predict(svm_model, X_test);

end

function pred = train_rbf_svm_and_predict(X_train, y_train, X_test)

svm_model = fitcsvm( ...
    X_train, ...
    y_train, ...
    'KernelFunction', 'rbf', ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', false);

pred = predict(svm_model, X_test);

end

function pred = train_boosted_tree_fallback(X_train, y_train, X_test, cfg)

tree = templateTree( ...
    'MaxNumSplits', ...
    cfg.classification.max_num_splits);

try
    model = fitcensemble( ...
        X_train, ...
        y_train, ...
        'Method', 'AdaBoostM1', ...
        'Learners', tree, ...
        'NumLearningCycles', ...
        cfg.classification.num_learning_cycles);
catch
    model = fitctree( ...
        X_train, ...
        y_train, ...
        'MaxNumSplits', ...
        cfg.classification.max_num_splits);
end

pred = predict(model, X_test);

end

function pred = train_python_catboost(X_train, y_train, X_test, cfg)

catboost = py.importlib.import_module('catboost');
np = py.importlib.import_module('numpy');

label_offset = min(y_train);
y_train_zero_based = y_train(:) - label_offset;

model = catboost.CatBoostClassifier(pyargs( ...
    'iterations', int32(cfg.classification.num_learning_cycles), ...
    'depth', int32(cfg.classification.depth), ...
    'learning_rate', cfg.classification.learning_rate, ...
    'loss_function', 'Logloss', ...
    'verbose', cfg.classification.verbose));

model.fit(np.array(X_train), np.array(y_train_zero_based));

py_pred = model.predict(np.array(X_test));
pred = double(py_pred);
pred = pred(:) + label_offset;

end

function weights = estimate_validation_weights(X_train, y_train, cfg, model_names)

n_models = numel(model_names);

class_labels = unique(y_train);
class_counts = ...
    arrayfun(@(label) sum(y_train == label), class_labels);

n_folds = min( ...
    cfg.classification.inner_weight_folds, ...
    min(class_counts));

if n_folds < 2
    weights = ones(1, n_models) / n_models;
    return;
end

cv = cvpartition(y_train, 'KFold', n_folds);

model_scores = zeros(cv.NumTestSets, n_models);

for fold_idx = 1:cv.NumTestSets
    train_idx = training(cv, fold_idx);
    test_idx = test(cv, fold_idx);

    X_inner_train = X_train(train_idx, :);
    y_inner_train = y_train(train_idx);
    X_inner_test = X_train(test_idx, :);
    y_inner_test = y_train(test_idx);

    for model_idx = 1:n_models
        model_pred = train_model_prediction( ...
            model_names{model_idx}, ...
            X_inner_train, ...
            y_inner_train, ...
            X_inner_test, ...
            cfg);

        model_scores(fold_idx, model_idx) = ...
            mean(model_pred(:) == y_inner_test(:));
    end
end

weights = mean(model_scores, 1);
weights = max(weights, eps);

if sum(weights) <= eps
    weights = ones(1, n_models) / n_models;
else
    weights = weights / sum(weights);
end

end

function pred = weighted_vote_predictions(all_pred, weights, class_labels)

n_samples = size(all_pred, 1);
pred = zeros(n_samples, 1);
class_labels = class_labels(:)';

for sample_idx = 1:n_samples
    class_scores = zeros(1, numel(class_labels));

    for class_idx = 1:numel(class_labels)
        class_scores(class_idx) = sum( ...
            weights(all_pred(sample_idx, :) == class_labels(class_idx)));
    end

    [best_score, best_idx] = max(class_scores);
    tied_idx = find(abs(class_scores - best_score) < 1e-12);

    if numel(tied_idx) > 1
        pred(sample_idx) = all_pred(sample_idx, 1);
    else
        pred(sample_idx) = class_labels(best_idx);
    end
end

end

function metrics = compute_metrics(y_true, y_pred)
    y_true = y_true(:);
    y_pred = y_pred(:);
    labels = unique([y_true; y_pred]);
    positive_label = labels(end);
    tp = sum(y_true == positive_label & y_pred == positive_label);
    tn = sum(y_true ~= positive_label & y_pred ~= positive_label);
    fp = sum(y_true ~= positive_label & y_pred == positive_label);
    fn = sum(y_true == positive_label & y_pred ~= positive_label);

    metrics = struct();
    metrics.accuracy = (tp + tn) / numel(y_true);
    metrics.precision = tp / max(tp + fp, eps);
    metrics.recall = tp / max(tp + fn, eps);
    metrics.f1 = 2 * metrics.precision * metrics.recall / max(metrics.precision + metrics.recall, eps);
    metrics.confusion_matrix = [tn, fp; fn, tp];
end
