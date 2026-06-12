function results = run_hybrid_csp_riemannian_ho_catboost(raw_eeg, events, cfg)
% RUN_HYBRID_CSP_RIEMANNIAN_HO_CATBOOST
% Hybrid EEG MI classification pipeline

if nargin >= 1 && (ischar(raw_eeg) || isstring(raw_eeg))

    mode = lower(string(raw_eeg));

    if nargin >= 2 && isstruct(events)
        cfg = merge_config(default_config(), events);
    else
        cfg = default_config();
    end 

    cfg = apply_runtime_overrides(cfg);

    results = run_physionet_batch(mode, cfg);
    return;
end

if nargin < 3 || isempty(cfg)
    cfg = default_config();
else
    cfg = merge_config(default_config(), cfg);
end

cfg = apply_runtime_overrides(cfg);

project_root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(project_root, 'PhysioNet_8Case_Benchmark', 'benchmark_framework', 'functions'));

if nargin < 2 || isempty(raw_eeg) || isempty(events)
    [raw_eeg, events] = create_synthetic_motor_imagery_data(cfg);
end

rng(cfg.seed, 'twister');

fprintf('Hybrid CSP-Riemannian-HO ensemble pipeline started.\n');
if cfg.runtime.fast_debug_mode
    fprintf(['Fast debug mode enabled: HO iterations=%d, population=%d, ' ...
        'block PCA cap=%d, final PCA cap=%d\n'], ...
        cfg.optimization.iterations, ...
        cfg.optimization.population_size, ...
        cfg.features.max_block_pca_components, ...
        cfg.features.max_final_pca_components);
end

[epochs, labels] = preprocess_pipeline(raw_eeg, events, cfg);

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

    %[X_train, X_test, channel_info] = ...
     %   select_adaptive_channels(X_train, y_train, X_test, cfg);

     channel_info = struct();
     channel_info.selected_channels = 1:size(X_train,1);
     channel_info.channel_scores = [];

     % Adaptive channel selection temporarily disabled

    [X_train, y_train, augmentation_info] = ...
        augment_training_epochs(X_train, y_train, cfg);

    [train_features, test_features, feature_names] = ...
        extract_all_feature_blocks(X_train, y_train, X_test, cfg);

    pipeline_is_fbcsp_mi = lower(string(cfg.pipeline_type)) == "fbcsp_mi";
    if pipeline_is_fbcsp_mi
        X_train_fused = train_features{1};
        X_test_fused = test_features{1};
        fusion_model = struct();
        fusion_model.mode = 'fbcsp_mi';
        fusion_model.block_names = feature_names;
        fusion_model.skipped_fusion = true;

        best_mask = true(1, size(X_train_fused, 2));
        ho_info = struct();
        ho_info.best_score = NaN;
        ho_info.convergence = [];

        fprintf('  Selected features after MI: %d/%d\n', ...
            sum(best_mask), numel(best_mask));

        X_train_selected = X_train_fused;
        X_test_selected = X_test_fused;

        [y_pred, classifier_info] = ...
            train_classifier_head( ...
                X_train_selected, ...
                y_train, ...
                X_test_selected, ...
                cfg, ...
                'linear_svm');
    else
        [X_train_fused, X_test_fused, fusion_model] = ...
            fuse_and_scale_features(train_features, test_features, cfg);

        [best_mask, ho_info] = ...
            binary_hippopotamus_optimizer(X_train_fused, y_train, cfg);

        fprintf('  Selected features after HO: %d/%d\n', ...
            sum(best_mask), numel(best_mask));

        X_train_selected = X_train_fused(:, best_mask);
        X_test_selected = X_test_fused(:, best_mask);

        [y_pred, classifier_info] = ...
            train_classifier_head( ...
                X_train_selected, ...
                y_train, ...
                X_test_selected, ...
                cfg);
    end

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

results.pipeline = pipeline_label(cfg.pipeline_type);

results.config = cfg;
results.fold_metrics = fold_metrics;

results.overall = overall;

results.true_labels = all_true;
results.predicted_labels = all_pred;

fprintf('\nOverall Accuracy: %.2f%%\n', overall.accuracy * 100);
fprintf('Overall F1-score: %.4f\n', overall.f1);

end


function cfg = default_config()

cfg = struct();

cfg.pipeline_type = 'spatial_only';


cfg.seed = 42;

cfg.fs = 160;

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
cfg.dataset.channel_indices = [];

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

cfg.preprocessing.bandpass = [4 32];
cfg.channel_mode = 'all';

cfg.preprocessing.butterworth_order = 4;

cfg.preprocessing.tmin = 0.5;
cfg.preprocessing.tmax = 3.5;
cfg.preprocessing.temporal_augmentation_windows = [
    0 3;
    0.5 3.5
];
cfg.preprocessing.use_temporal_augmentation = true;

% ICA disabled for stability/speed
cfg.preprocessing.use_ica = false;

cfg.preprocessing.ica_component_z_threshold = 3;

cfg.features.csp_bands = [
    8 12;
    12 16;
    16 22;
    22 30
];

cfg.features.csp_components = 6;
cfg.features.adaptive_csp_band_count = 3;
cfg.features.adaptive_channel_count = 12;
cfg.features.min_adaptive_channel_count = 6;

cfg.features.spectral_bands = ...
    [4 8
    8 12
    12 16
    16 20
    20 24
    24 30];
cfg.fbcsp_mi = struct( ...
    'bands', [4 8; 8 12; 12 16; 16 20; 20 24; 24 30], ...
    'top_k', 20, ...
    'mi_bins', 8, ...
    'filter_order', 4);

cfg.features.riemannian_epsilon = 1e-4;

cfg.features.sample_entropy_m = 2;
cfg.features.sample_entropy_r = 0.2;

cfg.features.sample_entropy_stride = 8;
cfg.features.sample_entropy_max_points = 80;
cfg.features.use_blockwise_pca = true;
cfg.features.block_pca_variance = 95;
cfg.features.max_block_pca_components = 12;
cfg.features.use_final_pca = true;
cfg.features.final_pca_variance = 95;
cfg.features.max_final_pca_components = 28;

cfg.optimization.population_size = 8;

cfg.optimization.iterations = 40;

cfg.optimization.threshold = 0.5;

cfg.optimization.mutation_rate = 0.02;

cfg.optimization.cv_folds = 3;

cfg.optimization.feature_penalty = 0.03;

cfg.classification.cv_folds = 3;

cfg.classification.decision_head = 'ensemble';
cfg.classification.ho_decision_head = 'ensemble';
cfg.classification.ensemble_backend = ...
    'linear_svm_only';
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

cfg.runtime.fast_debug_mode = false;

end

function cfg = apply_runtime_overrides(cfg)

if ~isfield(cfg, 'runtime') || ~isstruct(cfg.runtime)
    cfg.runtime = struct();
end

if ~isfield(cfg.runtime, 'fast_debug_mode') || ...
        isempty(cfg.runtime.fast_debug_mode)
    cfg.runtime.fast_debug_mode = false;
end

if cfg.runtime.fast_debug_mode
    cfg.optimization.iterations = 10;
    cfg.optimization.population_size = 4;
    cfg.features.final_pca_variance = ...
        min(cfg.features.final_pca_variance, 95);
    cfg.features.max_final_pca_components = ...
        min(cfg.features.max_final_pca_components, 12);
    cfg.features.max_block_pca_components = ...
        min(cfg.features.max_block_pca_components, 6);
end

end


function results = run_physionet_batch(mode, cfg)

switch mode

    case "physionet109"
        subject_ids = cfg.dataset.subjects;

    case "physionet3"
        subject_ids = 1:3;

    case "physionet1"
        subject_ids = 1;

    case "physionet10"
        subject_ids = 1:10;

    otherwise
        error( ...
            'Unknown mode: %s', ...
            mode);

end

ensure_directory(cfg.results.output_dir);

completed = ...
    existing_completed_subjects(cfg.results.csv_file);

rows = [];

fprintf('PhysioNet batch mode: %s\n', mode);

fprintf('Subjects requested: %d\n', ...
    numel(subject_ids));

fprintf('Cache: %s\n', ...
    cfg.dataset.cache_dir);

fprintf('CSV output: %s\n', ...
    cfg.results.csv_file);

for subject_id = subject_ids

    subject_name = sprintf('s%03d', subject_id);

    if cfg.results.resume && any(completed == subject_id)

        fprintf( ...
            'Skipping %s because it already exists in the CSV.\n', ...
            subject_name);

        continue;

    end

    row = initialize_result_row(subject_id, subject_name);

    try

        fprintf('\nProcessing %s\n', subject_name);

        [raw_eeg, events, subject_info] = ...
            load_physionet_subject_as_raw(subject_id, cfg);

        subject_cfg = cfg;

        subject_cfg.fs = subject_info.fs;

        subject_results = ...
            run_hybrid_csp_riemannian_ho_catboost( ...
                raw_eeg, ...
                events, ...
                subject_cfg);

        row.status = "OK";

        row.n_trials = subject_info.n_trials;

        row.n_channels = size(raw_eeg, 1);

        row.accuracy = subject_results.overall.accuracy;

        row.precision = subject_results.overall.precision;

        row.recall = subject_results.overall.recall;

        row.f1 = subject_results.overall.f1;

        row.mean_selected_features = ...
            mean([subject_results.fold_metrics.selected_feature_count]);

        row.total_features = ...
            subject_results.fold_metrics(1).total_feature_count;

        row.classifier_backend = ...
            string(subject_results.fold_metrics(1).classifier_backend);

    catch ME

        row.status = "ERROR";

        row.error_message = string(ME.message);

        fprintf('  ERROR: %s\n', ME.message);

    end

    append_result_row(cfg.results.csv_file, row);

    rows = [rows; row];

end

all_rows = read_result_rows(cfg.results.csv_file);

ok_rows = all_rows(all_rows.status == "OK", :);

results = struct();

results.mode = mode;

results.csv_file = cfg.results.csv_file;

results.rows = all_rows;

if ~isempty(ok_rows)

    results.summary = table( ...
        mean(ok_rows.accuracy), ...
        std(ok_rows.accuracy), ...
        median(ok_rows.accuracy), ...
        mean(ok_rows.f1), ...
        height(ok_rows), ...
        'VariableNames', ...
        {'MeanAccuracy', ...
         'StdAccuracy', ...
         'MedianAccuracy', ...
         'MeanF1', ...
         'CompletedSubjects'});

    disp(results.summary);

else

    results.summary = table();

end

save(cfg.results.mat_file, 'results');

end
function cfg = merge_config(base_cfg, override_cfg)

cfg = base_cfg;

override_fields = fieldnames(override_cfg);

for idx = 1:numel(override_fields)

    field_name = override_fields{idx};

    if isstruct(override_cfg.(field_name)) && ...
       isfield(cfg, field_name) && ...
       isstruct(cfg.(field_name))

        cfg.(field_name) = merge_config( ...
            cfg.(field_name), ...
            override_cfg.(field_name));

    else

        cfg.(field_name) = override_cfg.(field_name);

    end

end

end


function row = initialize_result_row(subject_id, subject_name)

row = table( ...
    subject_id, ...
    string(subject_name), ...
    "PENDING", ...
    0, ...
    0, ...
    NaN, ...
    NaN, ...
    NaN, ...
    NaN, ...
    NaN, ...
    NaN, ...
    string(""), ...
    string(""), ...
    'VariableNames', ...
    {'subject_id', ...
     'subject_name', ...
     'status', ...
     'n_trials', ...
     'n_channels', ...
     'accuracy', ...
     'precision', ...
     'recall', ...
     'f1', ...
     'mean_selected_features', ...
     'total_features', ...
     'classifier_backend', ...
     'error_message'});

end


function completed = existing_completed_subjects(csv_file)

completed = [];

if exist(csv_file, 'file')

    rows = read_result_rows(csv_file);

    completed = rows.subject_id(rows.status == "OK");

end

end


function rows = read_result_rows(csv_file)

if exist(csv_file, 'file')

    rows = readtable(csv_file, 'TextType', 'string');

else

    rows = table();

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


function ensure_directory(folder_path)

if ~exist(folder_path, 'dir')

    mkdir(folder_path);

end

end


function [raw_eeg, events, subject_info] = ...
    load_physionet_subject_as_raw(subject_id, cfg)

subject_code = sprintf('S%03d', subject_id);

subject_dir = fullfile(cfg.dataset.cache_dir, subject_code);

if ~exist(subject_dir, 'dir')

    error('Subject cache folder not found: %s', subject_dir);

end

raw_runs = {};

events = [];

sample_offset = 0;

fs = cfg.fs;

samples_per_trial = ...
    round((cfg.preprocessing.tmax - cfg.preprocessing.tmin) * fs);

left_count = 0;
right_count = 0;

for run_id = cfg.dataset.run_ids

    file_name = sprintf('%sR%02d.edf', subject_code, run_id);

    local_path = fullfile(subject_dir, file_name);

    if ~exist(local_path, 'file')

        if cfg.dataset.download_if_missing

            url = sprintf( ...
                'https://physionet.org/files/eegmmidb/1.0.0/%s/%s', ...
                subject_code, ...
                file_name);

            download_file_with_retry(url, local_path, 5);

        else

            error('Missing cached EDF file: %s', local_path);

        end

    end

    info = edfinfo(local_path);

    timetable_data = edfread(local_path);

    run_raw = timetable_to_numeric_matrix(timetable_data);

    if ~isempty(cfg.dataset.channel_indices)

        run_raw = run_raw(cfg.dataset.channel_indices, :);

    end

    raw_runs{end + 1} = run_raw;

    annotations = info.Annotations;

    disp(annotations)

    for ann_idx = 1:height(annotations)

        label = string(annotations.Annotations(ann_idx));

        if label ~= "T1" && label ~= "T2"
            continue;
        end

        onset_sec = seconds(annotations.Properties.RowTimes(ann_idx));

        start_idx = sample_offset + floor(onset_sec * fs) + 1;

        stop_idx = start_idx + samples_per_trial - 1;

        if stop_idx > sample_offset + size(run_raw, 2)
            continue;
        end

        class_label = 1;

        if label == "T2"

            class_label = 2;

            right_count = right_count + 1;

        else

            left_count = left_count + 1;

        end

        events = [events; start_idx, class_label];

    end

    sample_offset = sample_offset + size(run_raw, 2);

end

balanced_trials = min(left_count, right_count);

if balanced_trials < cfg.dataset.min_balanced_trials

    error( ...
        'Insufficient balanced trials for %s: left=%d, right=%d', ...
        subject_code, ...
        left_count, ...
        right_count);

end

raw_eeg = cell2mat(raw_runs);

events = balance_binary_events(events, balanced_trials);

subject_info = struct();

subject_info.subject_id = subject_id;

subject_info.subject_name = lower(subject_code);

subject_info.fs = fs;

subject_info.n_trials = size(events, 1);

end
function events_balanced = balance_binary_events(events, n_per_class)

class_1 = events(events(:, 2) == 1, :);

class_2 = events(events(:, 2) == 2, :);

events_balanced = [ ...
    class_1(1:n_per_class, :); ...
    class_2(1:n_per_class, :)];

[~, order] = sort(events_balanced(:, 1));

events_balanced = events_balanced(order, :);

end


function data = timetable_to_numeric_matrix(tt)

n_channels = width(tt);

n_records = height(tt);

first_cell = tt{1, 1};

samples_per_record = numel(first_cell{1});

data = zeros(n_channels, n_records * samples_per_record);

for channel_idx = 1:n_channels

    signal = zeros(n_records * samples_per_record, 1);

    cursor = 1;

    for record_idx = 1:n_records

        cell_value = tt{record_idx, channel_idx};

        chunk = cell_value{1};

        next_cursor = cursor + numel(chunk) - 1;

        signal(cursor:next_cursor) = chunk;

        cursor = next_cursor + 1;

    end

    data(channel_idx, :) = signal.';

end

end


function download_file_with_retry(url, local_path, max_attempts)

last_error = [];

for attempt = 1:max_attempts

    try

        websave(local_path, url);

        return;

    catch ME

        last_error = ME;

        pause(min(2^attempt, 30));

    end

end

rethrow(last_error);

end


function [raw_eeg, events] = create_synthetic_motor_imagery_data(cfg)

fs = cfg.fs;

n_channels = cfg.synthetic.n_channels;

n_trials = cfg.synthetic.n_trials;

trial_samples = round(cfg.synthetic.trial_duration_sec * fs);

total_samples = n_trials * trial_samples;

raw_eeg = 0.35 * randn(n_channels, total_samples);

events = zeros(n_trials, 2);

t = (0:trial_samples - 1) / fs;

for trial_idx = 1:n_trials

    label = 1 + mod(trial_idx, 2);

    start_idx = (trial_idx - 1) * trial_samples + 1;

    stop_idx = start_idx + trial_samples - 1;

    events(trial_idx, :) = [start_idx, label];

    if label == 1

        rhythm = ...
            sin(2 * pi * 10 * t) + ...
            0.4 * sin(2 * pi * 20 * t);

        channel_weights = [ ...
            ones(1, floor(n_channels / 2)), ...
            -0.4 * ones(1, n_channels - floor(n_channels / 2))];

    else

        rhythm = ...
            sin(2 * pi * 12 * t) + ...
            0.4 * sin(2 * pi * 24 * t);

        channel_weights = [ ...
            -0.4 * ones(1, floor(n_channels / 2)), ...
            ones(1, n_channels - floor(n_channels / 2))];

    end

    raw_eeg(:, start_idx:stop_idx) = ...
        raw_eeg(:, start_idx:stop_idx) + ...
        channel_weights(:) * rhythm;

end

end


function [epochs, labels] = preprocess_pipeline(raw_eeg, events, cfg)

fs = cfg.fs;

% Fixed channel-mode selection is applied before filtering so every
% downstream feature family sees the same sensor subset.
data = apply_fixed_channel_mode_to_raw(double(raw_eeg), cfg);

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

function data = apply_fixed_channel_mode_to_raw(data, cfg)

channel_mode = "all";
if isfield(cfg, 'channel_mode') && strlength(string(cfg.channel_mode)) > 0
    channel_mode = string(cfg.channel_mode);
end

if lower(channel_mode) == "all"
    return;
end

channel_labels = configured_channel_labels(size(data, 1));
[selected_idx, ~] = resolve_local_channel_mode_indices(channel_mode, channel_labels);
data = data(selected_idx, :);

end

function labels = configured_channel_labels(channel_count)

standard64 = [ ...
    "FC5","FC3","FC1","FCz","FC2","FC4","FC6", ...
    "C5","C3","C1","Cz","C2","C4","C6", ...
    "CP5","CP3","CP1","CPz","CP2","CP4","CP6", ...
    "FP1","FPz","FP2","AF7","AF3","AFz","AF4","AF8", ...
    "F7","F5","F3","F1","Fz","F2","F4","F6","F8", ...
    "FT7","FT8","T7","T8","T9","T10","TP7","TP8", ...
    "P7","P5","P3","P1","Pz","P2","P4","P6","P8", ...
    "PO7","PO3","POz","PO4","PO8","O1","Oz","O2", ...
    "Iz"];

if channel_count == 64
    labels = standard64;
elseif channel_count == 68
    labels = [standard64, "AUX1", "AUX2", "AUX3", "AUX4"];
else
    error('Channel mode selection supports 64 or 68 channels for this raw layout. Found %d.', channel_count);
end

end

function [selected_idx, selected_labels] = resolve_local_channel_mode_indices(channel_mode, channel_labels)

channel_mode = lower(string(channel_mode));
channel_labels = string(channel_labels(:));
normalized_labels = normalize_local_channel_labels(channel_labels);

switch channel_mode
    case "all"
        selected_idx = 1:numel(channel_labels);
    case "motor_basic"
        selected_idx = local_labels_to_indices(["C3","Cz","C4"], normalized_labels, channel_mode);
    case "motor_core"
        selected_idx = local_labels_to_indices(["C3","Cz","C4"], normalized_labels, channel_mode);
    case "sensorimotor"
        selected_idx = local_labels_to_indices( ...
            ["FC5","FC3","FC1","FCz","FC2","FC4","FC6", ...
             "C5","C3","C1","Cz","C2","C4","C6", ...
             "CP5","CP3","CP1","CPz","CP2","CP4","CP6"], ...
            normalized_labels, channel_mode);
    case "motor_extended"
        selected_idx = local_labels_to_indices( ...
            ["FC3","FC1","FCz","FC2","FC4", ...
             "C3","C1","Cz","C2","C4", ...
             "CP3","CP1","CPz","CP2","CP4"], ...
            normalized_labels, channel_mode);
    otherwise
        error('Unknown channel_mode: %s', channel_mode);
end

selected_labels = channel_labels(selected_idx);

end

function idx = local_labels_to_indices(target_labels, normalized_labels, channel_mode)
idx = zeros(1, numel(target_labels));
for k = 1:numel(target_labels)
    hit = find(normalized_labels == normalize_local_channel_labels(target_labels(k)), 1);
    if isempty(hit)
        error('Channel %s is not available for channel_mode %s.', target_labels(k), channel_mode);
    end
    idx(k) = hit;
end
end

function normalized = normalize_local_channel_labels(labels)
normalized = lower(regexprep(string(labels), '[^A-Za-z0-9]', ''));
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

function [train_blocks, test_blocks, feature_names] = ...
    extract_all_feature_blocks(X_train, y_train, X_test, cfg)

train_blocks = {};
test_blocks = {};
feature_names = {};

pipeline = lower(string(cfg.pipeline_type));

switch pipeline

    case {"spatial_only", "csp"}

        fprintf('  Extracting spatial-only CSP features...\n');

        [csp_train, csp_test] = extract_spatial_csp_features( ...
            X_train, ...
            y_train, ...
            X_test, ...
            cfg);

        [train_blocks, test_blocks, feature_names] = append_feature_block( ...
            train_blocks, ...
            test_blocks, ...
            feature_names, ...
            csp_train, ...
            csp_test, ...
            'CSP');

    case "spatial_spectral"

        fprintf('  Extracting spatial + spectral features...\n');

        [csp_train, csp_test] = extract_spatial_csp_features( ...
            X_train, ...
            y_train, ...
            X_test, ...
            cfg);
        spectral_train = extract_spectral_features(X_train, cfg);
        spectral_test = extract_spectral_features(X_test, cfg);

        [train_blocks, test_blocks, feature_names] = append_feature_block( ...
            train_blocks, test_blocks, feature_names, ...
            csp_train, csp_test, 'CSP');
        [train_blocks, test_blocks, feature_names] = append_feature_block( ...
            train_blocks, test_blocks, feature_names, ...
            spectral_train, spectral_test, 'Spectral');

    case {"spatial_temporal", "csp_temporal"}

        fprintf('  Extracting spatial + temporal features...\n');

        [csp_train, csp_test] = extract_spatial_csp_features( ...
            X_train, ...
            y_train, ...
            X_test, ...
            cfg);
        temporal_train = extract_temporal_features(X_train, cfg);
        temporal_test = extract_temporal_features(X_test, cfg);

        [train_blocks, test_blocks, feature_names] = append_feature_block( ...
            train_blocks, test_blocks, feature_names, ...
            csp_train, csp_test, 'CSP');
        [train_blocks, test_blocks, feature_names] = append_feature_block( ...
            train_blocks, test_blocks, feature_names, ...
            temporal_train, temporal_test, 'Temporal');

    case "spatial_spectral_temporal"

        fprintf('  Extracting spatial + spectral + temporal features...\n');

        [csp_train, csp_test] = extract_spatial_csp_features( ...
            X_train, ...
            y_train, ...
            X_test, ...
            cfg);
        spectral_train = extract_spectral_features(X_train, cfg);
        spectral_test = extract_spectral_features(X_test, cfg);
        temporal_train = extract_temporal_features(X_train, cfg);
        temporal_test = extract_temporal_features(X_test, cfg);

        [train_blocks, test_blocks, feature_names] = append_feature_block( ...
            train_blocks, test_blocks, feature_names, ...
            csp_train, csp_test, 'CSP');
        [train_blocks, test_blocks, feature_names] = append_feature_block( ...
            train_blocks, test_blocks, feature_names, ...
            spectral_train, spectral_test, 'Spectral');
        [train_blocks, test_blocks, feature_names] = append_feature_block( ...
            train_blocks, test_blocks, feature_names, ...
            temporal_train, temporal_test, 'Temporal');

    case "spatial_spectral_temporal_riemannian"

        fprintf('  Extracting spatial + spectral + temporal + Riemannian features...\n');

        [csp_train, csp_test] = extract_spatial_csp_features( ...
            X_train, ...
            y_train, ...
            X_test, ...
            cfg);
        spectral_train = extract_spectral_features(X_train, cfg);
        spectral_test = extract_spectral_features(X_test, cfg);
        temporal_train = extract_temporal_features(X_train, cfg);
        temporal_test = extract_temporal_features(X_test, cfg);
        [riem_train, riem_test] = extract_riemannian_features( ...
            X_train, ...
            X_test, ...
            cfg);

        [train_blocks, test_blocks, feature_names] = append_feature_block( ...
            train_blocks, test_blocks, feature_names, ...
            csp_train, csp_test, 'CSP');
        [train_blocks, test_blocks, feature_names] = append_feature_block( ...
            train_blocks, test_blocks, feature_names, ...
            spectral_train, spectral_test, 'Spectral');
        [train_blocks, test_blocks, feature_names] = append_feature_block( ...
            train_blocks, test_blocks, feature_names, ...
            temporal_train, temporal_test, 'Temporal');
        [train_blocks, test_blocks, feature_names] = append_feature_block( ...
            train_blocks, test_blocks, feature_names, ...
            riem_train, riem_test, 'Riemannian tangent space');

    case "csp_de"

        fprintf('  Extracting CSP + Differential Entropy features...\n');

        [csp_train, csp_test] = extract_spatial_csp_features( ...
            X_train, ...
            y_train, ...
            X_test, ...
            cfg);

        de_train = extract_differential_entropy_features(X_train);
        de_test = extract_differential_entropy_features(X_test);

        [train_blocks, test_blocks, feature_names] = append_feature_block( ...
            train_blocks, test_blocks, feature_names, ...
            csp_train, csp_test, 'CSP');
        [train_blocks, test_blocks, feature_names] = append_feature_block( ...
            train_blocks, test_blocks, feature_names, ...
            de_train, de_test, 'Differential entropy');

    case "fbcsp"

        fprintf('  Extracting FBCSP features only...\n');

        [fbcsp_train, fbcsp_test] = extract_csp_features( ...
            X_train, ...
            y_train, ...
            X_test, ...
            cfg);

        [train_blocks, test_blocks, feature_names] = append_feature_block( ...
            train_blocks, ...
            test_blocks, ...
            feature_names, ...
            fbcsp_train, ...
            fbcsp_test, ...
            'FBCSP');

    case "csp_fbcsp"

        fprintf('  Extracting CSP + FBCSP features...\n');

        [csp_train, csp_test] = extract_spatial_csp_features( ...
            X_train, ...
            y_train, ...
            X_test, ...
            cfg);

        [fbcsp_train, fbcsp_test] = extract_csp_features( ...
            X_train, ...
            y_train, ...
            X_test, ...
            cfg);

        [train_blocks, test_blocks, feature_names] = append_feature_block( ...
            train_blocks, test_blocks, feature_names, ...
            csp_train, csp_test, 'CSP');
        [train_blocks, test_blocks, feature_names] = append_feature_block( ...
            train_blocks, test_blocks, feature_names, ...
            fbcsp_train, fbcsp_test, 'FBCSP');

    case "fbcsp_mi"

        fprintf('  Extracting FBCSP + MI features...\n');

        fbcsp_mi_bundle = extract_fbcsp_mi_features( ...
            X_train, ...
            y_train, ...
            X_test, ...
            cfg);

        [train_blocks, test_blocks, feature_names] = append_feature_block( ...
            train_blocks, test_blocks, feature_names, ...
            fbcsp_mi_bundle.train_X, ...
            fbcsp_mi_bundle.test_X, ...
            'FBCSP-MI');

    case {"csp_riemannian", "riemannian"}

        if pipeline == "riemannian"
            fprintf('  Extracting Riemannian features only...\n');
        else
            fprintf('  Extracting CSP + Riemannian features...\n');
        end

        if pipeline == "csp_riemannian"
            [csp_train, csp_test] = extract_spatial_csp_features( ...
                X_train, ...
                y_train, ...
                X_test, ...
                cfg);
            [train_blocks, test_blocks, feature_names] = append_feature_block( ...
                train_blocks, test_blocks, feature_names, ...
                csp_train, csp_test, 'CSP');
        end

        [riem_train, riem_test] = extract_riemannian_features( ...
            X_train, ...
            X_test, ...
            cfg);

        [train_blocks, test_blocks, feature_names] = append_feature_block( ...
            train_blocks, test_blocks, feature_names, ...
            riem_train, riem_test, 'Riemannian tangent space');

    otherwise

        fprintf('  Extracting hybrid features...\n');

        [csp_train, csp_test] = extract_csp_features( ...
            X_train, ...
            y_train, ...
            X_test, ...
            cfg);

        [riem_train, riem_test] = extract_riemannian_features( ...
            X_train, ...
            X_test, ...
            cfg);

        spectral_train = extract_spectral_features(X_train, cfg);
        spectral_test = extract_spectral_features(X_test, cfg);

        temporal_train = extract_temporal_features(X_train, cfg);
        temporal_test = extract_temporal_features(X_test, cfg);

        [train_blocks, test_blocks, feature_names] = append_feature_block( ...
            train_blocks, test_blocks, feature_names, ...
            csp_train, csp_test, 'CSP/FBCSP');
        [train_blocks, test_blocks, feature_names] = append_feature_block( ...
            train_blocks, test_blocks, feature_names, ...
            riem_train, riem_test, 'Riemannian tangent space');
        [train_blocks, test_blocks, feature_names] = append_feature_block( ...
            train_blocks, test_blocks, feature_names, ...
            spectral_train, spectral_test, 'Spectral');
        [train_blocks, test_blocks, feature_names] = append_feature_block( ...
            train_blocks, test_blocks, feature_names, ...
            temporal_train, temporal_test, 'Temporal');

end

end

function [train_blocks, test_blocks, feature_names] = append_feature_block( ...
    train_blocks, test_blocks, feature_names, train_block, test_block, block_name)

train_block = cho_validate_numeric_matrix(train_block, sprintf('feature block train (%s)', block_name));
test_block = cho_validate_numeric_matrix(test_block, sprintf('feature block test (%s)', block_name));
% cho_trace_matrix(sprintf('append_feature_block train (%s)', block_name), train_block);
% cho_trace_matrix(sprintf('append_feature_block test (%s)', block_name), test_block);

train_blocks{end+1} = train_block;
test_blocks{end+1} = test_block;
feature_names{end+1} = block_name;

end

function [train_features, test_features] = extract_spatial_csp_features( ...
    X_train, y_train, X_test, cfg)

csp_cfg = cfg;
csp_cfg.features.csp_bands = single_csp_band(cfg);

[train_features, test_features] = extract_csp_features( ...
    X_train, ...
    y_train, ...
    X_test, ...
    csp_cfg);

end

function feature_bundle = extract_fbcsp_mi_features(X_train, y_train, X_test, cfg)
X_train = cho_validate_numeric_matrix(X_train, 'FBCSP-MI input X_train');
X_test = cho_validate_numeric_matrix(X_test, 'FBCSP-MI input X_test');
%cho_trace_matrix('FBCSP-MI input X_train', X_train);
%cho_trace_matrix('FBCSP-MI input X_test', X_test);

fbcsp_cfg = resolve_cho_fbcsp_mi_cfg(cfg);
bands = fbcsp_cfg.bands;

labels = unique(y_train);
if numel(labels) ~= 2
    error('FBCSP-MI requires exactly two classes.');
end

band_train_blocks = cell(1, size(bands, 1));
band_test_blocks = cell(1, size(bands, 1));
band_feature_widths = zeros(1, size(bands, 1));
band_train_rows = zeros(1, size(bands, 1));
band_test_rows = zeros(1, size(bands, 1));

for band_idx = 1:size(bands, 1)
    band = bands(band_idx, :);
    [X_train_band, X_test_band] = cho_bandpass_epochs( ...
        X_train, X_test, cfg.fs, band, fbcsp_cfg.filter_order);

    band_cfg = cfg;
    band_cfg.features.csp_bands = band;
    band_cfg.features.csp_components = resolve_cho_csp_components(cfg);

    [train_band_features, test_band_features] = compute_csp_feature_pair( ...
        X_train_band, ...
        y_train, ...
        X_test_band, ...
        band_cfg.features.csp_components, ...
        labels);

    train_band_features = cho_validate_numeric_matrix( ...
        train_band_features, sprintf('FBCSP band %d train', band_idx));
    test_band_features = cho_validate_numeric_matrix( ...
        test_band_features, sprintf('FBCSP band %d test', band_idx));

    band_train_blocks{band_idx} = train_band_features;
    band_test_blocks{band_idx} = test_band_features;
    band_feature_widths(band_idx) = size(train_band_features, 2);
    band_train_rows(band_idx) = size(train_band_features, 1);
    band_test_rows(band_idx) = size(test_band_features, 1);

    %cho_trace_matrix(sprintf('FBCSP band %d train [%g %g]', band_idx, band(1), band(2)), train_band_features);
    %cho_trace_matrix(sprintf('FBCSP band %d test [%g %g]', band_idx, band(1), band(2)), test_band_features);

    fprintf('  FBCSP band %d train size: [%d %d] | test size: [%d %d]\n', ...
        band_idx, size(train_band_features, 1), size(train_band_features, 2), ...
        size(test_band_features, 1), size(test_band_features, 2));
end

if numel(unique(band_feature_widths)) ~= 1
    error('FBCSP-MI band widths are inconsistent: %s', mat2str(band_feature_widths));
end
if numel(unique(band_train_rows)) ~= 1 || numel(unique(band_test_rows)) ~= 1
    error('FBCSP-MI band row counts are inconsistent: train=%s test=%s', ...
        mat2str(band_train_rows), mat2str(band_test_rows));
end

train_X = cat(2, band_train_blocks{:});
test_X = cat(2, band_test_blocks{:});

train_X = cho_validate_numeric_matrix(train_X, 'FBCSP-MI concatenated train_X');
test_X = cho_validate_numeric_matrix(test_X, 'FBCSP-MI concatenated test_X');

%cho_trace_matrix('FBCSP-MI concatenated train_X', train_X);
%cho_trace_matrix('FBCSP-MI concatenated test_X', test_X);

[selected_idx, mi_scores, ranked_idx] = select_mutual_information_features( ...
    train_X, y_train, fbcsp_cfg.top_k, fbcsp_cfg.mi_bins);
fprintf('  FBCSP-MI selected %d/%d features | indices=%s\n', ...
    numel(selected_idx), numel(mi_scores), mat2str(selected_idx));

feature_bundle = struct();
feature_bundle.train_X = train_X(:, selected_idx);
feature_bundle.test_X = test_X(:, selected_idx);
feature_bundle.train_Y = y_train;
feature_bundle.test_Y = [];
feature_bundle.selected_feature_idx = selected_idx;
feature_bundle.ranked_feature_idx = ranked_idx;
feature_bundle.mi_scores = mi_scores;
feature_bundle.raw_feature_count = size(train_X, 2);
feature_bundle.selected_feature_count = numel(selected_idx);
feature_bundle.selected_bands = bands;
feature_bundle.fbcsp_mi = fbcsp_cfg;
feature_bundle.feature_blocks = {'fbcsp_mi'};
end

function cfg_out = resolve_cho_fbcsp_mi_cfg(cfg)
cfg_out = struct();
cfg_out.bands = [4 8; 8 12; 12 16; 16 20; 20 24; 24 30];
cfg_out.top_k = 20;
cfg_out.mi_bins = 8;
cfg_out.filter_order = 4;

if isfield(cfg, 'fbcsp_mi') && isstruct(cfg.fbcsp_mi)
    if isfield(cfg.fbcsp_mi, 'bands') && ~isempty(cfg.fbcsp_mi.bands)
        cfg_out.bands = cfg.fbcsp_mi.bands;
    end
    if isfield(cfg.fbcsp_mi, 'top_k') && ~isempty(cfg.fbcsp_mi.top_k)
        cfg_out.top_k = cfg.fbcsp_mi.top_k;
    end
    if isfield(cfg.fbcsp_mi, 'mi_bins') && ~isempty(cfg.fbcsp_mi.mi_bins)
        cfg_out.mi_bins = cfg.fbcsp_mi.mi_bins;
    end
    if isfield(cfg.fbcsp_mi, 'filter_order') && ~isempty(cfg.fbcsp_mi.filter_order)
        cfg_out.filter_order = cfg.fbcsp_mi.filter_order;
    end
end

cfg_out.top_k = max(1, round(cfg_out.top_k));
cfg_out.mi_bins = max(4, round(cfg_out.mi_bins));
cfg_out.filter_order = max(1, round(cfg_out.filter_order));
end

function n_components = resolve_cho_csp_components(cfg)
if isfield(cfg, 'features') && isfield(cfg.features, 'csp_components') && ~isempty(cfg.features.csp_components)
    n_components = cfg.features.csp_components;
elseif isfield(cfg, 'csp_components') && ~isempty(cfg.csp_components)
    n_components = cfg.csp_components;
else
    n_components = 4;
end
end

function [X_train_f, X_test_f] = cho_bandpass_epochs(X_train, X_test, fs, band, filter_order)
[b, a] = butter(filter_order, band / (fs / 2), 'bandpass');
X_train_f = zeros(size(X_train));
X_test_f = zeros(size(X_test));

for epoch_idx = 1:size(X_train, 3)
    X_train_f(:, :, epoch_idx) = filtfilt(b, a, double(X_train(:, :, epoch_idx)') )';
end
for epoch_idx = 1:size(X_test, 3)
    X_test_f(:, :, epoch_idx) = filtfilt(b, a, double(X_test(:, :, epoch_idx)') )';
end
end

function band = single_csp_band(cfg)

if isfield(cfg, 'features') && ...
   isfield(cfg.features, 'single_csp_band') && ...
   numel(cfg.features.single_csp_band) == 2
    band = cfg.features.single_csp_band;
elseif isfield(cfg, 'preprocessing') && ...
       isfield(cfg.preprocessing, 'bandpass') && ...
       numel(cfg.preprocessing.bandpass) == 2
    band = cfg.preprocessing.bandpass;
else
    band = [8 30];
end

end

function label = pipeline_label(pipeline_type)
switch lower(string(pipeline_type))
    case "spatial_only"
        label = 'Spatial only';
    case "spatial_spectral"
        label = 'Spatial + Spectral';
    case "spatial_temporal"
        label = 'Spatial + Temporal';
    case "spatial_spectral_temporal"
        label = 'Spatial + Spectral + Temporal';
    case "spatial_spectral_temporal_riemannian"
        label = 'Spatial + Spectral + Temporal + Riemannian';
    case "csp"
        label = 'CSP';
    case "csp_de"
        label = 'CSP + Differential Entropy';
    case "csp_temporal"
        label = 'CSP + Temporal';
    case "csp_riemannian"
        label = 'CSP + Riemannian';
    case "csp_fbcsp"
        label = 'CSP + FBCSP';
    case "fbcsp_mi"
        label = 'FBCSP + MI';
    case "fbcsp"
        label = 'FBCSP';
    case "riemannian"
        label = 'Riemannian';
    otherwise
        label = char(string(pipeline_type));
end
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

global_cov = (cov_1 + cov_2) / 2;

beta = 0.15;

cov_1 = ...
    (1 - beta) * cov_1 + ...
    beta * global_cov;

cov_2 = ...
    (1 - beta) * cov_2 + ...
    beta * global_cov;

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
    alpha = 0.12;

    target = trace(c) / size(c,1) * eye(size(c));

    c = (1 - alpha) * c + alpha * target;

    % Numerical stabilization
    c = c + 1e-4 * eye(size(c));

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
        alpha = 0.12;

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
    for iter = 1:10
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
        spd_matrix = ...
            inv_sqrt_ref * covs(:, :, epoch_idx) * inv_sqrt_ref;

        spd_matrix = ...
            (spd_matrix + spd_matrix') / 2;

        spd_matrix = ...
            spd_matrix + 1e-4 * eye(size(spd_matrix));

        tangent = real(logm(spd_matrix));
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
    features = zeros(n_epochs, n_bands * n_channels * 2);

    write_idx = 1;
    for band_idx = 1:n_bands
        [b, a] = butter(4, bands(band_idx, :) / (fs / 2), 'bandpass');
        X_band = filter_epochs(X, b, a);
        de_block = extract_differential_entropy_features(X_band);
        bp_block = extract_band_power_features(X_band);
        band_block = [de_block, bp_block];
        cols = write_idx:(write_idx + size(band_block, 2) - 1);
        features(:, cols) = band_block;
        write_idx = cols(end) + 1;
    end
end

function features = extract_differential_entropy_features(X)
    n_epochs = size(X, 3);
    n_channels = size(X, 1);
    features = zeros(n_epochs, n_channels);

    for epoch_idx = 1:n_epochs
        epoch = squeeze(X(:, :, epoch_idx))';
        channel_var = var(epoch, 0, 1) + eps;
        features(epoch_idx, :) = 0.5 * log(2 * pi * exp(1) * channel_var);
    end
end

function features = extract_band_power_features(X)
    n_epochs = size(X, 3);
    n_channels = size(X, 1);
    features = zeros(n_epochs, n_channels);

    for epoch_idx = 1:n_epochs
        epoch = squeeze(X(:, :, epoch_idx))';
        features(epoch_idx, :) = log(mean(epoch .^ 2, 1) + eps);
    end
end

function features = extract_temporal_features(X, cfg)
    hjorth_features = extract_hjorth_features(X);
    sample_entropy_features = extract_sample_entropy_features(X, cfg);
    features = [hjorth_features, sample_entropy_features];
end

function features = extract_hjorth_features(X)
    n_epochs = size(X, 3);
    n_channels = size(X, 1);
    features = zeros(n_epochs, n_channels * 3);

    for epoch_idx = 1:n_epochs
        epoch = squeeze(X(:, :, epoch_idx))';
        dx = diff(epoch, 1, 1);
        ddx = diff(dx, 1, 1);

        var_x = var(epoch, 0, 1) + eps;
        var_dx = var(dx, 0, 1) + eps;
        var_ddx = var(ddx, 0, 1) + eps;

        activity = var_x;
        mobility = sqrt(var_dx ./ var_x);
        complexity = sqrt(var_ddx ./ var_dx) ./ (mobility + eps);

        features(epoch_idx, :) = [activity, mobility, complexity];
    end
end

function features = extract_sample_entropy_features(X, cfg)
    n_epochs = size(X, 3);
    n_channels = size(X, 1);
    features = zeros(n_epochs, n_channels);

    for epoch_idx = 1:n_epochs
        epoch = squeeze(X(:, :, epoch_idx))';
        sample_entropy = zeros(1, n_channels);
        for channel_idx = 1:n_channels
            sample_entropy(channel_idx) = compute_sample_entropy( ...
                epoch(:, channel_idx), ...
                cfg.features.sample_entropy_m, ...
                cfg.features.sample_entropy_r, ...
                cfg.features.sample_entropy_stride, ...
                cfg.features.sample_entropy_max_points);
        end
        features(epoch_idx, :) = sample_entropy;
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
    total_features_before_pca = 0;
    total_features_after_block_pca = 0;

    for block_idx = 1:n_blocks
        train_block = train_blocks{block_idx};
        test_block = test_blocks{block_idx};
        block_name = sprintf('fusion block %d', block_idx);

        train_block = cho_validate_numeric_matrix(train_block, [block_name ' train input']);
        test_block = cho_validate_numeric_matrix(test_block, [block_name ' test input']);
        %cho_trace_matrix([block_name ' train input'], train_block);
        %cho_trace_matrix([block_name ' test input'], test_block);

        block_model = struct();
        [train_block, test_block, cleanup_info] = ...
            prune_feature_columns(train_block, test_block);
        block_model.cleanup = cleanup_info;
        total_features_before_pca = ...
            total_features_before_pca + cleanup_info.feature_count_after_cleanup;

        block_model.mu = mean(train_block, 1);
        block_model.sigma = std(train_block, 0, 1);
        block_model.sigma(block_model.sigma == 0) = 1;

        train_block = ...
            (train_block - block_model.mu) ./ block_model.sigma;
        test_block = ...
            (test_block - block_model.mu) ./ block_model.sigma;

        train_block = cho_validate_numeric_matrix(train_block, [block_name ' standardized train']);
        test_block = cho_validate_numeric_matrix(test_block, [block_name ' standardized test']);
        %cho_trace_matrix([block_name ' standardized train'], train_block);
        %cho_trace_matrix([block_name ' standardized test'], test_block);

        if cfg.features.use_blockwise_pca && size(train_block, 2) > 1 && ...
                size(train_block, 1) > 1
            cho_trace_matrix([block_name ' before block PCA'], train_block);
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

        total_features_after_block_pca = ...
            total_features_after_block_pca + size(train_block, 2);

        transformed_train_blocks{block_idx} = train_block;
        transformed_test_blocks{block_idx} = test_block;
        model.blocks{block_idx} = block_model;
    end

    train_row_counts = cellfun(@(m) size(m, 1), transformed_train_blocks);
    test_row_counts = cellfun(@(m) size(m, 1), transformed_test_blocks);
    if numel(unique(train_row_counts)) ~= 1
        error('Fusion train row counts are inconsistent: %s', mat2str(train_row_counts));
    end
    if numel(unique(test_row_counts)) ~= 1
        error('Fusion test row counts are inconsistent: %s', mat2str(test_row_counts));
    end
    fprintf('  Fusion row counts train=%s test=%s\n', ...
        mat2str(train_row_counts), mat2str(test_row_counts));

    X_train_fused = cell2mat(transformed_train_blocks);
    X_test_fused = cell2mat(transformed_test_blocks);
    X_train_fused = cho_validate_numeric_matrix(X_train_fused, 'fused train before cleanup');
    X_test_fused = cho_validate_numeric_matrix(X_test_fused, 'fused test before cleanup');
    %cho_trace_matrix('fused train before cleanup', X_train_fused);
    %cho_trace_matrix('fused test before cleanup', X_test_fused);

    fprintf('  Feature count before PCA: %d\n', ...
        total_features_before_pca);

    [X_train_fused, X_test_fused, final_cleanup] = ...
        prune_feature_columns(X_train_fused, X_test_fused);
    model.final_cleanup = final_cleanup;
    model.feature_count_before_pca = total_features_before_pca;
    model.feature_count_after_block_pca = total_features_after_block_pca;

    model.final_mu = mean(X_train_fused, 1);
    model.final_sigma = std(X_train_fused, 0, 1);
    model.final_sigma(model.final_sigma == 0) = 1;

    X_train_fused = ...
        (X_train_fused - model.final_mu) ./ model.final_sigma;
    X_test_fused = ...
        (X_test_fused - model.final_mu) ./ model.final_sigma;
    X_train_fused = cho_validate_numeric_matrix(X_train_fused, 'fused train standardized');
    X_test_fused = cho_validate_numeric_matrix(X_test_fused, 'fused test standardized');
    %cho_trace_matrix('fused train standardized', X_train_fused);
    %cho_trace_matrix('fused test standardized', X_test_fused);

    if cfg.features.use_final_pca && size(X_train_fused, 2) > 1 && ...
            size(X_train_fused, 1) > 1
        cho_trace_matrix('fused train before final PCA', X_train_fused);
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

    fprintf('  Feature count after PCA: %d\n', ...
        size(X_train_fused, 2));

    model.total_components = size(X_train_fused, 2);
end

function [train_block, test_block, info] = ...
    prune_feature_columns(train_block, test_block)

info = struct();
info.original_feature_count = size(train_block, 2);
info.zero_variance_removed = 0;
info.duplicate_removed = 0;
info.feature_count_after_cleanup = info.original_feature_count;

if isempty(train_block)
    return;
end

variance_mask = var(train_block, 0, 1) > 0;
info.zero_variance_removed = ...
    info.original_feature_count - sum(variance_mask);

if any(variance_mask)
    train_block = train_block(:, variance_mask);
    test_block = test_block(:, variance_mask);
else
    train_block = zeros(size(train_block, 1), 1);
    test_block = zeros(size(test_block, 1), 1);
    info.feature_count_after_cleanup = 1;
    return;
end

try
    [~, unique_idx] = unique(train_block', 'rows', 'stable');
    unique_idx = sort(unique_idx);
    info.duplicate_removed = size(train_block, 2) - numel(unique_idx);
    train_block = train_block(:, unique_idx);
    test_block = test_block(:, unique_idx);
catch
    info.duplicate_removed = 0;
end

if isempty(train_block)
    train_block = zeros(size(train_block, 1), 1);
    test_block = zeros(size(test_block, 1), 1);
end

info.feature_count_after_cleanup = size(train_block, 2);

end

function [best_mask, info] = binary_hippopotamus_optimizer(X, y, cfg)
    X = cho_validate_numeric_matrix(X, 'HO input X');
    y = cho_validate_numeric_matrix(y(:), 'HO input y');
    %cho_trace_matrix('HO input X', X);
    %cho_trace_matrix('HO input y', y);

    n_features = size(X, 2);
    pop_size = cfg.optimization.population_size;
    iterations = cfg.optimization.iterations;
    threshold = cfg.optimization.threshold;
    mutation_rate = cfg.optimization.mutation_rate;

    rng(cfg.seed);

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

X_train = cho_validate_numeric_matrix(X_train, 'classifier input X_train');
X_test = cho_validate_numeric_matrix(X_test, 'classifier input X_test');
%cho_trace_matrix('classifier input X_train', X_train);
%cho_trace_matrix('classifier input X_test', X_test);

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
    case {"linear_svm", "svm"}
        if isfield(cfg, 'pipeline_type') && strcmpi(string(cfg.pipeline_type), 'fbcsp_mi')
            %cho_trace_matrix('classifier fitcsvm X_train', X_train);
            %cho_trace_matrix('classifier fitcsvm X_test', X_test);
            svm_model = fitcsvm( ...
                X_train, ...
                y_train, ...
                'KernelFunction', 'linear', ...
                'Standardize', false);
            pred = predict(svm_model, X_test);
        else
            pred = train_svm_and_predict(X_train, y_train, X_test);
        end
        info.backend = 'linear_svm';
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

model_names = {'linear_svm'};

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

X_train = cho_validate_numeric_matrix(X_train, 'train_svm_and_predict X_train');
X_test = cho_validate_numeric_matrix(X_test, 'train_svm_and_predict X_test');
%cho_trace_matrix('train_svm_and_predict X_train', X_train);
%cho_trace_matrix('train_svm_and_predict X_test', X_test);

[X_train, X_test] = auto_reduce_dimensions_for_svm(X_train, X_test);

svm_model = fitcsvm( ...
    X_train, ...
    y_train, ...
    'KernelFunction', 'linear', ...
    'BoxConstraint', 0.5, ...
    'Standardize', false);

pred = predict(svm_model, X_test);

end

function pred = train_rbf_svm_and_predict(X_train, y_train, X_test)

[X_train, X_test] = auto_reduce_dimensions_for_svm(X_train, X_test);

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

function [X_train_reduced, X_test_reduced] = ...
    auto_reduce_dimensions_for_svm(X_train, X_test)

X_train_reduced = X_train;
X_test_reduced = X_test;

X_train_reduced = cho_validate_numeric_matrix(X_train_reduced, 'auto_reduce_dimensions_for_svm X_train');
X_test_reduced = cho_validate_numeric_matrix(X_test_reduced, 'auto_reduce_dimensions_for_svm X_test');
%cho_trace_matrix('auto_reduce_dimensions_for_svm X_train', X_train_reduced);
%cho_trace_matrix('auto_reduce_dimensions_for_svm X_test', X_test_reduced);

if size(X_train, 2) <= size(X_train, 1) || size(X_train, 2) <= 1 || ...
        size(X_train, 1) <= 1
    return;
end

[coeff, score, ~, ~, ~, mu] = pca(X_train);
n_components = max(1, min(size(X_train, 1) - 1, size(score, 2)));

X_train_reduced = score(:, 1:n_components);
X_test_reduced = ...
    (X_test - mu) * coeff(:, 1:n_components);

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

function tf = cho_debug_trace_enabled()
raw = lower(strtrim(char(string(getenv('CHO_DEBUG_TRACE')))));
if isempty(raw)
    tf = true;
else
    tf = any(strcmp(raw, {'1','true','yes','on'}));
end
end

function value = cho_validate_numeric_matrix(value, label)
if ~cho_debug_trace_enabled()
    if isnumeric(value) || islogical(value)
        value = double(value);
        return;
    end
end

ws = whos('value');
shape = mat2str(size(value));
if isempty(ws)
    ws_desc = 'whos: <empty>';
else
    ws_desc = sprintf('whos: name=%s class=%s size=%s bytes=%d', ...
        ws.name, ws.class, mat2str(ws.size), ws.bytes);
end

%fprintf('[CHO-DBG] %s | class=%s | size=%s | ndims=%d | numeric=%d | cell=%d | table=%d | string=%d | categorical=%d | logical=%d | %s\n', ...
  %  label, class(value), shape, ndims(value), isnumeric(value), iscell(value), istable(value), isstring(value), iscategorical(value), islogical(value), ws_desc);

if istable(value) || iscell(value) || isstring(value) || iscategorical(value)
    error('Cho numeric guard failed for %s: class=%s size=%s.', label, class(value), shape);
end
if ~(isnumeric(value) || islogical(value))
    error('Cho numeric guard failed for %s: class=%s size=%s.', label, class(value), shape);
end
if ndims(value) ~= 2
    error('Cho numeric guard failed for %s: expected 2-D matrix, got ndims=%d.', label, ndims(value));
end

value = double(value);
end

function cho_trace_matrix(label, value)
if ~cho_debug_trace_enabled()
    return;
end

ws = whos('value');
if isempty(ws)
    ws_desc = 'whos: <empty>';
else
    ws_desc = sprintf('whos: name=%s class=%s size=%s bytes=%d', ...
        ws.name, ws.class, mat2str(ws.size), ws.bytes);
end

%fprintf('[CHO-DBG] %s | class=%s | size=%s | ndims=%d | numeric=%d | cell=%d | table=%d | string=%d | categorical=%d | logical=%d | %s\n', ...
 %   label, class(value), mat2str(size(value)), ndims(value), isnumeric(value), iscell(value), istable(value), isstring(value), iscategorical(value), islogical(value), ws_desc);
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

for n = [8 12 14 16]

    cfg = struct();
    cfg.features.csp_components = n;

    cfg.results.resume = false;

    fprintf('\n=============================\n');
    fprintf('Running CSP = %d\n', n);
    fprintf('=============================\n');

    results = run_csp_catboost('smoke', cfg);

end