function run_case04_riemannian_rf_common(stage_name, subject_list, force_rerun)
% run_case04_riemannian_rf_common Complete standalone pipeline for case04_riemannian_rf_common
%
% This file contains the ENTIRE pipeline codebase for this specific case.
% All helper functions are appended as local functions at the bottom.

if nargin < 1 || isempty(stage_name)
    stage_name = 'full';
end

if nargin < 2
    subject_list = [];
end

if nargin < 3 || isempty(force_rerun)
    force_rerun = false;
end

% Set environment variables to restrict execution to this specific case
setenv('BENCH_CASE_LIST', '4');
if force_rerun
    setenv('BENCH_FORCE_RERUN', '1');
else
    setenv('BENCH_FORCE_RERUN', '');
end

if isempty(subject_list)
    setenv('BENCH_SUBJECT_LIST', '');
else
    setenv('BENCH_SUBJECT_LIST', char(strjoin(string(subject_list), ',')));
end

fprintf('Starting standalone pipeline for case04_riemannian_rf_common\n');
% Call the bundled run_benchmark_stage
run_benchmark_stage(stage_name);

end

% =========================================================================
% BUNDLED FRAMEWORK FUNCTIONS
% =========================================================================
function run_benchmark_stage(stage_name)
%RUN_BENCHMARK_STAGE Execute one benchmark stage across all 8 configurations.

if nargin < 1 || strlength(string(stage_name)) == 0
    stage_name = "smoke";
end

framework_dir = fileparts(mfilename('fullpath'));
functions_dir = fullfile(framework_dir, 'functions');
try
    addpath(functions_dir);
catch
    % backgroundPool (thread-based) workers do not support addpath.
    % When launched from benchmark_app the path is already set.
end

cfg = default_benchmark_cfg(framework_dir);
cfg.stage_name = char(stage_name);
cfg.force_rerun = is_truthy_env(getenv('BENCH_FORCE_RERUN'));

switch lower(cfg.stage_name)
    case 'smoke'
        cfg.subject_ids = 1:3;
    case 'pilot'
        cfg.subject_ids = 1:10;
    case 'full'
        cfg.subject_ids = 1:109;
    otherwise
        error('Unknown stage_name: %s', cfg.stage_name);
end

subject_list_override = string(getenv('BENCH_SUBJECT_LIST'));
if strlength(strtrim(subject_list_override)) > 0
    tokens = regexp(char(subject_list_override), '[,; ]+', 'split');
    tokens = tokens(~cellfun(@isempty, tokens));
    override_ids = [];
    for i = 1:numel(tokens)
        n = str2double(tokens{i});
        if ~isnan(n)
            override_ids(end + 1) = floor(n); %#ok<AGROW>
        end
    end
    cfg.subject_ids = unique(override_ids(override_ids >= 1 & override_ids <= 109), 'stable');
end

stage_root = fullfile(cfg.results_root, ['stage_' lower(cfg.stage_name)]);
ensure_dir(stage_root);
ensure_dir(fullfile(stage_root, 'case_results'));
ensure_dir(fullfile(stage_root, 'figures'));
ensure_dir(fullfile(stage_root, 'artifacts'));

stage_log = fullfile(stage_root, 'stage_log.txt');
log_message(stage_log, sprintf('Stage %s starting with %d subjects', cfg.stage_name, numel(cfg.subject_ids)));

case_cfgs = build_case_configs(cfg);
case_override = string(getenv('BENCH_CASE_LIST'));
if strlength(strtrim(case_override)) > 0
    tokens = regexp(char(case_override), '[,; ]+', 'split');
    tokens = tokens(~cellfun(@isempty, tokens));
    keep = false(1, numel(case_cfgs));
    for i = 1:numel(tokens)
        tok = string(tokens{i});
        idx = str2double(tok);
        if ~isnan(idx) && idx >= 1 && idx <= numel(case_cfgs)
            keep(idx) = true;
        else
            for j = 1:numel(case_cfgs)
                if strcmpi(case_cfgs(j).case_id, tok)
                    keep(j) = true;
                end
            end
        end
    end
    case_cfgs = case_cfgs(keep);
end

for case_idx = 1:numel(case_cfgs)
    case_cfg = case_cfgs(case_idx);
    case_root = fullfile(stage_root, 'case_results', case_cfg.case_id);
    ensure_dir(case_root);
    ensure_dir(fullfile(case_root, 'curves'));
    ensure_dir(fullfile(case_root, 'mat'));
    case_cfg.case_root = case_root;
    case_cfg.case_csv = fullfile(case_root, [case_cfg.case_id '.csv']);
    case_cfg.case_log = fullfile(case_root, 'case_log.txt');

    log_message(stage_log, sprintf('Starting case %s', case_cfg.case_id));
    log_message(case_cfg.case_log, sprintf('Config: %s | %s | %s', ...
        case_cfg.feature_type, case_cfg.classifier_type, case_cfg.filter_type));

    for subject_num = cfg.subject_ids
        subject_name = sprintf('s%03d', subject_num);
        if ~cfg.force_rerun && subject_already_completed(case_cfg.case_csv, subject_name)
            log_message(case_cfg.case_log, sprintf('Skipping completed subject %s', subject_name));
            continue;
        end

        log_message(stage_log, sprintf('  [%s] Processing %s ...', case_cfg.case_id, subject_name));
        log_message(case_cfg.case_log, sprintf('Processing %s', subject_name));
        try
            result_row = run_subject_case(subject_num, case_cfg);
            log_message(stage_log, sprintf('  [%s] %s  ->  Acc %.3f  F1 %.3f', ...
                case_cfg.case_id, subject_name, result_row.Accuracy, result_row.F1));
        catch ME
            result_row = build_failure_row(subject_num, case_cfg, ME);
            log_message(stage_log, sprintf('  [%s] %s  ->  FAILED: %s', ...
                case_cfg.case_id, subject_name, ME.message));
            log_message(case_cfg.case_log, sprintf('FAILED %s | %s', subject_name, ME.message));
        end

        upsert_case_result(case_cfg.case_csv, result_row);
        try drawnow('limitrate'); catch; end  % keep UI responsive
    end

    summarize_case_outputs(case_cfg);
    log_message(stage_log, sprintf('Finished case %s', case_cfg.case_id));
end

summarize_stage_results(stage_root, build_case_configs(cfg), cfg);
log_message(stage_log, sprintf('Stage %s completed', cfg.stage_name));
end

function tf = is_truthy_env(raw_value)
value = lower(strtrim(char(string(raw_value))));
tf = any(strcmp(value, {'1','true','yes','on'}));
end

function done = subject_already_completed(case_csv, subject_name)
done = false;
if ~exist(case_csv, 'file')
    return;
end
opts = detectImportOptions(case_csv);
opts.VariableNamingRule = 'preserve';
tbl = readtable(case_csv, opts);
if isempty(tbl) || ~ismember('Subject', tbl.Properties.VariableNames)
    return;
end
idx = strcmp(string(tbl.Subject), string(subject_name));
if any(idx) && ismember('Status', tbl.Properties.VariableNames)
    done = all(strcmp(string(tbl.Status(idx)), "OK"));
end
end

function row = build_failure_row(subject_num, case_cfg, ME)
subject_name = sprintf('s%03d', subject_num);
row = table( ...
    string(subject_name), string(case_cfg.case_id), string(case_cfg.feature_type), ...
    string(case_cfg.classifier_type), string(case_cfg.filter_type), ...
    -1, -1, -1, -1, 0, inf, 0, 0, 0, 0, string("FAILED"), string(regexprep(ME.message, '[\r\n]+', ' ')), ...
    'VariableNames', {'Subject','CaseId','FeatureType','Classifier','FilterType', ...
    'Accuracy','Precision','Recall','F1','SelectedFeatureCount','HOBestScore', ...
    'Conf11','Conf12','Conf21','Conf22','Status','Notes'});
end

function summarize_case_outputs(case_cfg)
try
    opts = detectImportOptions(case_cfg.case_csv);
    opts.VariableNamingRule = 'preserve';
    tbl = readtable(case_cfg.case_csv, opts);
    tbl = sortrows(tbl, 'Subject');
    valid = strcmp(string(tbl.Status), "OK");
    tbl_valid = tbl(valid, :);
    if isempty(tbl_valid)
        return;
    end

    agg_conf = [sum(tbl_valid.Conf11), sum(tbl_valid.Conf12); sum(tbl_valid.Conf21), sum(tbl_valid.Conf22)];
    fig_dir = fullfile(fileparts(fileparts(case_cfg.case_root)), 'figures');

    f1 = figure('Visible', 'off');
    imagesc(agg_conf);
    colorbar;
    axis equal tight;
    title(sprintf('Aggregated Confusion Matrix - %s', case_cfg.case_id), 'Interpreter', 'none');
    xlabel('Predicted');
    ylabel('True');
    set(gca, 'XTick', 1:2, 'YTick', 1:2);
    saveas(f1, fullfile(fig_dir, [case_cfg.case_id '_confusion.png']));
    close(f1);

    curve_files = dir(fullfile(case_cfg.case_root, 'curves', '*.mat'));
    if ~isempty(curve_files)
        curves = {};
        max_len = 0;
        for i = 1:numel(curve_files)
            S = load(fullfile(curve_files(i).folder, curve_files(i).name), 'ho_curve');
            curves{end + 1} = S.ho_curve(:); %#ok<AGROW>
            max_len = max(max_len, numel(S.ho_curve));
        end
        curve_mat = nan(numel(curves), max_len);
        for i = 1:numel(curves)
            curve_mat(i, 1:numel(curves{i})) = curves{i};
        end
        mean_curve = mean(curve_mat, 1, 'omitnan');
        f2 = figure('Visible', 'off');
        plot(mean_curve, 'LineWidth', 2);
        xlabel('Iteration');
        ylabel('Mean Fitness/Error');
        title(sprintf('Mean HO Convergence - %s', case_cfg.case_id), 'Interpreter', 'none');
        saveas(f2, fullfile(fig_dir, [case_cfg.case_id '_convergence.png']));
        close(f2);
    end
catch
    % figure/plot calls are not supported on thread-based backgroundPool
    % workers. Skip plotting — CSV results are the critical output.
    % Figures can be regenerated from the saved data after the run.
end
end


function selected_idx = anova_rank_features(X, Y, top_k)
D = size(X, 2);
F = zeros(1, D);
idx1 = (Y == 1);
idx2 = (Y == 2);
N1 = sum(idx1);
N2 = sum(idx2);
for d = 1:D
    col = X(:, d);
    mu = mean(col);
    mu1 = mean(col(idx1));
    mu2 = mean(col(idx2));
    var1 = var(col(idx1));
    var2 = var(col(idx2));
    between = N1 * (mu1 - mu)^2 + N2 * (mu2 - mu)^2;
    within = (N1 - 1) * var1 + (N2 - 1) * var2;
    if within > 0
        F(d) = between / within;
    else
        F(d) = 0;
    end
end
[~, order] = sort(F, 'descend');
selected_idx = order(1:min(top_k, D));
end


function [X_left_f, X_right_f] = apply_bandpass_filter(X_left, X_right, fs, filter_type, cfg)
X_left_f = X_left;
X_right_f = X_right;

switch lower(filter_type)
    case 'common'
        bpFilt = designfilt('bandpassiir', 'FilterOrder', cfg.common_filter_order, ...
            'HalfPowerFrequency1', cfg.band(1), 'HalfPowerFrequency2', cfg.band(2), ...
            'SampleRate', fs);
        for i = 1:size(X_left, 3)
            for ch = 1:size(X_left, 1)
                X_left_f(ch, :, i) = filtfilt(bpFilt, double(X_left(ch, :, i)));
                X_right_f(ch, :, i) = filtfilt(bpFilt, double(X_right(ch, :, i)));
            end
        end
    case 'butterworth'
        [b, a] = butter(cfg.butterworth_order, cfg.band / (fs / 2), 'bandpass');
        for i = 1:size(X_left, 3)
            for ch = 1:size(X_left, 1)
                X_left_f(ch, :, i) = filtfilt(b, a, double(X_left(ch, :, i)));
                X_right_f(ch, :, i) = filtfilt(b, a, double(X_right(ch, :, i)));
            end
        end
    otherwise
        error('Unknown filter_type: %s', filter_type);
end
end


function case_cfgs = build_case_configs(cfg)
case_matrix = {
    'csp',        'svm', 'common'
    'riemannian', 'svm', 'common'
    'csp',        'rf',  'common'
    'riemannian', 'rf',  'common'
    'csp',        'svm', 'butterworth'
    'riemannian', 'svm', 'butterworth'
    'csp',        'rf',  'butterworth'
    'riemannian', 'rf',  'butterworth'
};

case_cfgs = repmat(cfg, 1, size(case_matrix, 1));
for idx = 1:size(case_matrix, 1)
    feature_type = case_matrix{idx, 1};
    classifier_type = case_matrix{idx, 2};
    filter_type = case_matrix{idx, 3};

    case_cfgs(idx).feature_type = feature_type;
    case_cfgs(idx).classifier_type = classifier_type;
    case_cfgs(idx).filter_type = filter_type;
    case_cfgs(idx).case_index = idx;
    case_cfgs(idx).case_id = sprintf('case%02d_%s_%s_%s', idx, feature_type, classifier_type, filter_type);
end
end


function [metrics, conf_mat] = compute_binary_metrics(y_true, y_pred, positive_label)
labels = [1 2];
conf_mat = confusionmat(y_true, y_pred, 'Order', labels);

pos_idx = find(labels == positive_label, 1);
tp = conf_mat(pos_idx, pos_idx);
fp = sum(conf_mat(:, pos_idx)) - tp;
fn = sum(conf_mat(pos_idx, :)) - tp;

accuracy = sum(diag(conf_mat)) / max(1, sum(conf_mat(:)));
precision = tp / max(1, tp + fp);
recall = tp / max(1, tp + fn);
f1 = (2 * precision * recall) / max(eps, precision + recall);

metrics = struct();
metrics.accuracy = accuracy;
metrics.precision = precision;
metrics.recall = recall;
metrics.f1 = f1;
end


function [W, eigvals] = compute_csp_filters(class1, class2)
n1 = size(class1, 3);
n2 = size(class2, 3);
cov1 = zeros(size(class1, 1));
cov2 = zeros(size(class2, 1));
for i = 1:n1
    trial = squeeze(class1(:, :, i));
    cov1 = cov1 + (trial * trial') / trace(trial * trial');
end
for i = 1:n2
    trial = squeeze(class2(:, :, i));
    cov2 = cov2 + (trial * trial') / trace(trial * trial');
end
cov1 = cov1 / n1;
cov2 = cov2 / n2;
[W_raw, D] = eig(cov1, cov1 + cov2);
[eigvals, idx] = sort(diag(D), 'descend');
W = W_raw(:, idx)';
end


function split = create_subject_split(labels, subject_num, cfg)
rng(cfg.seed + subject_num, 'twister');
cv = cvpartition(labels, 'HoldOut', cfg.eval.holdout);
split = struct();
split.train_idx = training(cv);
split.test_idx = test(cv);
end


function cfg = default_benchmark_cfg(framework_dir)
workspace_dir = fileparts(framework_dir);
parent_dir = fileparts(workspace_dir);
cfg = struct();
cfg.framework_dir = framework_dir;
cfg.workspace_dir = workspace_dir;
shared_cache_dir = fullfile(parent_dir, 'physionet_cache');
if exist(shared_cache_dir, 'dir')
    cfg.cache_dir = shared_cache_dir;
else
    cfg.cache_dir = fullfile(workspace_dir, 'physionet_cache');
end
cfg.results_root = fullfile(workspace_dir, 'benchmark_results');
cfg.seed = 42;
cfg.band = [8 30];
cfg.trial_duration_sec = 4;
cfg.run_ids = [4 8 12];
cfg.min_balanced_trials = 10;
cfg.common_filter_order = 4;
cfg.butterworth_order = 4;
cfg.ho = struct('agents', 10, 'iterations', 30, 'threshold', 0.5);
cfg.eval = struct('holdout', 0.2, 'ho_folds', 5, 'positive_label', 2);
cfg.classifier = struct('svm_kernel', 'linear', 'rf_cycles', 100, 'rf_min_leaf', 1);
cfg.riemannian = struct('epsilon', 1e-6, 'top_k', 0);
cfg.runtime = struct('enable_intermediate_cache', true);
end


function download_with_retry(url, local_path, max_attempts, timeout_sec)
last_error = [];
for attempt = 1:max_attempts
    try
        opts = weboptions('Timeout', timeout_sec);
        websave(local_path, url, opts);
        return;
    catch ME
        last_error = ME;
        if attempt < max_attempts
            pause(2 * attempt);
        end
    end
end
rethrow(last_error);
end


function ensure_dir(path_str)
if ~exist(path_str, 'dir')
    mkdir(path_str);
end
end


function error_val = evaluate_mask_fitness(position, X, Y, classifier_type, cfg, subject_num, cv)
if islogical(position)
    mask = position;
else
    mask = position > cfg.ho.threshold;
end
if ~any(mask)
    error_val = 1;
    return;
end
X_sel = X(:, mask);

if nargin < 7 || isempty(cv)
    rng(cfg.seed + 1000 + subject_num, 'twister');
    cv = cvpartition(Y, 'KFold', cfg.eval.ho_folds);
end

accs = zeros(1, cv.NumTestSets);
rf_tree = [];
if strcmpi(classifier_type, 'rf')
    rf_tree = templateTree('MinLeafSize', cfg.classifier.rf_min_leaf);
end

for k = 1:cv.NumTestSets
    X_train = X_sel(training(cv, k), :);
    Y_train = Y(training(cv, k));
    X_test = X_sel(test(cv, k), :);
    Y_test = Y(test(cv, k));
    switch lower(classifier_type)
        case 'svm'
            model = fitcsvm(X_train, Y_train, 'KernelFunction', cfg.classifier.svm_kernel);
        case 'rf'
            model = fitcensemble(X_train, categorical(Y_train), 'Method', 'Bag', ...
                'Learners', rf_tree, 'NumLearningCycles', cfg.classifier.rf_cycles);
        otherwise
            error('Unknown classifier_type: %s', classifier_type);
    end
    pred = predict(model, X_test);
    if iscategorical(pred)
        pred = double(pred);
    end
    accs(k) = mean(pred == Y_test);
end
error_val = 1 - mean(accs);
end


function feature_bundle = extract_csp_features(X_left, X_right, split, cfg)
X_all = cat(3, X_left, X_right);
labels = [ones(size(X_left, 3), 1); 2 * ones(size(X_right, 3), 1)];

train_left = X_all(:, :, find(split.train_idx & labels == 1));
train_right = X_all(:, :, find(split.train_idx & labels == 2));
[filters, ~] = compute_csp_filters(train_left, train_right);
n_filters = size(filters, 1);
selected_filter_idx = unique(max(1, [1, 2, n_filters - 1, n_filters]));
W = filters(selected_filter_idx, :);

features = zeros(size(X_all, 3), numel(selected_filter_idx));
for i = 1:size(X_all, 3)
    projected = W * squeeze(X_all(:, :, i));
    var_features = var(projected, 0, 2);
    features(i, :) = log(var_features ./ sum(var_features));
end

feature_bundle = struct();
feature_bundle.train_X = features(split.train_idx, :);
feature_bundle.test_X = features(split.test_idx, :);
feature_bundle.train_Y = labels(split.train_idx);
feature_bundle.test_Y = labels(split.test_idx);
feature_bundle.selected_filter_idx = selected_filter_idx;
end


function feature_bundle = extract_riemannian_features(X_left, X_right, split, cfg)
n_trials = size(X_left, 3);
n_channels = size(X_left, 1);
epsilon = cfg.riemannian.epsilon;
diag_mask = triu(true(n_channels), 0);
offdiag_mask = triu(true(n_channels), 1);

X_all = cat(3, X_left, X_right);
labels = [ones(n_trials, 1); 2 * ones(n_trials, 1)];
num_total = size(X_all, 3);

cov_matrices = zeros(n_channels, n_channels, num_total);
for i = 1:num_total
    trial = squeeze(X_all(:, :, i));
    cov_matrices(:, :, i) = (trial * trial') / trace(trial * trial') + epsilon * eye(n_channels);
end

train_cov = cov_matrices(:, :, split.train_idx);
mean_cov = riemannian_mean_spd(train_cov);
P_half = sqrtm(mean_cov);
P_half_inv = P_half \ eye(size(P_half));

n_features = sum(diag_mask(:)) + sum(offdiag_mask(:));
features = zeros(num_total, n_features);
for i = 1:num_total
    tangent_matrix = logm(P_half_inv * cov_matrices(:, :, i) * P_half_inv);
    diag_terms = tangent_matrix(diag_mask);
    offdiag_terms = sqrt(2) * tangent_matrix(offdiag_mask);
    features(i, :) = [diag_terms; offdiag_terms].';
end

feature_bundle = struct();
feature_bundle.train_X = features(split.train_idx, :);
feature_bundle.test_X = features(split.test_idx, :);
feature_bundle.train_Y = labels(split.train_idx);
feature_bundle.test_Y = labels(split.test_idx);
end


function [best_score, best_pos, ho_curve] = hippopotamus_optimization(dim, fitness_fn, cfg)
SearchAgents = cfg.ho.agents;
Max_iterations = cfg.ho.iterations;
lb = zeros(1, dim);
ub = ones(1, dim);

X = lb + rand(SearchAgents, dim) .* (ub - lb);
fit = zeros(1, SearchAgents);
for i = 1:SearchAgents
    fit(i) = fitness_fn(X(i, :));
end

fbest = inf;
Xbest = X(1, :);
ho_curve = zeros(1, Max_iterations);
for t = 1:Max_iterations
    [best, loc] = min(fit);
    if t == 1 || best < fbest
        fbest = best;
        Xbest = X(loc, :);
    end
    for i = 1:SearchAgents
        r = rand(1, dim);
        X_new = X(i, :) + r .* (Xbest - X(i, :));
        X_new = min(max(X_new, lb), ub);
        f_new = fitness_fn(X_new);
        if f_new < fit(i)
            X(i, :) = X_new;
            fit(i) = f_new;
        end
    end
    ho_curve(t) = fbest;
end

best_score = fbest;
best_pos = Xbest;
end


function subject_data = load_physionet_subject(subject_num, cfg)
subject_code = sprintf('S%03d', subject_num);
subject_dir = fullfile(cfg.cache_dir, subject_code);
ensure_dir(subject_dir);

fs = 160;
samples_per_trial = cfg.trial_duration_sec * fs;
left_trials = {};
right_trials = {};

for run_id = cfg.run_ids
    file_name = sprintf('%sR%02d.edf', subject_code, run_id);
    local_path = fullfile(subject_dir, file_name);
    url = sprintf('https://physionet.org/files/eegmmidb/1.0.0/%s/%s', subject_code, file_name);

    if ~exist(local_path, 'file')
        download_with_retry(url, local_path, 5, 60);
    end

    try
        info = edfinfo(local_path);
        tt = edfread(local_path);
    catch
        if exist(local_path, 'file')
            delete(local_path);
        end
        download_with_retry(url, local_path, 5, 60);
        info = edfinfo(local_path);
        tt = edfread(local_path);
    end

    raw = timetable_to_matrix(tt);
    anns = info.Annotations;

    for i = 1:height(anns)
        label = string(anns.Annotations(i));
        if label ~= "T1" && label ~= "T2"
            continue;
        end
        onset_sec = seconds(anns.Properties.RowTimes(i));
        start_idx = floor(onset_sec * fs) + 1;
        stop_idx = start_idx + samples_per_trial - 1;
        if stop_idx > size(raw, 2)
            continue;
        end
        trial = raw(:, start_idx:stop_idx);
        if label == "T1"
            left_trials{end + 1} = trial; %#ok<AGROW>
        else
            right_trials{end + 1} = trial; %#ok<AGROW>
        end
    end
end

n_trials = min(numel(left_trials), numel(right_trials));
if n_trials < cfg.min_balanced_trials
    error('Insufficient balanced trials for S%03d: left=%d right=%d', subject_num, numel(left_trials), numel(right_trials));
end

left_trials = left_trials(1:n_trials);
right_trials = right_trials(1:n_trials);

X_left = zeros(size(left_trials{1}, 1), size(left_trials{1}, 2), n_trials);
X_right = zeros(size(right_trials{1}, 1), size(right_trials{1}, 2), n_trials);
for i = 1:n_trials
    X_left(:, :, i) = left_trials{i};
    X_right(:, :, i) = right_trials{i};
end

subject_data = struct();
subject_data.subject_num = subject_num;
subject_data.subject_name = sprintf('s%03d', subject_num);
subject_data.fs = fs;
subject_data.X_left = X_left;
subject_data.X_right = X_right;
subject_data.n_trials = n_trials;
end


function log_message(log_file, message)
fid = fopen(log_file, 'a');
if fid == -1
    return;
end
fprintf(fid, '[%s] %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'), message);
fclose(fid);
end


function mean_cov = riemannian_mean_spd(cov_matrices)
num_matrices = size(cov_matrices, 3);
mean_cov = mean(cov_matrices, 3);
for iter = 1:5
    P_half = sqrtm(mean_cov);
    P_half_inv = inv(P_half);
    sum_log = zeros(size(mean_cov));
    for i = 1:num_matrices
        tmp = P_half_inv * cov_matrices(:, :, i) * P_half_inv;
        sum_log = sum_log + logm(tmp);
    end
    mean_cov = P_half * expm(sum_log / num_matrices) * P_half;
    mean_cov = (mean_cov + mean_cov') / 2;
end
end


function ho_result = run_ho_feature_selection(train_X, train_Y, case_cfg, subject_num)
candidate_idx = 1:size(train_X, 2);
X_ho = train_X;

if strcmpi(case_cfg.feature_type, 'riemannian') && case_cfg.riemannian.top_k > 0
    candidate_idx = anova_rank_features(train_X, train_Y, case_cfg.riemannian.top_k);
    X_ho = train_X(:, candidate_idx);
end

rng(case_cfg.seed + 1000 + subject_num, 'twister');
cv = cvpartition(train_Y, 'KFold', case_cfg.eval.ho_folds);
fitness_cache = containers.Map('KeyType', 'char', 'ValueType', 'double');
fitness_fn = @cached_mask_fitness;
[best_score, best_pos, ho_curve] = hippopotamus_optimization(size(X_ho, 2), fitness_fn, case_cfg);
mask = best_pos > case_cfg.ho.threshold;
if ~any(mask)
    mask = true(1, size(X_ho, 2));
end

selected_idx = candidate_idx(mask);
ho_result = struct();
ho_result.selected_idx = selected_idx;
ho_result.selected_count = numel(selected_idx);
ho_result.best_score = best_score;
ho_result.best_pos = best_pos;
ho_result.ho_curve = ho_curve;

    function error_val = cached_mask_fitness(position)
        mask_local = position > case_cfg.ho.threshold;
        if ~any(mask_local)
            error_val = 1;
            return;
        end

        mask_key = char(mask_local + '0');
        if isKey(fitness_cache, mask_key)
            error_val = fitness_cache(mask_key);
            return;
        end

        error_val = evaluate_mask_fitness(mask_local, X_ho, train_Y, ...
            case_cfg.classifier_type, case_cfg, subject_num, cv);
        fitness_cache(mask_key) = error_val;
    end
end


function result_row = run_subject_case(subject_num, case_cfg)
subject_name = sprintf('s%03d', subject_num);
[X_left_f, X_right_f] = load_or_build_filtered_trials(subject_num, case_cfg);

labels = [ones(size(X_left_f, 3), 1); 2 * ones(size(X_right_f, 3), 1)];
split = create_subject_split(labels, subject_num, case_cfg);
feature_bundle = load_or_build_feature_bundle(subject_num, subject_name, X_left_f, X_right_f, split, case_cfg);

[train_X_norm, test_X_norm] = zscore_train_test(feature_bundle.train_X, feature_bundle.test_X);
ho_result = run_ho_feature_selection(train_X_norm, feature_bundle.train_Y, case_cfg, subject_num);

train_sel = train_X_norm(:, ho_result.selected_idx);
test_sel = test_X_norm(:, ho_result.selected_idx);
eval_result = train_and_evaluate_classifier(train_sel, feature_bundle.train_Y, test_sel, feature_bundle.test_Y, case_cfg);

curve_dir = fullfile(case_cfg.case_root, 'curves');
mat_dir = fullfile(case_cfg.case_root, 'mat');
ho_curve = ho_result.ho_curve; %#ok<NASGU>
save(fullfile(curve_dir, [subject_name '_curve.mat']), 'ho_curve');
save(fullfile(mat_dir, [subject_name '_result.mat']), 'feature_bundle', 'ho_result', 'eval_result');

conf = eval_result.conf_mat;
result_row = table( ...
    string(subject_name), string(case_cfg.case_id), string(case_cfg.feature_type), ...
    string(case_cfg.classifier_type), string(case_cfg.filter_type), ...
    eval_result.accuracy, eval_result.precision, eval_result.recall, eval_result.f1, ...
    ho_result.selected_count, ho_result.best_score, ...
    conf(1,1), conf(1,2), conf(2,1), conf(2,2), string("OK"), string(""), ...
    'VariableNames', {'Subject','CaseId','FeatureType','Classifier','FilterType', ...
    'Accuracy','Precision','Recall','F1','SelectedFeatureCount','HOBestScore', ...
    'Conf11','Conf12','Conf21','Conf22','Status','Notes'});
end

function [X_left_f, X_right_f] = load_or_build_filtered_trials(subject_num, case_cfg)
cache_root = fullfile(case_cfg.workspace_dir, 'benchmark_cache', 'filtered');
cache_file = fullfile(cache_root, sprintf('s%03d_%s.mat', subject_num, case_cfg.filter_type));

if case_cfg.runtime.enable_intermediate_cache && exist(cache_file, 'file')
    S = load(cache_file, 'X_left_f', 'X_right_f');
    X_left_f = S.X_left_f;
    X_right_f = S.X_right_f;
    return;
end

subject_data = load_physionet_subject(subject_num, case_cfg);
[X_left_f, X_right_f] = apply_bandpass_filter(subject_data.X_left, subject_data.X_right, ...
    subject_data.fs, case_cfg.filter_type, case_cfg);

if case_cfg.runtime.enable_intermediate_cache
    ensure_dir(cache_root);
    save(cache_file, 'X_left_f', 'X_right_f');
end
end

function feature_bundle = load_or_build_feature_bundle(subject_num, subject_name, X_left_f, X_right_f, split, case_cfg)
cache_root = fullfile(case_cfg.workspace_dir, 'benchmark_cache', 'features');
cache_file = fullfile(cache_root, sprintf('%s_%s_%s.mat', subject_name, case_cfg.filter_type, case_cfg.feature_type));

if case_cfg.runtime.enable_intermediate_cache && exist(cache_file, 'file')
    S = load(cache_file, 'feature_bundle');
    feature_bundle = S.feature_bundle;
    return;
end

switch lower(case_cfg.feature_type)
    case 'csp'
        feature_bundle = extract_csp_features(X_left_f, X_right_f, split, case_cfg);
    case 'riemannian'
        feature_bundle = extract_riemannian_features(X_left_f, X_right_f, split, case_cfg);
    otherwise
        error('Unknown feature_type: %s', case_cfg.feature_type);
end

if case_cfg.runtime.enable_intermediate_cache
    ensure_dir(cache_root);
    save(cache_file, 'feature_bundle');
end
end


function summarize_stage_results(stage_root, case_cfgs, cfg)
summary_rows = table();
for i = 1:numel(case_cfgs)
    case_csv = fullfile(stage_root, 'case_results', case_cfgs(i).case_id, [case_cfgs(i).case_id '.csv']);
    if ~exist(case_csv, 'file')
        continue;
    end
    opts = detectImportOptions(case_csv);
    opts.VariableNamingRule = 'preserve';
    tbl = readtable(case_csv, opts);
    valid = strcmp(string(tbl.Status), "OK");
    tbl = tbl(valid, :);
    if isempty(tbl)
        continue;
    end
    [best_accuracy, best_idx] = max(tbl.Accuracy);
    [worst_accuracy, worst_idx] = min(tbl.Accuracy);
    subject_count = height(tbl);
    subject_coverage = 100 * subject_count / numel(cfg.subject_ids);
    summary_rows = [summary_rows; table( ... %#ok<AGROW>
        string(case_cfgs(i).case_id), string(case_cfgs(i).feature_type), ...
        string(case_cfgs(i).classifier_type), string(case_cfgs(i).filter_type), ...
        subject_count, subject_coverage, ...
        mean(tbl.Accuracy), std(tbl.Accuracy), median(tbl.Accuracy), ...
        best_accuracy, worst_accuracy, ...
        sum(tbl.Accuracy >= 0.7), sum(tbl.Accuracy >= 0.8), sum(tbl.Accuracy >= 0.9), ...
        mean(tbl.Precision), mean(tbl.Recall), mean(tbl.F1), ...
        mean(tbl.SelectedFeatureCount), mean(tbl.HOBestScore), ...
        string(tbl.Subject(best_idx)), string(tbl.Subject(worst_idx)), ...
        'VariableNames', {'CaseId','FeatureType','Classifier','FilterType','SubjectCount', ...
        'SubjectCoveragePct','MeanAccuracy','StdAccuracy','MedianAccuracy','BestAccuracy','WorstAccuracy', ...
        'SubjectsAbove70','SubjectsAbove80','SubjectsAbove90','MeanPrecision','MeanRecall','MeanF1', ...
        'MeanSelectedFeatureCount','MeanHOBestScore','BestSubject','WorstSubject'})];
end

summary_csv = fullfile(stage_root, 'artifacts', 'stage_summary.csv');
writetable(summary_rows, summary_csv);

if isempty(summary_rows)
    return;
end

summary_rows = sortrows(summary_rows, 'MeanAccuracy', 'descend');
try
    fig = figure('Visible', 'off');
    bar(categorical(summary_rows.CaseId), summary_rows.MeanAccuracy);
    xtickangle(45);
    ylabel('Mean Accuracy');
    title(sprintf('Stage %s Mean Accuracy by Case', cfg.stage_name), 'Interpreter', 'none');
    saveas(fig, fullfile(stage_root, 'figures', 'stage_mean_accuracy.png'));
    close(fig);
catch
    % figure/plot not supported on thread-based backgroundPool workers.
    % The summary CSV is already saved; figures can be regenerated later.
end
end


function data = timetable_to_matrix(tt)
n_channels = width(tt);
n_records = height(tt);
samples_per_record = numel(tt{1, 1}{1});
data = zeros(n_channels, n_records * samples_per_record);

for ch = 1:n_channels
    signal = zeros(n_records * samples_per_record, 1);
    cursor = 1;
    for rec = 1:n_records
        chunk = tt{rec, ch}{1};
        next_cursor = cursor + numel(chunk) - 1;
        signal(cursor:next_cursor) = chunk;
        cursor = next_cursor + 1;
    end
    data(ch, :) = signal.';
end
end


function eval_result = train_and_evaluate_classifier(train_X, train_Y, test_X, test_Y, case_cfg)
switch lower(case_cfg.classifier_type)
    case 'svm'
        model = fitcsvm(train_X, train_Y, 'KernelFunction', case_cfg.classifier.svm_kernel);
    case 'rf'
        tree = templateTree('MinLeafSize', case_cfg.classifier.rf_min_leaf);
        model = fitcensemble(train_X, categorical(train_Y), 'Method', 'Bag', ...
            'Learners', tree, 'NumLearningCycles', case_cfg.classifier.rf_cycles);
    otherwise
        error('Unknown classifier_type: %s', case_cfg.classifier_type);
end

pred = predict(model, test_X);
if iscategorical(pred)
    pred = double(pred);
end

[metrics, conf_mat] = compute_binary_metrics(test_Y, pred, case_cfg.eval.positive_label);
eval_result = metrics;
eval_result.conf_mat = conf_mat;
end


function upsert_case_result(csv_path, row_tbl)
if exist(csv_path, 'file')
    opts = detectImportOptions(csv_path);
    opts.VariableNamingRule = 'preserve';
    tbl = readtable(csv_path, opts);
    idx = strcmp(string(tbl.Subject), string(row_tbl.Subject));
    tbl(idx, :) = [];
    tbl = [tbl; row_tbl];
else
    tbl = row_tbl;
end
tbl = sortrows(tbl, 'Subject');
writetable(tbl, csv_path);
end


function [train_X_norm, test_X_norm, stats] = zscore_train_test(train_X, test_X)
mu = mean(train_X, 1);
sigma = std(train_X, 0, 1);
sigma(sigma == 0) = 1;
train_X_norm = (train_X - mu) ./ sigma;
test_X_norm = (test_X - mu) ./ sigma;
stats = struct('mu', mu, 'sigma', sigma);
end
