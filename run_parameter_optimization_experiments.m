function results = run_parameter_optimization_experiments(subject_mode, dataset_filter)
%RUN_PARAMETER_OPTIMIZATION_EXPERIMENTS Grid runner for MI timing/CSP studies.
% This experiment layer keeps the existing preprocessing, PCA, HO feature
% selection, and linear SVM classifier paths. It changes only the scientific
% variable under test: fixed channel mode on the optimized CSP baseline.
%
% Examples:
%   run_parameter_optimization_experiments("smoke")
%   run_parameter_optimization_experiments("pilot", "physionet")
%   run_parameter_optimization_experiments("full", "all")

if nargin < 1 || strlength(string(subject_mode)) == 0
    subject_mode = "smoke";
end
if nargin < 2 || strlength(string(dataset_filter)) == 0
    dataset_filter = "all";
end

subject_mode = lower(string(subject_mode));
dataset_filter = lower(string(dataset_filter));

project_root = fileparts(mfilename('fullpath'));
results_root = fullfile(project_root, 'optimization_experiment_results');
details_root = fullfile(results_root, 'details');
ensure_local_dir(results_root);
ensure_local_dir(details_root);
notes_root = fullfile(results_root, 'research_tracking');
ensure_local_dir(notes_root);

run_stamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
summary_csv = fullfile(results_root, ['parameter_optimization_summary_' run_stamp '.csv']);
summary_mat = fullfile(results_root, ['parameter_optimization_summary_' run_stamp '.mat']);
log_file = fullfile(results_root, ['parameter_optimization_log_' run_stamp '.txt']);
notes_txt = fullfile(notes_root, 'research_notes.txt');
research_summary_csv = fullfile(notes_root, ['research_summary_' run_stamp '.csv']);
research_summary_mat = fullfile(notes_root, ['research_summary_' run_stamp '.mat']);
research_comparison_csv = fullfile(notes_root, ['research_comparisons_' run_stamp '.csv']);
research_comparison_mat = fullfile(notes_root, ['research_comparisons_' run_stamp '.mat']);
research_ranking_csv = fullfile(notes_root, ['research_rankings_' run_stamp '.csv']);
research_ranking_mat = fullfile(notes_root, ['research_rankings_' run_stamp '.mat']);
research_summary_txt = fullfile(notes_root, ['research_summary_' run_stamp '.txt']);

pipelines = [
    "spatial_only"
    "spatial_spectral"
    "spatial_temporal"
    "spatial_spectral_temporal"
    "spatial_spectral_temporal_riemannian"
    "fbcsp_mi"
];
channel_modes = ["motor_core","sensorimotor","all"];
bandpass_ranges = [8 30];

datasets = strings(0, 1);
if dataset_filter == "all" || dataset_filter == "cho"
    datasets(end + 1) = "cho";
end
if dataset_filter == "all" || dataset_filter == "physionet" || dataset_filter == "eegmmidb"
    datasets(end + 1) = "physionet";
end
if isempty(datasets)
    error('Unknown dataset_filter: %s', dataset_filter);
end

log_line(log_file, 'Parameter optimization run started.');
log_line(log_file, sprintf('Subject mode: %s | Dataset filter: %s', subject_mode, dataset_filter));
log_line(log_file, 'Evaluation protocol: train/test splits are rebuilt inside each subject; fold-local feature processing is fit on training folds only, and FBCSP-MI skips HO.');

summary_rows = table();
experiment_index = 0;
framework_start = tic;

for dataset = datasets(:)'
    dataset_str = string(dataset);
    if dataset == "cho"
        mi_windows = [3.5 6.5];
        csp_counts = [4];
    else
        mi_windows = [1 4];
        csp_counts = [6];
    end

    for pipeline_idx = 1:numel(pipelines)
        pipeline_type = pipelines(pipeline_idx);
        pipeline_type_str = string(pipeline_type);
        if pipeline_type_str == "fbcsp_mi"
            pipeline_bandpass_ranges = [4 30];
            fbcsp_mi_top_k = 20;
        else
            pipeline_bandpass_ranges = bandpass_ranges;
            fbcsp_mi_top_k = NaN;
        end
        for window_idx = 1:size(mi_windows, 1)
            mi_window = mi_windows(window_idx, :);
            for band_idx = 1:size(pipeline_bandpass_ranges, 1)
                bandpass_range = pipeline_bandpass_ranges(band_idx, :);
                for csp_components = csp_counts
                    csp_components_val = double(csp_components);
                    for channel_idx = 1:numel(channel_modes)
                        channel_mode = channel_modes(channel_idx);
                        channel_mode_str = string(channel_mode);
                        experiment_index = experiment_index + 1;
                        experiment_id = build_experiment_id( ...
                            dataset_str, pipeline_type_str, mi_window, bandpass_range, csp_components_val, channel_mode_str, fbcsp_mi_top_k);

                        log_line(log_file, sprintf( ...
                            'Experiment %03d starting: %s', experiment_index, experiment_id));
                        fprintf('\n[%03d] %s\n', experiment_index, experiment_id);
                        experiment_tic = tic;

                        try
                            if dataset == "cho"
                                detail = run_cho_experiment(project_root, details_root, experiment_id, ...
                                    subject_mode, pipeline_type, mi_window, bandpass_range, csp_components, channel_mode, log_file);
                            else
                                detail = run_physionet_experiment(project_root, details_root, experiment_id, ...
                                    subject_mode, pipeline_type, mi_window, bandpass_range, csp_components, channel_mode, log_file);
                            end

                            valid = detail.rows(detail.rows.status == "OK", :);
                            if isempty(valid)
                                accuracy = NaN;
                                f1_score = NaN;
                                selected_features = NaN;
                                status = "NO_VALID_SUBJECTS";
                            else
                                accuracy = mean(valid.accuracy);
                                f1_score = mean(valid.f1_score);
                                selected_features = mean(valid.selected_features);
                                status = "OK";
                            end
                            notes = "";
                        catch ME
                            accuracy = NaN;
                            f1_score = NaN;
                            selected_features = NaN;
                            status = "FAILED";
                            notes = string(regexprep(ME.message, '[\r\n]+', ' '));
                            detail = struct('csv_file', "", 'mat_file', "", 'completed_subjects', 0, ...
                                'mean_subject_runtime_sec', NaN, 'experiment_runtime_sec', NaN, 'rows', table());
                            log_line(log_file, sprintf('Experiment %s failed: %s', experiment_id, ME.message));
                        end
                        experiment_runtime_sec = toc(experiment_tic);

                        row_timestamp = string(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
                        mi_window_str = window_to_string(mi_window);
                        window_duration_sec = double(mi_window(2) - mi_window(1));
                        bandpass_range_str = band_to_string(bandpass_range);
                        accuracy = double(accuracy);
                        f1_score = double(f1_score);
                        selected_features = double(selected_features);
                        subject_mode_str = string(subject_mode);
                        experiment_runtime_sec = double(experiment_runtime_sec);
                        runtime_sec = double(experiment_runtime_sec);
                        mean_subject_runtime_sec = double(detail.mean_subject_runtime_sec);
                        completed_subjects = double(detail.completed_subjects);
                        status_str = string(status);
                        notes_str = string(notes);
                        detail_csv_str = string(detail.csv_file);
                        detail_mat_str = string(detail.mat_file);
                        assert(isscalar(accuracy));
                        assert(isscalar(f1_score));
                        assert(isscalar(selected_features));
                        assert(isscalar(window_duration_sec));
                        assert(isscalar(csp_components_val));
                        assert(isscalar(experiment_runtime_sec));
                        assert(isscalar(runtime_sec));
                        assert(isscalar(mean_subject_runtime_sec));
                        assert(isscalar(completed_subjects));
                        row = table( ...
                            dataset_str, pipeline_type_str, channel_mode_str, mi_window_str, ...
                            window_duration_sec, bandpass_range_str, csp_components_val, accuracy, f1_score, ...
                            selected_features, subject_mode_str, row_timestamp, ...
                            experiment_runtime_sec, runtime_sec, mean_subject_runtime_sec, completed_subjects, status_str, notes_str, ...
                            detail_csv_str, detail_mat_str, ...
                            'VariableNames', {'dataset','pipeline_type','channel_mode','mi_window','window_duration_sec','bandpass_range', ...
                            'csp_components','accuracy','f1_score','selected_features', ...
                            'subject_mode','timestamp','experiment_runtime_sec','runtime_sec','mean_subject_runtime_sec','completed_subjects','status','notes', ...
                            'detail_csv','detail_mat'});
                        summary_rows = [summary_rows; row]; %#ok<AGROW>
                        writetable(summary_rows, summary_csv);
                        save(summary_mat, 'summary_rows');
                        try
                            current_note = build_experiment_note(row, detail, summary_rows);
                            append_text_block(notes_txt, current_note);
                        catch noteME
                            warning('Research note generation failed for %s: %s', experiment_id, noteME.message);
                            log_line(log_file, sprintf('Warning: note generation failed for %s: %s', experiment_id, noteME.message));
                        end
                        log_line(log_file, sprintf('Experiment %s finished with status %s.', experiment_id, status));
                    end
                end
            end
        end
    end
end

final_summary_rows = sortrows(summary_rows, {'accuracy','f1_score'}, {'descend','descend'});
writetable(final_summary_rows, summary_csv);
final_validation_summary = table( ...
    string(final_summary_rows.dataset), ...
    string(final_summary_rows.pipeline_type), ...
    string(final_summary_rows.mi_window), ...
    string(final_summary_rows.channel_mode), ...
    final_summary_rows.csp_components, ...
    final_summary_rows.accuracy, ...
    final_summary_rows.f1_score, ...
    'VariableNames', {'dataset','final_pipeline','mi_window','channels','csp_components','accuracy','f1_score'});
final_validation_summary = sortrows(final_validation_summary, 'dataset');
try
    analysis = build_research_analysis(summary_rows, toc(framework_start));
catch analysisME
    warning('Research analysis generation failed: %s', analysisME.message);
    log_line(log_file, sprintf('Warning: research analysis generation failed: %s', analysisME.message));
    analysis = struct();
    analysis.total_runtime_sec = toc(framework_start);
    analysis.research_summary_rows = summary_rows;
    analysis.research_comparison_rows = table();
    analysis.research_ranking_rows = table();
    analysis.research_summary_text = sprintf('Research analysis unavailable: %s', analysisME.message);
    analysis.best_overall = table();
    analysis.worst_overall = table();
    analysis.best_window = "";
    analysis.best_channel = "";
    analysis.best_bandpass = "";
    analysis.best_csp = "";
end

research_summary_rows = analysis.research_summary_rows;
research_comparison_rows = analysis.research_comparison_rows;
research_ranking_rows = analysis.research_ranking_rows;

try
    writetable(research_summary_rows, research_summary_csv);
    writetable(research_comparison_rows, research_comparison_csv);
    writetable(research_ranking_rows, research_ranking_csv);
    save(summary_mat, 'summary_rows', 'final_summary_rows', 'final_validation_summary', 'analysis');
    save(research_summary_mat, 'research_summary_rows', 'analysis');
    save(research_comparison_mat, 'research_comparison_rows', 'analysis');
    save(research_ranking_mat, 'research_ranking_rows', 'analysis');
    write_text_file(research_summary_txt, analysis.research_summary_text);
    append_text_block(notes_txt, analysis.research_summary_text);
catch exportME
    warning('Research note export failed: %s', exportME.message);
    log_line(log_file, sprintf('Warning: research note export failed: %s', exportME.message));
end

results = struct();
results.summary = summary_rows;
results.final_summary = final_summary_rows;
results.final_validation_summary = final_validation_summary;
results.summary_csv = summary_csv;
results.summary_mat = summary_mat;
results.log_file = log_file;
results.research_summary_csv = research_summary_csv;
results.research_summary_mat = research_summary_mat;
results.research_summary_txt = research_summary_txt;
results.research_comparison_csv = research_comparison_csv;
results.research_comparison_mat = research_comparison_mat;
results.research_ranking_csv = research_ranking_csv;
results.research_ranking_mat = research_ranking_mat;
results.research_notes_txt = notes_txt;
results.total_runtime_sec = analysis.total_runtime_sec;

fprintf('\nOptimization results saved:\n');
fprintf('  %s\n', summary_csv);
fprintf('  %s\n', summary_mat);
fprintf('  %s\n', log_file);
fprintf('  %s\n', research_summary_csv);
fprintf('  %s\n', notes_txt);
disp(final_validation_summary);
end

function detail = run_cho_experiment(project_root, details_root, experiment_id, subject_mode, ...
    pipeline_type, mi_window, bandpass_range, csp_components, channel_mode, log_file)

hybrid_dir = fullfile(project_root, 'Hybrid_CSP_Riemannian_HO_CatBoost_SingleFile');
cho_dir = fullfile(project_root, 'gigadb');
addpath(hybrid_dir);
addpath(fullfile(project_root, 'PhysioNet_8Case_Benchmark', 'benchmark_framework', 'functions'));

subject_ids = subject_ids_for_mode(subject_mode, 52, 1, 3);
rows = table();

detail_dir = fullfile(details_root, experiment_id);
ensure_local_dir(detail_dir);
detail_csv = fullfile(detail_dir, [experiment_id '_subjects.csv']);
detail_mat = fullfile(detail_dir, [experiment_id '_subjects.mat']);
mi_top_k = NaN;
if string(pipeline_type) == "fbcsp_mi"
    mi_top_k = 20;
end
detail_cfg = struct('dataset', "cho", 'pipeline_type', string(pipeline_type), ...
    'channel_mode', string(channel_mode), 'mi_window', mi_window, ...
    'bandpass_range', bandpass_range, 'csp_components', csp_components, ...
    'mi_top_k', mi_top_k);
experiment_tic = tic;

for subject_id = subject_ids
    subject_name = sprintf('s%02d', subject_id);
    mat_file = fullfile(cho_dir, [subject_name '.mat']);
    log_line(log_file, sprintf('  Cho %s loading.', subject_name));
    subject_tic = tic;

    try
        S = load(mat_file, 'eeg');
        [raw_eeg, events, fs, n_trials] = cho_trials_to_raw_events(S.eeg);

        % Legacy branches keep the same classifier/HO path. FBCSP-MI switches
        % to the isolated FBCSP + MI + linear SVM path.
        cfg = struct();
        cfg.pipeline_type = char(pipeline_type);
        cfg.fs = fs;
        cfg.preprocessing.tmin = mi_window(1);
        cfg.preprocessing.tmax = mi_window(2);
        cfg.preprocessing.bandpass = bandpass_range;
        cfg.preprocessing.use_temporal_augmentation = false;
        cfg.features.single_csp_band = bandpass_range;
        cfg.features.csp_components = csp_components;
        cfg.channel_mode = char(channel_mode);
        cfg.classifier.svm_kernel = 'linear';
        cfg.runtime.fast_debug_mode = fast_debug_enabled_for_subject_mode(subject_mode);
        cfg.fbcsp_mi = struct('bands', [4 8; 8 12; 12 16; 16 20; 20 24; 24 30], ...
            'top_k', 20, 'mi_bins', 8, 'filter_order', 4);
        if string(pipeline_type) == "fbcsp_mi"
            cfg.classification.decision_head = 'linear_svm';
            cfg.classification.ho_decision_head = 'linear_svm';
            cfg.classification.use_weighted_ensemble = false;
        else
            cfg.classification.decision_head = 'ensemble';
            cfg.classification.ho_decision_head = 'ensemble';
        end

        cfg = apply_optional_ho_override(cfg);
        subject_results = run_hybrid_csp_riemannian_ho_catboost(raw_eeg, events, cfg);
        selected_features = mean([subject_results.fold_metrics.selected_feature_count]);
        subject_runtime_sec = toc(subject_tic);

        row = subject_result_row("cho", string(pipeline_type), string(channel_mode), ...
            mi_window, mi_window(2) - mi_window(1), bandpass_range, csp_components, subject_name, "OK", n_trials, ...
            subject_results.overall.accuracy, subject_results.overall.f1, ...
            selected_features, subject_runtime_sec, "");
        log_line(log_file, sprintf('  Cho %s OK Acc %.4f F1 %.4f.', ...
            subject_name, row.accuracy, row.f1_score));
    catch ME
        subject_runtime_sec = toc(subject_tic);
        row = subject_result_row("cho", string(pipeline_type), string(channel_mode), ...
            mi_window, mi_window(2) - mi_window(1), bandpass_range, csp_components, subject_name, "FAILED", 0, NaN, NaN, NaN, ...
            subject_runtime_sec, regexprep(ME.message, '[\r\n]+', ' '));
        log_line(log_file, sprintf('  Cho %s failed: %s', subject_name, ME.message));
    end

    rows = [rows; row]; %#ok<AGROW>
    writetable(rows, detail_csv);
    save(detail_mat, 'rows', 'detail_cfg');
end

detail = struct();
detail.rows = rows;
detail.csv_file = detail_csv;
detail.mat_file = detail_mat;
detail.completed_subjects = sum(rows.status == "OK");
detail.experiment_runtime_sec = toc(experiment_tic);
detail.mean_subject_runtime_sec = mean(rows.runtime_sec, 'omitnan');
end

function detail = run_physionet_experiment(project_root, details_root, experiment_id, subject_mode, ...
    pipeline_type, mi_window, bandpass_range, csp_components, channel_mode, log_file)

framework_dir = fullfile(project_root, 'PhysioNet_8Case_Benchmark', 'benchmark_framework');
functions_dir = fullfile(framework_dir, 'functions');
addpath(functions_dir);

base_cfg = default_benchmark_cfg(framework_dir);
base_cfg.subject_ids = subject_ids_for_mode(subject_mode, 109, 3, 10);
base_cfg.feature_type = char(pipeline_type);
base_cfg.classifier_type = 'svm';
base_cfg.classifier.svm_kernel = 'linear';
base_cfg.filter_type = 'butterworth';
base_cfg.case_index = 0;
base_cfg.case_id = experiment_id;
base_cfg.band = bandpass_range;
base_cfg.mi_window_sec = mi_window;
base_cfg.csp_components = csp_components;
base_cfg.channel_mode = char(channel_mode);
base_cfg.fbcsp_mi = struct('bands', [4 8; 8 12; 12 16; 16 20; 20 24; 24 30], ...
    'top_k', 20, 'mi_bins', 8, 'filter_order', 4);
base_cfg.runtime.enable_intermediate_cache = false;
base_cfg.runtime.fast_debug_mode = fast_debug_enabled_for_subject_mode(subject_mode);

if base_cfg.runtime.fast_debug_mode
    base_cfg.ho.iterations = 10;
    base_cfg.ho.agents = 4;
end

base_cfg = apply_optional_physionet_ho_override(base_cfg);

detail_dir = fullfile(details_root, experiment_id);
ensure_local_dir(detail_dir);
ensure_local_dir(fullfile(detail_dir, 'curves'));
ensure_local_dir(fullfile(detail_dir, 'mat'));
detail_csv = fullfile(detail_dir, [experiment_id '_subjects.csv']);
detail_mat = fullfile(detail_dir, [experiment_id '_subjects.mat']);
mi_top_k = NaN;
if string(pipeline_type) == "fbcsp_mi"
    mi_top_k = 20;
end
detail_cfg = struct('dataset', "physionet", 'pipeline_type', string(pipeline_type), ...
    'channel_mode', string(channel_mode), 'mi_window', mi_window, ...
    'bandpass_range', bandpass_range, 'csp_components', csp_components, ...
    'mi_top_k', mi_top_k);
experiment_tic = tic;

base_cfg.case_root = detail_dir;
base_cfg.case_csv = detail_csv;
base_cfg.case_log = fullfile(detail_dir, [experiment_id '_case_log.txt']);

rows = table();
for subject_id = base_cfg.subject_ids
    subject_name = sprintf('s%03d', subject_id);
    subject_tic = tic;
    try
        log_line(log_file, sprintf('  PhysioNet %s processing.', subject_name));
        result_row = run_subject_case(subject_id, base_cfg);
        subject_runtime_sec = toc(subject_tic);
        row = subject_result_row("physionet", string(pipeline_type), string(channel_mode), ...
            mi_window, mi_window(2) - mi_window(1), bandpass_range, csp_components, subject_name, "OK", NaN, ...
            result_row.Accuracy, result_row.F1, result_row.SelectedFeatureCount, subject_runtime_sec, "");
        log_line(log_file, sprintf('  PhysioNet %s OK Acc %.4f F1 %.4f.', ...
            subject_name, row.accuracy, row.f1_score));
    catch ME
        subject_runtime_sec = toc(subject_tic);
        row = subject_result_row("physionet", string(pipeline_type), string(channel_mode), ...
            mi_window, mi_window(2) - mi_window(1), bandpass_range, csp_components, subject_name, "FAILED", 0, NaN, NaN, NaN, ...
            subject_runtime_sec, regexprep(ME.message, '[\r\n]+', ' '));
        log_line(log_file, sprintf('  PhysioNet %s failed: %s', subject_name, ME.message));
    end

    rows = [rows; row]; %#ok<AGROW>
    writetable(rows, detail_csv);
    save(detail_mat, 'rows', 'detail_cfg');
end

detail = struct();
detail.rows = rows;
detail.csv_file = detail_csv;
detail.mat_file = detail_mat;
detail.completed_subjects = sum(rows.status == "OK");
detail.experiment_runtime_sec = toc(experiment_tic);
detail.mean_subject_runtime_sec = mean(rows.runtime_sec, 'omitnan');
end

function [raw_eeg, events, fs, n_trials] = cho_trials_to_raw_events(eeg)
fs = eeg.srate;
n_channels = 68;
trial_len = numel(eeg.imagery_left) / (n_channels * eeg.n_imagery_trials);

left_trials = reshape(eeg.imagery_left, n_channels, trial_len, eeg.n_imagery_trials);
right_trials = reshape(eeg.imagery_right, n_channels, trial_len, eeg.n_imagery_trials);
all_trials = cat(3, left_trials, right_trials);
labels = [ones(eeg.n_imagery_trials, 1); 2 * ones(eeg.n_imagery_trials, 1)];

gap_samples = round(0.25 * fs);
n_trials = size(all_trials, 3);
raw_eeg = zeros(n_channels, n_trials * (trial_len + gap_samples));
events = zeros(n_trials, 2);
cursor = 1;

for trial_idx = 1:n_trials
    raw_eeg(:, cursor:(cursor + trial_len - 1)) = double(all_trials(:, :, trial_idx));
    events(trial_idx, :) = [cursor, labels(trial_idx)];
    cursor = cursor + trial_len + gap_samples;
end
end

function cfg = apply_optional_ho_override(cfg)
raw_iterations = str2double(getenv('OPT_HO_ITERATIONS'));
if ~isnan(raw_iterations) && raw_iterations > 0
    cfg.optimization.iterations = floor(raw_iterations);
end
raw_population = str2double(getenv('OPT_HO_POPULATION'));
if ~isnan(raw_population) && raw_population > 0
    cfg.optimization.population_size = floor(raw_population);
end
end

function cfg = apply_optional_physionet_ho_override(cfg)
raw_iterations = str2double(getenv('OPT_HO_ITERATIONS'));
if ~isnan(raw_iterations) && raw_iterations > 0
    cfg.ho.iterations = floor(raw_iterations);
end
raw_population = str2double(getenv('OPT_HO_POPULATION'));
if ~isnan(raw_population) && raw_population > 0
    cfg.ho.agents = floor(raw_population);
end
end

function tf = fast_debug_enabled_for_subject_mode(subject_mode)
tf = lower(string(subject_mode)) ~= "full";
end

function ids = subject_ids_for_mode(subject_mode, max_subject, smoke_count, pilot_count)
switch lower(string(subject_mode))
    case "smoke"
        ids = 1:min(smoke_count, max_subject);
    case "pilot"
        ids = 1:min(pilot_count, max_subject);
    case "full"
        ids = 1:max_subject;
    otherwise
        tokens = regexp(char(subject_mode), '[,; ]+', 'split');
        ids = [];
        for i = 1:numel(tokens)
            value = str2double(tokens{i});
            if ~isnan(value) && value >= 1 && value <= max_subject
                ids(end + 1) = floor(value); %#ok<AGROW>
            end
        end
        ids = unique(ids, 'stable');
        if isempty(ids)
            error('Unknown subject_mode: %s', subject_mode);
        end
end
end

function row = subject_result_row(dataset, pipeline_type, channel_mode, mi_window, window_duration_sec, bandpass_range, csp_components, ...
    subject_name, status, n_trials, accuracy, f1_score, selected_features, runtime_sec, notes)
row_timestamp = string(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
row = table( ...
    string(dataset), string(pipeline_type), string(channel_mode), window_to_string(mi_window), ...
    window_duration_sec, band_to_string(bandpass_range), csp_components, string(subject_name), string(status), n_trials, ...
    accuracy, f1_score, selected_features, runtime_sec, row_timestamp, string(notes), ...
    'VariableNames', {'dataset','pipeline_type','channel_mode','mi_window','window_duration_sec','bandpass_range', ...
    'csp_components','subject','status','n_trials','accuracy', ...
    'f1_score','selected_features','runtime_sec','timestamp','notes'});
end

function experiment_id = build_experiment_id(dataset, pipeline_type, mi_window, bandpass_range, csp_components, channel_mode, mi_top_k)
experiment_id = sprintf('%s_%s_%s_win%s_bp%s_csp%d', ...
    char(dataset), char(pipeline_type), char(channel_mode), ...
    compact_pair(mi_window), compact_pair(bandpass_range), csp_components);
if nargin >= 7 && ~isnan(mi_top_k)
    experiment_id = sprintf('%s_mi%d', experiment_id, round(mi_top_k));
end
experiment_id = regexprep(experiment_id, '[^A-Za-z0-9_]', 'p');
end

function value = compact_pair(pair_value)
value = sprintf('%g_%g', pair_value(1), pair_value(2));
end

function value = window_to_string(pair_value)
value = string(sprintf('[%g %g]', pair_value(1), pair_value(2)));
end

function value = band_to_string(pair_value)
value = string(sprintf('[%g %g]', pair_value(1), pair_value(2)));
end

function ensure_local_dir(path_value)
if ~exist(path_value, 'dir')
    mkdir(path_value);
end
end

function log_line(log_file, message)
fid = fopen(log_file, 'a');
if fid == -1
    fprintf('%s\n', message);
    return;
end
fprintf(fid, '[%s] %s\n', char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')), message);
fclose(fid);
fprintf('%s\n', message);
end

function append_text_block(path_value, text_value)
fid = fopen(path_value, 'a');
if fid == -1
    return;
end
fprintf(fid, '%s\n', text_value);
fclose(fid);
end

function write_text_file(path_value, text_value)
fid = fopen(path_value, 'w');
if fid == -1
    return;
end
fprintf(fid, '%s\n', text_value);
fclose(fid);
end

function analysis = build_research_analysis(summary_rows, total_runtime_sec)
summary_rows = add_window_duration_columns(summary_rows);
if has_table_field(summary_rows, 'status')
    valid_rows = summary_rows(summary_rows.status == "OK", :);
else
    valid_rows = table();
end

analysis = struct();
analysis.total_runtime_sec = total_runtime_sec;
analysis.research_summary_rows = summary_rows;
analysis.research_comparison_rows = build_comparison_rows(valid_rows);
analysis.research_ranking_rows = build_ranking_rows(valid_rows);
analysis.research_summary_text = build_research_summary_text(summary_rows, valid_rows, total_runtime_sec);
analysis.best_overall = best_row_by_accuracy(valid_rows);
analysis.worst_overall = worst_row_by_accuracy(valid_rows);
analysis.best_window = best_group_value(valid_rows, 'window_duration_sec');
analysis.best_channel = best_group_value(valid_rows, 'channel_mode');
analysis.best_bandpass = best_group_value(valid_rows, 'bandpass_range');
analysis.best_csp = best_group_value(valid_rows, 'csp_components');
end

function rows = add_window_duration_columns(rows)
if isempty(rows) || ~has_table_field(rows, 'mi_window')
    return;
end
rows.window_start_sec = nan(height(rows), 1);
rows.window_end_sec = nan(height(rows), 1);
rows.window_duration_sec = nan(height(rows), 1);

for i = 1:height(rows)
    [start_sec, end_sec, duration_sec] = parse_window_value(rows.mi_window(i));
    rows.window_start_sec(i) = start_sec;
    rows.window_end_sec(i) = end_sec;
    rows.window_duration_sec(i) = duration_sec;
end
end

function [start_sec, end_sec, duration_sec] = parse_window_value(window_str)
tokens = regexp(char(window_str), '\[([\d\.]+)\s+([\d\.]+)\]', 'tokens', 'once');
if isempty(tokens)
    start_sec = NaN;
    end_sec = NaN;
    duration_sec = NaN;
    return;
end
start_sec = str2double(tokens{1});
end_sec = str2double(tokens{2});
duration_sec = end_sec - start_sec;
end

function comparison_rows = build_comparison_rows(valid_rows)
comparison_rows = table();
comparison_rows = [comparison_rows; aggregate_by_field(valid_rows, 'pipeline_type', 'Feature Combination Comparison')];
comparison_rows = [comparison_rows; aggregate_by_field(valid_rows, 'channel_mode', 'Channel Configuration')];
comparison_rows = [comparison_rows; aggregate_by_field(valid_rows, 'dataset', 'Dataset Comparison')];
comparison_rows = [comparison_rows; aggregate_by_field(valid_rows, 'window_duration_sec', 'Window Duration Comparison')];
comparison_rows = [comparison_rows; aggregate_by_field(valid_rows, 'bandpass_range', 'Bandpass Comparison')];
comparison_rows = [comparison_rows; aggregate_by_field(valid_rows, 'csp_components', 'CSP Component Comparison')];
end

function out_tbl = aggregate_by_field(rows, field_name, comparison_group)
out_tbl = table();
if isempty(rows)
    return;
end
runtime_field = resolve_runtime_field(rows);
if ~has_table_field(rows, field_name) || ~has_table_field(rows, 'accuracy') || ~has_table_field(rows, 'f1_score') || runtime_field == ""
    return;
end

values = rows.(field_name);
if iscell(values)
    values = string(values);
end
if iscategorical(values)
    values = string(values);
end
if ~isstring(values) && ~isnumeric(values)
    values = string(values);
end

[unique_vals, ~, group_idx] = unique(values, 'stable');
for i = 1:numel(unique_vals)
    idx = group_idx == i;
    subset = rows(idx, :);
    item_label = value_to_label(unique_vals(i), field_name);
    out_tbl = [out_tbl; table( ...
        string(comparison_group), string(field_name), string(item_label), ...
        sum(idx), mean(subset.accuracy, 'omitnan'), mean(subset.f1_score, 'omitnan'), ...
        mean(subset.(runtime_field), 'omitnan'), ...
        'VariableNames', {'comparison_group','field_name','item','experiment_count', ...
        'mean_accuracy','mean_f1','mean_runtime_sec'})]; %#ok<AGROW>
end
end

function ranking_rows = build_ranking_rows(valid_rows)
ranking_rows = table();
if isempty(valid_rows)
    return;
end
runtime_field = resolve_runtime_field(valid_rows);
if runtime_field == ""
    return;
end

top_experiments = sortrows(valid_rows, ["accuracy","f1_score",runtime_field], ["descend","descend","ascend"]);
top_experiments = top_experiments(1:min(5, height(top_experiments)), :);
top_experiments = table( ...
    repmat("top_experiments", height(top_experiments), 1), ...
    (1:height(top_experiments)).', ...
    string(top_experiments.dataset), string(top_experiments.pipeline_type), string(top_experiments.channel_mode), ...
    string(top_experiments.mi_window), string(top_experiments.bandpass_range), top_experiments.csp_components, ...
    string(top_experiments.subject), top_experiments.accuracy, top_experiments.f1_score, top_experiments.(runtime_field), ...
    'VariableNames', {'ranking_type','rank','dataset','pipeline_type','channel_mode','mi_window','bandpass_range', ...
    'csp_components','subject','accuracy','f1_score','runtime_sec'});

window_tbl = aggregate_group_ranking(valid_rows, 'window_duration_sec', 'top_windows');
channel_tbl = aggregate_group_ranking(valid_rows, 'channel_mode', 'top_channel_configs');

ranking_rows = [top_experiments; window_tbl; channel_tbl];
end

function ranking_tbl = aggregate_group_ranking(rows, field_name, rank_type)
ranking_tbl = table();
if isempty(rows)
    return;
end
runtime_field = resolve_runtime_field(rows);
if ~has_table_field(rows, field_name) || ~has_table_field(rows, 'accuracy') || ~has_table_field(rows, 'f1_score') || runtime_field == ""
    return;
end
values = rows.(field_name);
if iscell(values) || iscategorical(values)
    values = string(values);
end
[unique_vals, ~, group_idx] = unique(values, 'stable');
agg = table();
for i = 1:numel(unique_vals)
    idx = group_idx == i;
    subset = rows(idx, :);
    agg = [agg; table( ...
        string(rank_type), string(value_to_label(unique_vals(i), field_name)), ...
        sum(idx), mean(subset.accuracy, 'omitnan'), mean(subset.f1_score, 'omitnan'), ...
        mean(subset.(runtime_field), 'omitnan'), ...
        'VariableNames', {'rank_type','item','experiment_count','mean_accuracy','mean_f1','mean_runtime_sec'})]; %#ok<AGROW>
end
agg = sortrows(agg, {'mean_accuracy','mean_f1'}, {'descend','descend'});
agg.rank = (1:height(agg)).';
agg = agg(1:min(5, height(agg)), :);
ranking_tbl = table( ...
    agg.rank_type, ...
    agg.rank, ...
    repmat("", height(agg), 1), ...
    repmat("", height(agg), 1), ...
    repmat("", height(agg), 1), ...
    string(agg.item), ...
    repmat("", height(agg), 1), ...
    nan(height(agg), 1), ...
    repmat("", height(agg), 1), ...
    agg.mean_accuracy, ...
    agg.mean_f1, ...
    agg.mean_runtime_sec, ...
    'VariableNames', {'ranking_type','rank','dataset','pipeline_type','channel_mode','mi_window','bandpass_range', ...
    'csp_components','subject','accuracy','f1_score','runtime_sec'});
end

function text = build_research_summary_text(summary_rows, valid_rows, total_runtime_sec)
lines = {};
lines{end + 1} = sprintf('Research summary generated at %s', char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')));
lines{end + 1} = sprintf('Total framework runtime: %.2f sec', total_runtime_sec);
lines{end + 1} = sprintf('Completed experiments: %d', height(summary_rows));
lines{end + 1} = sprintf('Successful experiments: %d', height(valid_rows));

best = best_row_by_accuracy(valid_rows);
worst = worst_row_by_accuracy(valid_rows);
best_window = best_group_value(valid_rows, 'window_duration_sec');
best_channel = best_group_value(valid_rows, 'channel_mode');
best_bandpass = best_group_value(valid_rows, 'bandpass_range');
best_csp = best_group_value(valid_rows, 'csp_components');
if ~isempty(best)
    lines{end + 1} = sprintf('Best overall: %s | %s | %s | %s | %s | CSP %d | Acc %.4f | F1 %.4f', ...
        char(best.dataset), char(best.pipeline_type), char(best.channel_mode), ...
        char(best.mi_window), char(best.bandpass_range), best.csp_components, best.accuracy, best.f1_score);
end
if ~isempty(worst)
    lines{end + 1} = sprintf('Worst overall: %s | %s | %s | %s | %s | CSP %d | Acc %.4f | F1 %.4f', ...
        char(worst.dataset), char(worst.pipeline_type), char(worst.channel_mode), ...
        char(worst.mi_window), char(worst.bandpass_range), worst.csp_components, worst.accuracy, worst.f1_score);
end
if ~isempty(best_window)
    lines{end + 1} = sprintf('Best window: %s', best_window);
end
if ~isempty(best_channel)
    lines{end + 1} = sprintf('Best channel configuration: %s', best_channel);
end
if ~isempty(best_bandpass)
    lines{end + 1} = sprintf('Best bandpass: %s', best_bandpass);
end
if ~isempty(best_csp)
    lines{end + 1} = sprintf('Best CSP component count: %s', best_csp);
end

lines = [lines, comparison_observation_lines(valid_rows)];
text = strjoin(lines, newline);
end

function lines = comparison_observation_lines(valid_rows)
lines = {};
if isempty(valid_rows)
    return;
end

dataset_tbl = aggregate_by_field(valid_rows, 'dataset', 'Dataset Comparison');
channel_tbl = aggregate_by_field(valid_rows, 'channel_mode', 'Channel Configuration');
pipeline_tbl = aggregate_by_field(valid_rows, 'pipeline_type', 'Feature Combination Comparison');
band_tbl = aggregate_by_field(valid_rows, 'bandpass_range', 'Bandpass Comparison');
window_tbl = aggregate_by_field(valid_rows, 'window_duration_sec', 'Window Duration Comparison');
csp_tbl = aggregate_by_field(valid_rows, 'csp_components', 'CSP Component Comparison');

best_dataset = best_item(dataset_tbl);
best_channel = best_item(channel_tbl);
best_pipeline = best_item(pipeline_tbl);
best_band = best_item(band_tbl);
best_window = best_item(window_tbl);
best_csp = best_item(csp_tbl);

if ~isempty(best_dataset)
    lines{end + 1} = sprintf('Dataset comparison: %s is currently leading by mean accuracy.', best_dataset);
    cho_mean = item_mean(dataset_tbl, 'cho');
    phys_mean = item_mean(dataset_tbl, 'physionet');
    if ~isnan(cho_mean) && ~isnan(phys_mean) && phys_mean > cho_mean
        lines{end + 1} = 'PhysioNet currently achieving higher overall accuracy than Cho2017.';
    end
end
if ~isempty(best_channel)
    lines{end + 1} = sprintf('Channel comparison: %s is currently the strongest channel configuration.', best_channel);
    all_mean = item_mean(channel_tbl, 'all');
    basic_mean = item_mean(channel_tbl, 'motor_basic');
    extended_mean = item_mean(channel_tbl, 'motor_extended');
    if ~isnan(all_mean) && ~isnan(basic_mean) && all_mean > basic_mean
        lines{end + 1} = 'All-channel CSP currently outperforming motor_basic.';
        lines{end + 1} = 'Channel reduction may remove useful spatial covariance information.';
    end
    if ~isnan(extended_mean) && ~isnan(basic_mean) && extended_mean > basic_mean
        lines{end + 1} = 'motor_extended currently outperforming motor_basic.';
    end
    if ~isnan(all_mean) && ~isnan(extended_mean) && all_mean > extended_mean
        lines{end + 1} = 'All-channel CSP currently outperforming motor_extended.';
    end
end
if ~isempty(best_pipeline)
    lines{end + 1} = sprintf('Feature comparison: %s currently has the best mean accuracy.', best_pipeline);
    spatial_mean = item_mean(pipeline_tbl, 'spatial_only');
    spectral_mean = item_mean(pipeline_tbl, 'spatial_spectral');
    temporal_mean = item_mean(pipeline_tbl, 'spatial_temporal');
    spectral_temporal_mean = item_mean(pipeline_tbl, 'spatial_spectral_temporal');
    riemannian_mean = item_mean(pipeline_tbl, 'spatial_spectral_temporal_riemannian');
    legacy_csp_mean = item_mean(pipeline_tbl, 'csp');
    legacy_de_mean = item_mean(pipeline_tbl, 'csp_de');
    legacy_temporal_mean = item_mean(pipeline_tbl, 'csp_temporal');
    legacy_riemannian_mean = item_mean(pipeline_tbl, 'csp_riemannian');
    legacy_fbcsp_mean = item_mean(pipeline_tbl, 'csp_fbcsp');
    mi_fbcsp_mean = item_mean(pipeline_tbl, 'fbcsp_mi');
    if ~isnan(spatial_mean) && ~isnan(temporal_mean)
        if spatial_mean > temporal_mean
            lines{end + 1} = 'Spatial-only currently outperforming Spatial + Temporal.';
        else
            lines{end + 1} = 'Spatial + Temporal currently outperforming Spatial-only.';
        end
    end
    if ~isnan(spectral_mean) && ~isnan(spatial_mean) && spectral_mean > spatial_mean
        lines{end + 1} = 'Spatial + Spectral currently outperforming Spatial-only.';
    end
    if ~isnan(spectral_temporal_mean) && ~isnan(spectral_mean) && spectral_temporal_mean > spectral_mean
        lines{end + 1} = 'Spatial + Spectral + Temporal currently outperforming Spatial + Spectral.';
    end
    if ~isnan(riemannian_mean) && ~isnan(spectral_temporal_mean) && riemannian_mean > spectral_temporal_mean
        lines{end + 1} = 'Spatial + Spectral + Temporal + Riemannian currently outperforming Spatial + Spectral + Temporal.';
    end
    if ~isnan(legacy_csp_mean) && ~isnan(legacy_temporal_mean)
        if legacy_csp_mean > legacy_temporal_mean
            lines{end + 1} = 'Legacy CSP currently outperforming legacy CSP + Temporal.';
        else
            lines{end + 1} = 'Legacy CSP + Temporal currently outperforming legacy CSP.';
        end
    end
    if ~isnan(legacy_de_mean) && ~isnan(legacy_csp_mean) && legacy_de_mean > legacy_csp_mean
        lines{end + 1} = 'Legacy CSP + Differential Entropy currently outperforming legacy CSP.';
    end
    if ~isnan(legacy_riemannian_mean) && ~isnan(legacy_csp_mean) && legacy_riemannian_mean > legacy_csp_mean
        lines{end + 1} = 'Legacy CSP + Riemannian currently outperforming legacy CSP.';
    end
    if ~isnan(legacy_fbcsp_mean) && ~isnan(legacy_csp_mean) && legacy_fbcsp_mean > legacy_csp_mean
        lines{end + 1} = 'Legacy CSP + FBCSP currently outperforming legacy CSP.';
    end
    if ~isnan(mi_fbcsp_mean) && ~isnan(legacy_csp_mean) && mi_fbcsp_mean > legacy_csp_mean
        lines{end + 1} = 'FBCSP + MI currently outperforming legacy CSP.';
    end
end
if ~isempty(best_band)
    lines{end + 1} = sprintf('Bandpass comparison: %s currently looks strongest.', best_band);
    bp_830 = item_mean(band_tbl, '[8 30]');
    bp_835 = item_mean(band_tbl, '[8 35]');
    if ~isnan(bp_830) && ~isnan(bp_835) && bp_830 > bp_835
        lines{end + 1} = '8-30 Hz currently appears strongest.';
    end
end
if ~isempty(best_window)
    lines{end + 1} = sprintf('Window comparison: %s currently looks strongest.', best_window);
    w3 = item_mean(window_tbl, '3.0 sec');
    w4 = item_mean(window_tbl, '4.0 sec');
    if ~isnan(w3) && ~isnan(w4)
        if w4 > w3
            lines{end + 1} = '4-second MI windows currently outperforming 3-second windows.';
        else
            lines{end + 1} = '3-second MI windows currently outperforming 4-second windows.';
        end
    end
end
if ~isempty(best_csp)
    lines{end + 1} = sprintf('CSP comparison: %s currently looks strongest.', best_csp);
    csp4 = item_mean(csp_tbl, '4 components');
    csp6 = item_mean(csp_tbl, '6 components');
    if ~isnan(csp4) && ~isnan(csp6)
        if csp4 > csp6
            lines{end + 1} = 'Lower CSP dimensions (4) currently look more stable than 6.';
        else
            lines{end + 1} = 'CSP component count 6 currently looks stronger than 4.';
        end
    end
end
end

function label = best_item(tbl)
label = '';
if isempty(tbl)
    return;
end
[~, idx] = max(tbl.mean_accuracy + 0.1 * tbl.mean_f1);
label = string(tbl.item(idx));
label = char(label);
end

function value = item_mean(tbl, target_item)
value = NaN;
if isempty(tbl)
    return;
end
idx = strcmp(string(tbl.item), string(target_item));
if any(idx)
    value = tbl.mean_accuracy(find(idx, 1));
end
end

function row = best_row_by_accuracy(rows)
row = table();
if isempty(rows)
    return;
end
[~, idx] = max(rows.accuracy + 0.1 * rows.f1_score);
row = rows(idx, :);
end

function row = worst_row_by_accuracy(rows)
row = table();
if isempty(rows)
    return;
end
[~, idx] = min(rows.accuracy + 0.1 * rows.f1_score);
row = rows(idx, :);
end

function value = best_group_value(rows, field_name)
value = '';
if isempty(rows)
    return;
end
if ~has_table_field(rows, field_name) || ~has_table_field(rows, 'accuracy') || ~has_table_field(rows, 'f1_score')
    return;
end
tbl = aggregate_by_field(rows, field_name, 'tmp');
if isempty(tbl)
    return;
end
[~, idx] = max(tbl.mean_accuracy + 0.1 * tbl.mean_f1);
value = char(string(tbl.item(idx)));
end

function item_label = value_to_label(value, field_name)
if strcmp(field_name, 'window_duration_sec')
    if isnumeric(value)
        item_label = sprintf('%.1f sec', value);
    else
        item_label = char(string(value));
    end
elseif strcmp(field_name, 'csp_components')
    item_label = sprintf('%s components', char(string(value)));
else
    item_label = char(string(value));
end
end

function note_text = build_experiment_note(row, ~, summary_rows)
note_lines = {};
note_lines{end + 1} = sprintf('[%s] %s | %s | %s | window=%s | band=%s | CSP=%d | acc=%.4f | f1=%.4f | runtime=%.2fs | subjects=%d', ...
    char(row.timestamp), char(row.dataset), char(row.pipeline_type), char(row.channel_mode), ...
    char(row.mi_window), char(row.bandpass_range), row.csp_components, row.accuracy, row.f1_score, ...
    row.experiment_runtime_sec, row.completed_subjects);

if has_table_field(summary_rows, 'status')
    valid_rows = summary_rows(summary_rows.status == "OK", :);
else
    valid_rows = table();
end
if ~isempty(valid_rows)
    best = best_row_by_accuracy(valid_rows);
    if ~isempty(best) && strcmp(row.dataset, best.dataset) && strcmp(row.pipeline_type, best.pipeline_type) && ...
            strcmp(row.channel_mode, best.channel_mode) && strcmp(row.mi_window, best.mi_window) && ...
            strcmp(row.bandpass_range, best.bandpass_range) && row.csp_components == best.csp_components
        note_lines{end + 1} = 'Observation: this configuration is the current best performer so far.';
    end
end

if ~isempty(valid_rows)
    note_lines = [note_lines, comparison_observation_lines(valid_rows)];
end

note_text = strjoin(note_lines, newline);
end

function tf = has_table_field(rows, field_name)
tf = istable(rows) && ismember(field_name, rows.Properties.VariableNames);
end

function field_name = resolve_runtime_field(rows)
field_name = "";
if ~istable(rows)
    return;
end
if has_table_field(rows, "runtime_sec")
    field_name = "runtime_sec";
elseif has_table_field(rows, "experiment_runtime_sec")
    field_name = "experiment_runtime_sec";
elseif has_table_field(rows, "mean_subject_runtime_sec")
    field_name = "mean_subject_runtime_sec";
end
end
