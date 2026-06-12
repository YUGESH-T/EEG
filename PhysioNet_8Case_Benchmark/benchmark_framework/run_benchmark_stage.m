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
