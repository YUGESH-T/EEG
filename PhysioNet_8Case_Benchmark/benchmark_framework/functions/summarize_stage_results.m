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
