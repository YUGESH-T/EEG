function result_row = run_subject_case(subject_num, case_cfg)
subject_name = sprintf('s%03d', subject_num);
[X_left_f, X_right_f] = load_or_build_filtered_trials(subject_num, case_cfg);

labels = [ones(size(X_left_f, 3), 1); 2 * ones(size(X_right_f, 3), 1)];
split = create_subject_split(labels, subject_num, case_cfg);
feature_bundle = load_or_build_feature_bundle(subject_num, subject_name, X_left_f, X_right_f, split, case_cfg);

if strcmpi(case_cfg.feature_type, 'fbcsp_mi')
    fbcsp_case_cfg = case_cfg;
    fbcsp_case_cfg.classifier.svm_kernel = 'linear';
    [train_X_norm, test_X_norm] = zscore_train_test(feature_bundle.train_X, feature_bundle.test_X);
    ho_result = struct();
    ho_result.selected_idx = feature_bundle.selected_feature_idx;
    ho_result.selected_count = feature_bundle.selected_feature_count;
    ho_result.best_score = NaN;
    ho_result.best_pos = [];
    ho_result.ho_curve = [];
    ho_result.selection_method = 'mutual_information';
    eval_result = train_and_evaluate_classifier(train_X_norm, feature_bundle.train_Y, ...
        test_X_norm, feature_bundle.test_Y, fbcsp_case_cfg);
else
    [train_X_norm, test_X_norm] = zscore_train_test(feature_bundle.train_X, feature_bundle.test_X);
    ho_result = run_ho_feature_selection(train_X_norm, feature_bundle.train_Y, case_cfg, subject_num);

    train_sel = train_X_norm(:, ho_result.selected_idx);
    test_sel = test_X_norm(:, ho_result.selected_idx);
    eval_result = train_and_evaluate_classifier(train_sel, feature_bundle.train_Y, test_sel, feature_bundle.test_Y, case_cfg);
end

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
cache_suffix = cache_mode_suffix(case_cfg);
cache_file = fullfile(cache_root, sprintf('s%03d_%s%s.mat', subject_num, case_cfg.filter_type, cache_suffix));

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
cache_suffix = cache_mode_suffix(case_cfg);
if strcmpi(case_cfg.feature_type, 'fbcsp_mi')
    cache_suffix = [cache_suffix sprintf('_k%d', resolve_fbcsp_mi_top_k(case_cfg))];
end
cache_file = fullfile(cache_root, sprintf('%s_%s_%s%s.mat', subject_name, case_cfg.filter_type, case_cfg.feature_type, cache_suffix));

if case_cfg.runtime.enable_intermediate_cache && exist(cache_file, 'file')
    S = load(cache_file, 'feature_bundle');
    feature_bundle = S.feature_bundle;
    return;
end

switch lower(case_cfg.feature_type)
    case {'spatial_only', 'csp'}
        feature_bundle = extract_csp_features(X_left_f, X_right_f, split, case_cfg);
    case 'spatial_spectral'
        feature_bundle = extract_spatial_spectral_features(X_left_f, X_right_f, split, case_cfg);
    case 'spatial_temporal'
        feature_bundle = extract_spatial_temporal_features(X_left_f, X_right_f, split, case_cfg);
    case 'spatial_spectral_temporal'
        feature_bundle = extract_spatial_spectral_temporal_features(X_left_f, X_right_f, split, case_cfg);
    case 'spatial_spectral_temporal_riemannian'
        feature_bundle = extract_spatial_spectral_temporal_riemannian_features(X_left_f, X_right_f, split, case_cfg);
    case 'csp_de'
        feature_bundle = extract_csp_de_features(X_left_f, X_right_f, split, case_cfg);
    case 'csp_temporal'
        feature_bundle = extract_csp_temporal_features(X_left_f, X_right_f, split, case_cfg);
    case 'csp_riemannian'
        feature_bundle = extract_csp_riemannian_features(X_left_f, X_right_f, split, case_cfg);
    case {'csp_fbcsp', 'fbcsp'}
        feature_bundle = extract_csp_fbcsp_features(X_left_f, X_right_f, split, case_cfg);
    case 'fbcsp_mi'
        feature_bundle = extract_fbcsp_mi_features(X_left_f, X_right_f, split, case_cfg);
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

function suffix = cache_mode_suffix(case_cfg)
suffix = "";
if isfield(case_cfg, 'channel_mode') && strlength(string(case_cfg.channel_mode)) > 0
    suffix = "_" + lower(string(case_cfg.channel_mode));
end
suffix = char(suffix);
end

function top_k = resolve_fbcsp_mi_top_k(case_cfg)
top_k = 20;
if isfield(case_cfg, 'fbcsp_mi') && isfield(case_cfg.fbcsp_mi, 'top_k') && ~isempty(case_cfg.fbcsp_mi.top_k)
    top_k = case_cfg.fbcsp_mi.top_k;
end
end
