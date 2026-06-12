function results = run_csp_pipeline_experiments(mode, cfg_override)
%RUN_CSP_PIPELINE_EXPERIMENTS CSP-focused PhysioNet optimization pipeline.
%
%   Phase 1: Window & CSP Component Optimization (using CSP+Spectral)
%   Phase 2: Ablation Experiments (E1-E13) using best Phase 1 window
%   Phase 3: Shrinkage CSP Experiments (E14-E16)
%
%   Usage:
%     run_csp_pipeline_experiments('smoke');
%     run_csp_pipeline_experiments('pilot');
%     run_csp_pipeline_experiments('full');

    if nargin < 1, mode = 'smoke'; end
    if nargin < 2, cfg_override = struct(); end

    cfg  = default_sweep_config();
    cfg  = merge_sweep_config(cfg, cfg_override);
    rng(cfg.seed, 'twister');
    mode = lower(string(mode));

    script_dir  = fileparts(mfilename('fullpath'));
    addpath(fullfile(script_dir, 'functions'));
    results_dir = fullfile(script_dir, 'results');
    if ~exist(results_dir, 'dir'), mkdir(results_dir); end

    run_stamp   = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
    log_file    = fullfile(results_dir, ['experiment_log_' run_stamp '.txt']);
    summary_csv = fullfile(results_dir, ['summary_' run_stamp '.csv']);
    
    subject_ids = resolve_subject_ids(mode);

    fprintf('\n================================================================\n');
    fprintf('  PhysioNet CSP Optimization Sweep V3\n');
    fprintf('  Mode: %s | Subjects: %d | Folds: %d | Seed: %d\n', ...
        mode, numel(subject_ids), cfg.cv_folds, cfg.seed);
    fprintf('================================================================\n\n');

    % =====================================================================
    % PHASE 1: WINDOW & CSP OPTIMIZATION
    % =====================================================================
    fprintf('--- PHASE 1: Window & CSP Component Optimization ---\n');
    
    windows = {[0.5 3.5], [0.75 3.5], [1.0 4.0]};
    csp_comps = [14, 16];
    
    phase1_configs = [];
    n = 0;
    for wi = 1:numel(windows)
        for ci = 1:numel(csp_comps)
            n = n + 1;
            id = sprintf('W%d', n);
            desc = sprintf('CSP=%d, Win=[%.2f %.2f] (CSP+Spec)', ...
                csp_comps(ci), windows{wi}(1), windows{wi}(2));
            phase1_configs = [phase1_configs, make_config(id, 'standard', ...
                csp_comps(ci), true, true, false, 'none', 0, windows{wi}, desc)]; %#ok<AGROW>
        end
    end

    all_results = table();
    sweep_tic = tic;
    
    % Run Phase 1
    for cfg_idx = 1:numel(phase1_configs)
        config = phase1_configs(cfg_idx);
        fprintf('\n[Phase 1] Config %d/%d: %s\n', cfg_idx, numel(phase1_configs), config.id);
        all_results = run_config(config, subject_ids, cfg, log_file, all_results);
    end
    
    % Select best config from Phase 1
    [best_window, best_csp] = select_best_window(all_results, phase1_configs);
    fprintf('\n>>> PHASE 1 COMPLETE. Selected Best: Window [%.2f %.2f], CSP=%d\n\n', ...
        best_window(1), best_window(2), best_csp);
        
    % =====================================================================
    % PHASE 2 & 3: MAIN EXPERIMENTS
    % =====================================================================
    fprintf('--- PHASE 2 & 3: Main E1-E16 Ablation & Shrinkage ---\n');
    
    main_configs = build_main_configs(best_window, best_csp);
    
    for cfg_idx = 1:numel(main_configs)
        config = main_configs(cfg_idx);
        fprintf('\n[Phase 2/3] Config %d/%d: %s\n', cfg_idx, numel(main_configs), config.id);
        all_results = run_config(config, subject_ids, cfg, log_file, all_results);
    end

    total_runtime = toc(sweep_tic);
    
    try
        writetable(all_results, summary_csv);
    catch ME_save
        fprintf('Warning: could not save results: %s\n', ME_save.message);
    end

    results = generate_ranked_summary(all_results, total_runtime, summary_csv, log_file);
end

% =========================================================================
% CONFIGURATION DEFINITIONS
% =========================================================================

function configs = build_main_configs(win, csp)
    n = 0;
    
    % Phase 2: E1-E13
    % (id, csp_type, csp_comps, use_csp, use_spec, use_temp, sel_method, sel_k, window, desc)
    n=n+1; configs(n) = make_config('E1', 'standard', csp, true, false, false, 'none', 0, win, 'CSP Baseline');
    n=n+1; configs(n) = make_config('E2', 'standard', csp, true, true, false, 'none', 0, win, 'CSP + Spectral');
    n=n+1; configs(n) = make_config('E3', 'standard', csp, true, true, false, 'anova', 32, win, 'CSP + Spectral + ANOVA');
    n=n+1; configs(n) = make_config('E4', 'standard', csp, true, true, false, 'mi', 32, win, 'CSP + Spectral + MI');
    n=n+1; configs(n) = make_config('E5', 'standard', csp, true, true, false, 'fisher', 32, win, 'CSP + Spectral + Fisher');
    
    n=n+1; configs(n) = make_config('E6', 'standard', csp, true, false, true, 'none', 0, win, 'CSP + Temporal');
    n=n+1; configs(n) = make_config('E7', 'standard', csp, true, false, true, 'anova', 32, win, 'CSP + Temporal + ANOVA');
    n=n+1; configs(n) = make_config('E8', 'standard', csp, true, false, true, 'mi', 32, win, 'CSP + Temporal + MI');
    n=n+1; configs(n) = make_config('E9', 'standard', csp, true, false, true, 'fisher', 32, win, 'CSP + Temporal + Fisher');
    
    n=n+1; configs(n) = make_config('E10', 'standard', csp, true, true, true, 'none', 0, win, 'CSP + Spectral + Temporal');
    n=n+1; configs(n) = make_config('E11', 'standard', csp, true, true, true, 'anova', 32, win, 'CSP + Spec + Temp + ANOVA');
    n=n+1; configs(n) = make_config('E12', 'standard', csp, true, true, true, 'mi', 32, win, 'CSP + Spec + Temp + MI');
    n=n+1; configs(n) = make_config('E13', 'standard', csp, true, true, true, 'fisher', 32, win, 'CSP + Spec + Temp + Fisher');
    
    % Phase 3: Shrinkage E14-E16
    n=n+1; configs(n) = make_config('E14', 'shrinkage', csp, true, true, false, 'none', 0, win, 'Shrinkage CSP + Spectral');
    n=n+1; configs(n) = make_config('E15', 'shrinkage', csp, true, false, true, 'none', 0, win, 'Shrinkage CSP + Temporal');
    n=n+1; configs(n) = make_config('E16', 'shrinkage', csp, true, true, true, 'none', 0, win, 'Shrinkage CSP + Spec + Temp');
end

function cfg = make_config(id, csp_type, csp_comps, use_csp, use_spec, use_temp, sel_method, sel_k, win, desc)
    cfg.id            = id;
    cfg.csp_type      = csp_type;
    cfg.csp_comps     = csp_comps;
    cfg.use_csp       = use_csp;
    cfg.use_spec      = use_spec;
    cfg.use_temp      = use_temp;
    cfg.sel_method    = sel_method;
    cfg.sel_k         = sel_k;
    cfg.mi_window     = win;
    cfg.description   = desc;
end

function cfg = default_sweep_config()
    cfg = struct();
    cfg.seed       = 42;
    cfg.fs         = 160;
    cfg.cv_folds   = 5;
    cfg.csp_beta   = 0.15;
    cfg.preprocessing_bandpass = [4 32];
    cfg.butterworth_order      = 4;
    cfg.run_ids             = [4 8 12];
    cfg.min_balanced_trials = 10;
    
    project_root  = fileparts(fileparts(mfilename('fullpath')));
    cfg.cache_dir = fullfile(project_root, 'physionet_cache');
    
    cfg.catboost.iterations    = 200;
    cfg.catboost.depth         = 6;
    cfg.catboost.learning_rate = 0.08;
end

% =========================================================================
% EXECUTION CORE
% =========================================================================

function all_results = run_config(config, subject_ids, cfg, log_file, all_results)
    fprintf('  Desc: %s | Win: [%.2f %.2f] | CSP: %d\n', config.description, config.mi_window(1), config.mi_window(2), config.csp_comps);
    config_tic = tic;
    
    for subj_idx = 1:numel(subject_ids)
        subject_id   = subject_ids(subj_idx);
        subject_name = sprintf('s%03d', subject_id);
        subject_tic  = tic;
        
        try
            [raw_eeg, events, subject_info] = load_physionet_raw(subject_id, cfg);
            [epochs, labels] = preprocess_and_epoch(raw_eeg, events, config.mi_window, cfg);

            labels     = labels(:);
            valid_mask = labels > 0;
            epochs     = epochs(:, :, valid_mask);
            labels     = labels(valid_mask);

            rng(cfg.seed + subject_id, 'twister');
            cv = cvpartition(labels, 'KFold', cfg.cv_folds);

            fold_accuracies  = zeros(cv.NumTestSets, 1);
            fold_f1s         = zeros(cv.NumTestSets, 1);
            feature_dim_pre  = 0;
            feature_dim_post = 0;

            for fold_idx = 1:cv.NumTestSets
                train_idx = training(cv, fold_idx);
                test_idx  = test(cv, fold_idx);

                X_train_ep = epochs(:, :, train_idx);
                y_train    = labels(train_idx);
                X_test_ep  = epochs(:, :, test_idx);
                y_test     = labels(test_idx);

                [b_csp, a_csp] = butter(4, [8 30] / (cfg.fs / 2), 'bandpass');
                X_train_f = filter_epochs_local(X_train_ep, b_csp, a_csp);
                X_test_f  = filter_epochs_local(X_test_ep,  b_csp, a_csp);

                [W, ~] = compute_csp_projection(X_train_f, y_train, config.csp_comps, config.csp_type, cfg.csp_beta);

                Z_train_f = project_csp_local(X_train_f, W);
                Z_test_f  = project_csp_local(X_test_f,  W);
                Z_train_b = project_csp_local(X_train_ep, W);
                Z_test_b  = project_csp_local(X_test_ep,  W);

                feat_train = [];
                feat_test  = [];
                
                if config.use_csp
                    csp_train = extract_logvar_local(Z_train_f);
                    csp_test  = extract_logvar_local(Z_test_f);
                    feat_train = [feat_train, csp_train]; %#ok<AGROW>
                    feat_test  = [feat_test,  csp_test]; %#ok<AGROW>
                end
                
                if config.use_spec
                    spec_train = extract_spectral_features_v2(Z_train_b, cfg.fs, '6band');
                    spec_test  = extract_spectral_features_v2(Z_test_b,  cfg.fs, '6band');
                    feat_train = [feat_train, spec_train]; %#ok<AGROW>
                    feat_test  = [feat_test,  spec_test]; %#ok<AGROW>
                end
                
                if config.use_temp
                    temp_train = extract_temporal_features(Z_train_b, cfg.fs);
                    temp_test  = extract_temporal_features(Z_test_b, cfg.fs);
                    feat_train = [feat_train, temp_train]; %#ok<AGROW>
                    feat_test  = [feat_test,  temp_test]; %#ok<AGROW>
                end

                feature_dim_pre = size(feat_train, 2);

                [feat_train, feat_test, ~] = normalize_features(feat_train, feat_test, 'zscore');

                if config.sel_k > 0 && config.sel_k < size(feat_train, 2)
                    if strcmp(config.sel_method, 'anova')
                        [feat_train, feat_test, ~, ~] = select_features_anova(feat_train, y_train, feat_test, config.sel_k);
                    elseif strcmp(config.sel_method, 'mi')
                        [feat_train, feat_test, ~, ~] = select_mutual_information_features(feat_train, y_train, feat_test, config.sel_k);
                    elseif strcmp(config.sel_method, 'fisher')
                        [feat_train, feat_test, ~, ~] = select_features_fisher(feat_train, y_train, feat_test, config.sel_k);
                    end
                end

                feature_dim_post = size(feat_train, 2);

                y_pred = classify_catboost(feat_train, y_train, feat_test, cfg);
                fold_m = compute_fold_metrics(y_test, y_pred);
                
                fold_accuracies(fold_idx) = fold_m.accuracy;
                fold_f1s(fold_idx)        = fold_m.f1;
            end

            mean_acc = mean(fold_accuracies);
            std_acc  = std(fold_accuracies);
            mean_f1  = mean(fold_f1s);

            subject_runtime = toc(subject_tic);

            row = build_result_row(config, subject_id, subject_name, ...
                'OK', mean_acc, std_acc, mean_f1, ...
                subject_info.n_trials, feature_dim_pre, feature_dim_post, ...
                subject_runtime, '');
            all_results = [all_results; row]; %#ok<AGROW>
            fprintf('  %s: Acc=%.2f%% | Dim: %d->%d | %.1fs\n', subject_name, mean_acc * 100, feature_dim_pre, feature_dim_post, subject_runtime);
            
        catch ME
            subject_runtime = toc(subject_tic);
            row = build_result_row(config, subject_id, subject_name, ...
                'ERROR', NaN, NaN, NaN, 0, 0, 0, subject_runtime, ME.message);
            all_results = [all_results; row]; %#ok<AGROW>
            fprintf('  %s: ERROR: %s\n', subject_name, ME.message);
        end
    end
end

% =========================================================================
% UTILITIES
% =========================================================================

function [best_window, best_csp] = select_best_window(all_results, phase1_configs)
    ok_rows = all_results(all_results.status == "OK", :);
    if isempty(ok_rows)
        best_window = [0.5 3.5]; best_csp = 16; return;
    end
    
    config_ids = unique({phase1_configs.id});
    best_acc = -1;
    best_id = '';
    
    for i = 1:numel(config_ids)
        cid = string(config_ids{i});
        subset = ok_rows(ok_rows.config_id == cid, :);
        if isempty(subset), continue; end
        m_acc = mean(subset.mean_accuracy, 'omitnan');
        if m_acc > best_acc
            best_acc = m_acc;
            best_id = cid;
        end
    end
    
    idx = find(strcmp({phase1_configs.id}, char(best_id)), 1);
    best_window = phase1_configs(idx).mi_window;
    best_csp = phase1_configs(idx).csp_comps;
end

function ids = resolve_subject_ids(mode)
    if strcmp(mode, 'smoke'), ids = 1:3;
    elseif strcmp(mode, 'pilot'), ids = 1:10;
    elseif strcmp(mode, 'full'), ids = 1:109;
    else, ids = 1:3;
    end
end

function cfg = merge_sweep_config(base, override)
    cfg = base; fields = fieldnames(override);
    for k = 1:numel(fields)
        f = fields{k};
        if isstruct(override.(f)) && isfield(cfg, f) && isstruct(cfg.(f))
            cfg.(f) = merge_sweep_config(cfg.(f), override.(f));
        else
            cfg.(f) = override.(f);
        end
    end
end

function X_f = filter_epochs_local(X, b, a)
    X_f = zeros(size(X));
    for e = 1:size(X, 3)
        X_f(:, :, e) = filtfilt(b, a, X(:, :, e)')';
    end
end

function Z = project_csp_local(X, W)
    n_comp = size(W, 2); n_time = size(X, 2); n_epochs = size(X, 3);
    Z = zeros(n_comp, n_time, n_epochs);
    for e = 1:n_epochs, Z(:, :, e) = W' * X(:, :, e); end
end

function features = extract_logvar_local(Z)
    n_epochs = size(Z, 3); n_comp = size(Z, 1);
    features = zeros(n_epochs, n_comp);
    for e = 1:n_epochs
        v = var(Z(:, :, e), 0, 2);
        features(e, :) = log(v / (sum(v) + eps) + eps)';
    end
end

function row = build_result_row(config, subject_id, subject_name, status, m_acc, s_acc, m_f1, n_trials, d_pre, d_post, runtime, err)
    row = table(string(config.id), string(config.csp_type), config.csp_comps, ...
        string(sprintf('[%.2f %.2f]', config.mi_window)), string(config.description), ...
        subject_id, string(subject_name), string(status), m_acc, s_acc, m_f1, ...
        n_trials, d_pre, d_post, runtime, string(err), ...
        'VariableNames', {'config_id', 'csp_type', 'csp_comps', 'mi_window', 'description', ...
        'subject_id', 'subject_name', 'status', 'mean_accuracy', 'std_accuracy', 'mean_f1', ...
        'n_trials', 'feature_dim_before', 'feature_dim_after', 'runtime_sec', 'error_message'});
end

function results = generate_ranked_summary(all_results, total_runtime, summary_csv, log_file)
    results.all_results = all_results;
    ok_rows = all_results(all_results.status == "OK" & startsWith(all_results.config_id, "E"), :);
    if isempty(ok_rows), return; end
    
    cids = unique(ok_rows.config_id, 'stable');
    st = table();
    for i = 1:numel(cids)
        sub = ok_rows(ok_rows.config_id == cids(i), :);
        st = [st; table(cids(i), sub.description(1), mean(sub.mean_accuracy), std(sub.mean_accuracy), ...
            round(mean(sub.feature_dim_after)), 'VariableNames', {'ID', 'Desc', 'Acc', 'Std', 'Dim'})]; %#ok<AGROW>
    end
    st = sortrows(st, 'Acc', 'descend');
    
    fprintf('\n================================================================\n');
    fprintf('  RANKED SUMMARY (E1-E16)\n');
    fprintf('================================================================\n');
    fprintf('%-5s %-30s %8s %8s %5s\n', 'ID', 'Description', 'MeanAcc', 'StdAcc', 'Dim');
    for r=1:height(st)
        fprintf('%-5s %-30s %7.2f%% %7.2f%% %5d\n', char(st.ID(r)), char(st.Desc(r)), st.Acc(r)*100, st.Std(r)*100, st.Dim(r));
    end
    
    % Compare against E2
    e2_mask = st.ID == "E2";
    if any(e2_mask)
        e2_acc = st.Acc(e2_mask);
        fprintf('\n  Baseline (E2): %.2f%%\n', e2_acc * 100);
        for r=1:height(st)
            if st.ID(r) == "E2", continue; end
            delta = (st.Acc(r) - e2_acc) * 100;
            fprintf('  %-5s vs baseline: %+.2f pp\n', char(st.ID(r)), delta);
        end
    end
    results.summary_table = st;
end
