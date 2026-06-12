function results = run_multidomain_sweep(mode, cfg_override)
%RUN_MULTIDOMAIN_SWEEP PhysioNet EEGMMIDB multi-domain fusion experiment suite.
%
%   Runs 15 configurations testing:
%     - Spatial (CSP, FBCSP)
%     - Spectral (PSD, Mu/Beta, Entropy)
%     - Temporal (Hjorth, RMS, WL, MAV, AR)
%     - Geometric (Riemannian tangent space)
%     - Feature selection (ANOVA, MI, Fisher)
%
%   Usage:
%     results = run_multidomain_sweep('smoke');   % 3 subjects, quick validation
%     results = run_multidomain_sweep('pilot');   % 10 subjects
%     results = run_multidomain_sweep('full');    % 109 subjects (hours)
%
%   All CSP, Riemannian reference, normalization, and feature selection
%   fitting uses training-fold data ONLY.

    if nargin < 1, mode = 'smoke'; end
    if nargin < 2, cfg_override = struct(); end

    cfg  = default_sweep_config();
    cfg  = merge_sweep_config(cfg, cfg_override);
    rng(cfg.seed, 'twister');
    mode = lower(string(mode));

    % ----- Setup paths and directories -----------------------------------
    script_dir  = fileparts(mfilename('fullpath'));
    addpath(fullfile(script_dir, 'functions'));

    results_dir = fullfile(script_dir, 'results');
    if ~exist(results_dir, 'dir'), mkdir(results_dir); end

    run_stamp   = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
    log_file    = fullfile(results_dir, ['experiment_log_' run_stamp '.txt']);
    summary_csv = fullfile(results_dir, ['summary_' run_stamp '.csv']);
    summary_mat = fullfile(results_dir, ['summary_' run_stamp '.mat']);

    % ----- Subject selection ---------------------------------------------
    subject_ids = resolve_subject_ids(mode);

    % ----- Build ablation configs ----------------------------------------
    configs = build_ablation_configs();

    log_experiment(log_file, sprintf( ...
        'Multi-domain sweep started. Mode: %s | Subjects: %d | Configs: %d', ...
        mode, numel(subject_ids), numel(configs)));

    fprintf('\n================================================================\n');
    fprintf('  PhysioNet EEGMMIDB Multi-Domain Fusion Sweep V1\n');
    fprintf('  Mode: %s | Subjects: %d | Configurations: %d\n', ...
        mode, numel(subject_ids), numel(configs));
    fprintf('  CSP components: %d | CV folds: %d | Seed: %d\n', ...
        cfg.csp_components, cfg.cv_folds, cfg.seed);
    fprintf('================================================================\n\n');

    % ----- Main experiment loop ------------------------------------------
    all_results = table();
    sweep_tic   = tic;

    for cfg_idx = 1:numel(configs)
        config = configs(cfg_idx);

        fprintf('\n--- Config %d/%d: %s ---\n', cfg_idx, numel(configs), config.id);
        fprintf('  Spatial: %d | Spectral: %d | Temporal: %d | Riem: %d\n', ...
            config.use_spatial, config.use_spectral, config.use_temporal, config.use_riemannian);
        fprintf('  FS: %s (k=%d) | Window: [%.1f, %.1f]\n', ...
            config.fs_method, config.fs_k, config.mi_window(1), config.mi_window(2));

        log_experiment(log_file, sprintf('Config %s starting: %s', ...
            config.id, config.description));
        config_tic = tic;

        subject_accuracies = [];
        subject_f1s        = [];

        for subj_idx = 1:numel(subject_ids)
            subject_id   = subject_ids(subj_idx);
            subject_name = sprintf('s%03d', subject_id);
            subject_tic  = tic;

            try
                % --- Load raw EEG ----------------------------------------
                [raw_eeg, events, subject_info] = load_physionet_raw(subject_id, cfg);

                % --- Preprocess and epoch --------------------------------
                [epochs, labels] = preprocess_and_epoch( ...
                    raw_eeg, events, config.mi_window, cfg);

                labels     = labels(:);
                valid_mask = labels > 0;
                epochs     = epochs(:, :, valid_mask);
                labels     = labels(valid_mask);

                % --- 5-fold stratified CV --------------------------------
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

                    feat_train = [];
                    feat_test  = [];

                    % =====================================================
                    % 1. SPATIAL BRANCH (CSP / FBCSP)
                    % =====================================================
                    if config.use_spatial > 0
                        if config.use_spatial == 1 % Standard CSP
                            % Bandpass 8-30 Hz
                            [b_csp, a_csp] = butter(4, [8 30] / (cfg.fs / 2), 'bandpass');
                            X_train_f = filter_epochs_local(X_train_ep, b_csp, a_csp);
                            X_test_f  = filter_epochs_local(X_test_ep,  b_csp, a_csp);

                            [W, ~] = compute_csp_projection( ...
                                X_train_f, y_train, cfg.csp_components, 'standard', cfg.csp_beta);

                            Z_train = project_csp_local(X_train_f, W);
                            Z_test  = project_csp_local(X_test_f,  W);

                            sp_train = extract_logvar_local(Z_train);
                            sp_test  = extract_logvar_local(Z_test);
                        elseif config.use_spatial == 2 % FBCSP
                            [sp_train, sp_test] = extract_fbcsp_features( ...
                                X_train_ep, y_train, X_test_ep, cfg);
                        end

                        feat_train = [feat_train, sp_train];
                        feat_test  = [feat_test,  sp_test];
                    end

                    % =====================================================
                    % 2. SPECTRAL BRANCH (PSD, Mu/Beta, Entropy)
                    % =====================================================
                    if config.use_spectral > 0
                        spec_train_raw = extract_spectral_features(X_train_ep, cfg.fs);
                        spec_test_raw  = extract_spectral_features(X_test_ep, cfg.fs);

                        spc_train = spec_train_raw.psd;
                        spc_test  = spec_test_raw.psd;

                        if config.use_spectral == 2 % PSD + Mu/Beta + Entropy
                            spc_train = [spc_train, spec_train_raw.mu, spec_train_raw.beta, spec_train_raw.entropy];
                            spc_test  = [spc_test,  spec_test_raw.mu,  spec_test_raw.beta,  spec_test_raw.entropy];
                        end

                        feat_train = [feat_train, spc_train];
                        feat_test  = [feat_test,  spc_test];
                    end

                    % =====================================================
                    % 3. TEMPORAL BRANCH (Hjorth, RMS, WL, MAV, AR)
                    % =====================================================
                    if config.use_temporal > 0
                        tmp_train = extract_temporal_features(X_train_ep, cfg.fs, 4);
                        tmp_test  = extract_temporal_features(X_test_ep, cfg.fs, 4);

                        feat_train = [feat_train, tmp_train];
                        feat_test  = [feat_test,  tmp_test];
                    end

                    % =====================================================
                    % 4. GEOMETRIC BRANCH (Riemannian)
                    % =====================================================
                    if config.use_riemannian > 0
                        % Train fitting
                        [rm_train, ref_matrix] = extract_riemannian_features(X_train_ep);
                        % Test projection
                        rm_test = extract_riemannian_features(X_test_ep, ref_matrix);

                        feat_train = [feat_train, rm_train];
                        feat_test  = [feat_test,  rm_test];
                    end

                    feature_dim_pre = size(feat_train, 2);

                    % =====================================================
                    % NORMALIZATION (TRAIN STATS ONLY)
                    % =====================================================
                    [feat_train, feat_test, ~] = normalize_features( ...
                        feat_train, feat_test, config.norm_mode);

                    % =====================================================
                    % FEATURE SELECTION (TRAIN ONLY)
                    % =====================================================
                    if config.fs_k > 0 && config.fs_k < size(feat_train, 2)
                        switch string(config.fs_method)
                            case "anova"
                                [feat_train, feat_test, ~, ~] = select_features_anova( ...
                                    feat_train, y_train, feat_test, config.fs_k);
                            case "mi"
                                [feat_train, feat_test, ~, ~] = select_features_mi( ...
                                    feat_train, y_train, feat_test, config.fs_k);
                            case "fisher"
                                [feat_train, feat_test, ~, ~] = select_features_fisher( ...
                                    feat_train, y_train, feat_test, config.fs_k);
                        end
                    end

                    feature_dim_post = size(feat_train, 2);

                    % =====================================================
                    % CLASSIFICATION (CatBoost)
                    % =====================================================
                    y_pred = classify_catboost( ...
                        feat_train, y_train, feat_test, cfg);

                    % --- Fold metrics ------------------------------------
                    fold_m = compute_fold_metrics(y_test, y_pred);

                    fold_accuracies(fold_idx) = fold_m.accuracy;
                    fold_f1s(fold_idx)        = fold_m.f1;
                end

                mean_acc = mean(fold_accuracies);
                std_acc  = std(fold_accuracies);
                mean_f1  = mean(fold_f1s);

                subject_accuracies(end + 1) = mean_acc; %#ok<AGROW>
                subject_f1s(end + 1)        = mean_f1;  %#ok<AGROW>

                subject_runtime = toc(subject_tic);

                row = build_result_row(config, subject_id, subject_name, ...
                    'OK', mean_acc, std_acc, mean_f1, ...
                    subject_info.n_trials, feature_dim_pre, feature_dim_post, ...
                    subject_runtime, '');
                all_results = [all_results; row]; %#ok<AGROW>

                fprintf('  %s: Acc=%.2f%% +/- %.2f%% | F1=%.4f | Dim: %d->%d | %.1fs\n', ...
                    subject_name, mean_acc * 100, std_acc * 100, mean_f1, ...
                    feature_dim_pre, feature_dim_post, subject_runtime);

            catch ME
                subject_runtime = toc(subject_tic);
                row = build_result_row(config, subject_id, subject_name, ...
                    'ERROR', NaN, NaN, NaN, 0, 0, 0, subject_runtime, ...
                    regexprep(ME.message, '[\r\n]+', ' '));
                all_results = [all_results; row]; %#ok<AGROW>

                log_experiment(log_file, sprintf('  %s ERROR: %s', ...
                    subject_name, ME.message));
                fprintf('  %s: ERROR: %s\n', subject_name, ...
                    truncate_message(ME.message, 80));
            end

            % Periodic save
            if mod(subj_idx, 5) == 0 || subj_idx == numel(subject_ids)
                try
                    writetable(all_results, summary_csv);
                catch
                end
            end
        end

        config_runtime = toc(config_tic);

        if ~isempty(subject_accuracies)
            fprintf('\n  Config %s: Mean Acc = %.2f%% +/- %.2f%% | F1 = %.4f | %.1fs\n', ...
                config.id, ...
                mean(subject_accuracies) * 100, std(subject_accuracies) * 100, ...
                mean(subject_f1s), config_runtime);

            log_experiment(log_file, sprintf( ...
                'Config %s done: Acc=%.4f +/- %.4f F1=%.4f (%d subjects, %.1fs)', ...
                config.id, mean(subject_accuracies), std(subject_accuracies), ...
                mean(subject_f1s), numel(subject_accuracies), config_runtime));
        end
    end

    total_runtime = toc(sweep_tic);

    % ----- Final save ----------------------------------------------------
    try
        writetable(all_results, summary_csv);
        save(summary_mat, 'all_results', 'configs');
    catch ME_save
        fprintf('Warning: could not save results: %s\n', ME_save.message);
    end

    % ----- Generate and print ranked summary -----------------------------
    results = generate_ranked_summary( ...
        all_results, configs, total_runtime, summary_csv, log_file);

    log_experiment(log_file, sprintf( ...
        'Sweep complete. Total runtime: %.1f sec. Results: %s', ...
        total_runtime, summary_csv));

    fprintf('\n================================================================\n');
    fprintf('  Sweep Complete | Total runtime: %.1f sec\n', total_runtime);
    fprintf('  Results: %s\n', summary_csv);
    fprintf('================================================================\n');
end


%% ========================================================================
%  DEFAULT CONFIGURATION
%  ========================================================================

function cfg = default_sweep_config()
    cfg = struct();
    cfg.seed       = 42;
    cfg.fs         = 160;
    cfg.cv_folds   = 5;

    % CSP
    cfg.csp_components = 16;
    cfg.csp_beta       = 0.15;

    % Preprocessing
    cfg.preprocessing_bandpass = [4 32];
    cfg.butterworth_order      = 4;

    % Dataset
    cfg.run_ids             = [4 8 12];
    cfg.min_balanced_trials = 10;

    project_root  = fileparts(fileparts(mfilename('fullpath')));
    cfg.cache_dir = fullfile(project_root, 'physionet_cache');

    % CatBoost
    cfg.catboost.iterations    = 200;
    cfg.catboost.depth         = 6;
    cfg.catboost.learning_rate = 0.08;
end


%% ========================================================================
%  ABLATION CONFIGURATION GRID (15 Experiments)
%  ========================================================================

function configs = build_ablation_configs()
    n = 0;
    win = [0.5 4.0];
    norm_m = 'zscore';

    % E1: CSP only
    n = n + 1;
    configs(n) = make_config('E1', 1, 0, 0, 0, 'none', 0, norm_m, win, 'CSP Only');

    % E2: CSP + PSD
    n = n + 1;
    configs(n) = make_config('E2', 1, 1, 0, 0, 'none', 0, norm_m, win, 'CSP + PSD');

    % E3: CSP + PSD + Mu/Beta
    n = n + 1;
    configs(n) = make_config('E3', 1, 2, 0, 0, 'none', 0, norm_m, win, 'CSP + PSD + Mu/Beta');

    % E4: CSP + PSD + Temporal
    n = n + 1;
    configs(n) = make_config('E4', 1, 1, 1, 0, 'none', 0, norm_m, win, 'CSP + PSD + Temporal');

    % E5: Riemannian only
    n = n + 1;
    configs(n) = make_config('E5', 0, 0, 0, 1, 'none', 0, norm_m, win, 'Riemannian Only');

    % E6: CSP + Riemannian
    n = n + 1;
    configs(n) = make_config('E6', 1, 0, 0, 1, 'none', 0, norm_m, win, 'CSP + Riemannian');

    % E7: Full Fusion (CSP + PSD + Temporal + Riemannian)
    n = n + 1;
    configs(n) = make_config('E7', 1, 1, 1, 1, 'none', 0, norm_m, win, 'Full Fusion (CSP+PSD+Temp+Riem)');

    % E8: Full Fusion + ANOVA
    n = n + 1;
    configs(n) = make_config('E8', 1, 1, 1, 1, 'anova', 32, norm_m, win, 'Full Fusion + ANOVA (k=32)');

    % E9: Full Fusion + MI
    n = n + 1;
    configs(n) = make_config('E9', 1, 1, 1, 1, 'mi', 32, norm_m, win, 'Full Fusion + MI (k=32)');

    % E10: FBCSP + PSD + Temporal + Riemannian
    n = n + 1;
    configs(n) = make_config('E10', 2, 1, 1, 1, 'none', 0, norm_m, win, 'FBCSP + PSD + Temp + Riem');

    % E11: CSP + Temporal
    n = n + 1;
    configs(n) = make_config('E11', 1, 0, 1, 0, 'none', 0, norm_m, win, 'CSP + Temporal');

    % E12: CSP + Temporal + ANOVA
    n = n + 1;
    configs(n) = make_config('E12', 1, 0, 1, 0, 'anova', 32, norm_m, win, 'CSP + Temporal + ANOVA (k=32)');

    % E13: CSP + Temporal + MI
    n = n + 1;
    configs(n) = make_config('E13', 1, 0, 1, 0, 'mi', 32, norm_m, win, 'CSP + Temporal + MI (k=32)');

    % E14: CSP + Temporal + Fisher
    n = n + 1;
    configs(n) = make_config('E14', 1, 0, 1, 0, 'fisher', 32, norm_m, win, 'CSP + Temporal + Fisher (k=32)');

    % E15: Full Fusion + Fisher
    n = n + 1;
    configs(n) = make_config('E15', 1, 1, 1, 1, 'fisher', 32, norm_m, win, 'Full Fusion + Fisher (k=32)');
end

function cfg = make_config(id, sp, spec, temp, riem, fs_m, fs_k, norm_m, win, desc)
    cfg.id             = id;
    % Feature switches:
    % spatial: 0=none, 1=CSP, 2=FBCSP
    % spectral: 0=none, 1=PSD, 2=PSD+Mu/Beta/Ent
    % temporal: 0=none, 1=enabled
    % riemannian: 0=none, 1=enabled
    cfg.use_spatial    = sp;
    cfg.use_spectral   = spec;
    cfg.use_temporal   = temp;
    cfg.use_riemannian = riem;
    cfg.fs_method      = fs_m;
    cfg.fs_k           = fs_k;
    cfg.norm_mode      = norm_m;
    cfg.mi_window      = win;
    cfg.description    = desc;
end

%% ========================================================================
%  SUBJECT ID RESOLUTION
%  ========================================================================

function ids = resolve_subject_ids(mode)
    mode = lower(string(mode));
    switch mode
        case "smoke"
            ids = 1:3;
        case "pilot"
            ids = 1:10;
        case "full"
            ids = 1:109;
        otherwise
            tokens = regexp(char(mode), '[,; ]+', 'split');
            ids = [];
            for i = 1:numel(tokens)
                v = str2double(tokens{i});
                if ~isnan(v) && v >= 1 && v <= 109
                    ids(end + 1) = floor(v); %#ok<AGROW>
                end
            end
            ids = unique(ids, 'stable');
            if isempty(ids)
                error('run_multidomain_sweep:unknownMode', ...
                    'Unknown mode: %s. Use ''smoke'', ''pilot'', ''full'', or numeric IDs.', mode);
            end
    end
end

%% ========================================================================
%  LOCAL HELPERS
%  ========================================================================

function X_f = filter_epochs_local(X, b, a)
    X_f = zeros(size(X));
    for e = 1:size(X, 3)
        X_f(:, :, e) = filtfilt(b, a, X(:, :, e)')';
    end
end

function Z = project_csp_local(X, W)
    n_comp   = size(W, 2);
    n_time   = size(X, 2);
    n_epochs = size(X, 3);
    Z = zeros(n_comp, n_time, n_epochs);
    for e = 1:n_epochs
        Z(:, :, e) = W' * X(:, :, e);
    end
end

function features = extract_logvar_local(Z)
    n_epochs = size(Z, 3);
    n_comp   = size(Z, 1);
    features = zeros(n_epochs, n_comp);
    for e = 1:n_epochs
        v = var(Z(:, :, e), 0, 2);
        total_v = sum(v) + eps;
        features(e, :) = log(v / total_v + eps)';
    end
end

%% ========================================================================
%  RESULT TABLE CONSTRUCTION
%  ========================================================================

function row = build_result_row(config, subject_id, subject_name, ...
        status, mean_accuracy, std_accuracy, mean_f1, ...
        n_trials, dim_before, dim_after, runtime_sec, error_msg)
    row = table( ...
        string(config.id), ...
        config.use_spatial, ...
        config.use_spectral, ...
        config.use_temporal, ...
        config.use_riemannian, ...
        string(config.fs_method), ...
        config.fs_k, ...
        string(config.norm_mode), ...
        string(sprintf('[%.1f %.1f]', config.mi_window)), ...
        string(config.description), ...
        subject_id, ...
        string(subject_name), ...
        string(status), ...
        mean_accuracy, ...
        std_accuracy, ...
        mean_f1, ...
        n_trials, ...
        dim_before, ...
        dim_after, ...
        runtime_sec, ...
        string(error_msg), ...
        'VariableNames', { ...
            'config_id', 'use_spatial', 'use_spectral', 'use_temporal', 'use_riemannian', ...
            'fs_method', 'fs_k', 'norm_mode', 'mi_window', 'description', ...
            'subject_id', 'subject_name', 'status', ...
            'mean_accuracy', 'std_accuracy', 'mean_f1', ...
            'n_trials', 'feature_dim_before', 'feature_dim_after', ...
            'runtime_sec', 'error_message'});
end

%% ========================================================================
%  RANKED SUMMARY GENERATION
%  ========================================================================

function results = generate_ranked_summary(all_results, configs, ...
        total_runtime, summary_csv, log_file)

    results = struct();
    results.all_results       = all_results;
    results.total_runtime_sec = total_runtime;
    results.summary_csv       = summary_csv;

    if isempty(all_results)
        fprintf('\nNo experiments completed.\n');
        return;
    end

    ok_rows = all_results(all_results.status == "OK", :);

    if isempty(ok_rows)
        fprintf('\nNo successful experiments.\n');
        return;
    end

    config_ids    = unique(ok_rows.config_id, 'stable');
    summary_table = table();

    for i = 1:numel(config_ids)
        cid    = config_ids(i);
        mask   = ok_rows.config_id == cid;
        subset = ok_rows(mask, :);

        row = table( ...
            cid, ...
            subset.use_spatial(1), ...
            subset.use_spectral(1), ...
            subset.use_temporal(1), ...
            subset.use_riemannian(1), ...
            subset.fs_method(1), ...
            subset.fs_k(1), ...
            mean(subset.mean_accuracy, 'omitnan'), ...
            std(subset.mean_accuracy,  'omitnan'), ...
            mean(subset.mean_f1,       'omitnan'), ...
            round(mean(subset.feature_dim_after, 'omitnan')), ...
            height(subset), ...
            sum(subset.runtime_sec, 'omitnan'), ...
            'VariableNames', { ...
                'config_id', 'use_spatial', 'use_spectral', 'use_temporal', 'use_riemannian', ...
                'fs_method', 'fs_k', ...
                'mean_accuracy', 'std_accuracy', 'mean_f1', ...
                'feature_dim', 'n_subjects', 'total_runtime_sec'});

        summary_table = [summary_table; row]; %#ok<AGROW>
    end

    summary_table = sortrows(summary_table, 'mean_accuracy', 'descend');

    fprintf('\n================================================================\n');
    fprintf('  RANKED CONFIGURATION SUMMARY\n');
    fprintf('================================================================\n');
    fprintf('%-5s %-4s %-22s %-6s %-4s %8s %8s %5s\n', ...
        'Rank', 'ID', 'Features (Sp/Spc/Te/Rm)', 'FS', 'k', ...
        'MeanAcc', 'StdAcc', 'Dim');
    fprintf('%s\n', repmat('-', 1, 75));

    for r = 1:height(summary_table)
        feat_str = sprintf('%d/%d/%d/%d', ...
            summary_table.use_spatial(r), summary_table.use_spectral(r), ...
            summary_table.use_temporal(r), summary_table.use_riemannian(r));
        fprintf('%-5d %-4s %-22s %-6s %-4d %7.2f%% %7.2f%% %5d\n', ...
            r, ...
            char(summary_table.config_id(r)), ...
            feat_str, ...
            char(summary_table.fs_method(r)), ...
            summary_table.fs_k(r), ...
            summary_table.mean_accuracy(r) * 100, ...
            summary_table.std_accuracy(r) * 100, ...
            summary_table.feature_dim(r));
    end

    baseline_mask = summary_table.config_id == "E1";
    if any(baseline_mask)
        baseline_acc = summary_table.mean_accuracy(baseline_mask);
        fprintf('\n  Baseline (E1): %.2f%%\n', baseline_acc * 100);

        for r = 1:height(summary_table)
            if summary_table.config_id(r) == "E1"
                continue;
            end
            delta = (summary_table.mean_accuracy(r) - baseline_acc) * 100;
            if delta >= 0
                sign_str = '+';
            else
                sign_str = '';
            end
            fprintf('  %s vs baseline: %s%.2f pp\n', ...
                char(summary_table.config_id(r)), sign_str, delta);
        end

        best_acc = summary_table.mean_accuracy(1);
        best_id  = summary_table.config_id(1);
        delta_pp = (best_acc - baseline_acc) * 100;
        log_experiment(log_file, sprintf( ...
            'Best config: %s (%.2f%%) vs baseline E1 (%.2f%%): %+.2f pp', ...
            best_id, best_acc * 100, baseline_acc * 100, delta_pp));
    end

    results.summary_table = summary_table;
    if height(summary_table) > 0
        results.best_config = summary_table(1, :);
    end
end

function cfg = merge_sweep_config(base, override)
    cfg    = base;
    fields = fieldnames(override);
    for k = 1:numel(fields)
        f = fields{k};
        if isstruct(override.(f)) && isfield(cfg, f) && isstruct(cfg.(f))
            cfg.(f) = merge_sweep_config(cfg.(f), override.(f));
        else
            cfg.(f) = override.(f);
        end
    end
end

function msg = truncate_message(msg, max_len)
    msg = regexprep(msg, '[\r\n]+', ' ');
    if numel(msg) > max_len
        msg = [msg(1:max_len - 3) '...'];
    end
end
