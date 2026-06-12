function results = run_validation_sweep(mode, cfg_override)
%RUN_VALIDATION_SWEEP PhysioNet EEGMMIDB validation experiment suite.
%
%   Runs controlled ablation experiments for motor imagery classification
%   across configurations testing:
%     - Baseline Verification
%     - Fine Window Sweep
%     - Shrinkage CSP + Best Window
%     - Compact ANOVA Verification
%
%   Usage:
%     results = run_validation_sweep('smoke');   % 3 subjects, quick validation
%     results = run_validation_sweep('pilot');   % 10 subjects
%     results = run_validation_sweep('full');    % 109 subjects (hours)
%
%   All CSP, normalization, and ANOVA fitting uses training-fold data ONLY.
%   No validated baseline files are modified or imported.

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
        'Optimization sweep started. Mode: %s | Subjects: %d | Configs: %d', ...
        mode, numel(subject_ids), numel(configs)));

    fprintf('\n================================================================\n');
    fprintf('  PhysioNet EEGMMIDB Validation Sweep V2\n');
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
        fprintf('  CSP: %s | Spectral: %s | ANOVA k: %d | Norm: %s | Window: [%.1f, %.1f]\n', ...
            config.csp_type, config.spectral_mode, config.anova_k, ...
            config.norm_mode, config.mi_window(1), config.mi_window(2));

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

                    % --- CSP bandpass [8, 30] Hz -------------------------
                    [b_csp, a_csp] = butter(4, [8 30] / (cfg.fs / 2), 'bandpass');
                    X_train_f = filter_epochs_local(X_train_ep, b_csp, a_csp);
                    X_test_f  = filter_epochs_local(X_test_ep,  b_csp, a_csp);

                    % --- Compute CSP filters (TRAIN ONLY) ----------------
                    [W, ~] = compute_csp_projection( ...
                        X_train_f, y_train, ...
                        config.csp_components, config.csp_type, cfg.csp_beta);

                    % --- Project through CSP filters ---------------------
                    % Filtered projections → CSP log-variance features
                    Z_train_f = project_csp_local(X_train_f, W);
                    Z_test_f  = project_csp_local(X_test_f,  W);

                    % Broadband projections → spectral features
                    Z_train_b = project_csp_local(X_train_ep, W);
                    Z_test_b  = project_csp_local(X_test_ep,  W);

                    % --- CSP log-variance features -----------------------
                    csp_train = extract_logvar_local(Z_train_f);
                    csp_test  = extract_logvar_local(Z_test_f);

                    % --- Spectral features from broadband projection -----
                    spec_train = extract_spectral_features_v2( ...
                        Z_train_b, cfg.fs, config.spectral_mode);
                    spec_test  = extract_spectral_features_v2( ...
                        Z_test_b,  cfg.fs, config.spectral_mode);

                    % --- Concatenate feature blocks ----------------------
                    feat_train = [csp_train, spec_train];
                    feat_test  = [csp_test,  spec_test];

                    feature_dim_pre = size(feat_train, 2);

                    % --- Normalize (TRAIN STATS ONLY) --------------------
                    [feat_train, feat_test, ~] = normalize_features( ...
                        feat_train, feat_test, config.norm_mode);

                    % --- ANOVA feature selection (TRAIN ONLY) ------------
                    if config.anova_k > 0 && config.anova_k < size(feat_train, 2)
                        [feat_train, feat_test, ~, ~] = select_features_anova( ...
                            feat_train, y_train, feat_test, config.anova_k);
                    end

                    feature_dim_post = size(feat_train, 2);

                    % --- CatBoost classification -------------------------
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

            % Periodic save (every 5 subjects or at end)
            if mod(subj_idx, 5) == 0 || subj_idx == numel(subject_ids)
                try
                    writetable(all_results, summary_csv);
                catch
                    % Non-fatal: log but continue
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
%  ABLATION CONFIGURATION GRID
%  ========================================================================

function configs = build_ablation_configs()
    n = 0;

    % --- Historical Baseline Verification -------------------------------
    n = n + 1;
    configs(n) = make_config('Hist', 'standard', 'none', 0, 'zscore', [0.5 3.5], 14, ...
        'Historical Baseline: standard CSP + no spectral + zscore + 0.5-3.5s + 14 CSP');

    % --- Baseline Verification ------------------------------------------
    n = n + 1;
    configs(n) = make_config('Base', 'standard', '3band', 0, 'zscore', [0 4], 16, ...
        'Baseline: standard CSP + 3-band spectral + zscore + 0-4s');

    % --- Task 1: Fine Window Sweep --------------------------------------
    n = n + 1;
    configs(n) = make_config('W1', 'standard', '3band', 0, 'zscore', [0.75 4.0], 16, ...
        'Window Sweep: 0.75-4.0s');
        
    n = n + 1;
    configs(n) = make_config('W2', 'standard', '3band', 0, 'zscore', [1.0 4.0], 16, ...
        'Window Sweep: 1.0-4.0s');
        
    n = n + 1;
    configs(n) = make_config('W3', 'standard', '3band', 0, 'zscore', [1.0 4.5], 16, ...
        'Window Sweep: 1.0-4.5s');
        
    n = n + 1;
    configs(n) = make_config('W4', 'standard', '3band', 0, 'zscore', [1.25 4.0], 16, ...
        'Window Sweep: 1.25-4.0s');
        
    n = n + 1;
    configs(n) = make_config('W5', 'standard', '3band', 0, 'zscore', [1.25 4.5], 16, ...
        'Window Sweep: 1.25-4.5s');

    % --- Task 2: Shrinkage CSP + Best Window ----------------------------
    n = n + 1;
    configs(n) = make_config('S1', 'shrinkage', '3band', 0, 'zscore', [1.0 4.0], 16, ...
        'Shrinkage CSP + Window 1.0-4.0s');
        
    n = n + 1;
    configs(n) = make_config('S2', 'shrinkage', '3band', 0, 'zscore', [1.0 4.5], 16, ...
        'Shrinkage CSP + Window 1.0-4.5s');
        
    n = n + 1;
    configs(n) = make_config('S3', 'shrinkage', '3band', 0, 'zscore', [1.25 4.0], 16, ...
        'Shrinkage CSP + Window 1.25-4.0s');

    % --- Task 4: Compact ANOVA Verification -----------------------------
    n = n + 1;
    configs(n) = make_config('A1_16', 'standard', '3band', 16, 'zscore', [1.0 4.0], 16, ...
        'ANOVA Top-16 + Window 1.0-4.0s');
        
    n = n + 1;
    configs(n) = make_config('A1_32', 'standard', '3band', 32, 'zscore', [1.0 4.0], 16, ...
        'ANOVA Top-32 + Window 1.0-4.0s');

    n = n + 1;
    configs(n) = make_config('A2_16', 'standard', '3band', 16, 'zscore', [1.0 4.5], 16, ...
        'ANOVA Top-16 + Window 1.0-4.5s');
        
    n = n + 1;
    configs(n) = make_config('A2_32', 'standard', '3band', 32, 'zscore', [1.0 4.5], 16, ...
        'ANOVA Top-32 + Window 1.0-4.5s');

    n = n + 1;
    configs(n) = make_config('A3_16', 'standard', '3band', 16, 'zscore', [1.25 4.0], 16, ...
        'ANOVA Top-16 + Window 1.25-4.0s');
        
    n = n + 1;
    configs(n) = make_config('A3_32', 'standard', '3band', 32, 'zscore', [1.25 4.0], 16, ...
        'ANOVA Top-32 + Window 1.25-4.0s');
end


function cfg = make_config(id, csp_type, spectral_mode, anova_k, ...
        norm_mode, mi_window, csp_components, description)
    cfg.id             = id;
    cfg.csp_type       = csp_type;
    cfg.spectral_mode  = spectral_mode;
    cfg.anova_k        = anova_k;
    cfg.norm_mode      = norm_mode;
    cfg.mi_window      = mi_window;
    cfg.csp_components = csp_components;
    cfg.description    = description;
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
            % Parse comma/semicolon/space separated numeric list
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
                error('run_validation_sweep:unknownMode', ...
                    'Unknown mode: %s. Use ''smoke'', ''pilot'', ''full'', or numeric IDs.', mode);
            end
    end
end


%% ========================================================================
%  PER-FOLD HELPER FUNCTIONS
%  ========================================================================

function X_f = filter_epochs_local(X, b, a)
%FILTER_EPOCHS_LOCAL Apply filter to each epoch (channels × time × epochs).
    X_f = zeros(size(X));
    for e = 1:size(X, 3)
        X_f(:, :, e) = filtfilt(b, a, X(:, :, e)')';
    end
end


function Z = project_csp_local(X, W)
%PROJECT_CSP_LOCAL Project epochs through CSP spatial filters.
%   X: [n_ch × n_time × n_epochs], W: [n_ch × n_comp]
%   Z: [n_comp × n_time × n_epochs]
    n_comp   = size(W, 2);
    n_time   = size(X, 2);
    n_epochs = size(X, 3);
    Z = zeros(n_comp, n_time, n_epochs);
    for e = 1:n_epochs
        Z(:, :, e) = W' * X(:, :, e);
    end
end


function features = extract_logvar_local(Z)
%EXTRACT_LOGVAR_LOCAL CSP log-normalized-variance features.
%   Z: [n_comp × n_time × n_epochs]
%   features: [n_epochs × n_comp]
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
        string(config.csp_type), ...
        string(config.spectral_mode), ...
        config.anova_k, ...
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
            'config_id', 'csp_type', 'spectral_mode', 'anova_k', ...
            'norm_mode', 'mi_window', 'description', ...
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

    % ----- Build per-config summary table --------------------------------
    config_ids    = unique(ok_rows.config_id, 'stable');
    summary_table = table();

    for i = 1:numel(config_ids)
        cid    = config_ids(i);
        mask   = ok_rows.config_id == cid;
        subset = ok_rows(mask, :);

        row = table( ...
            cid, ...
            subset.csp_type(1), ...
            subset.spectral_mode(1), ...
            subset.anova_k(1), ...
            subset.norm_mode(1), ...
            subset.mi_window(1), ...
            mean(subset.mean_accuracy, 'omitnan'), ...
            std(subset.mean_accuracy,  'omitnan'), ...
            mean(subset.mean_f1,       'omitnan'), ...
            round(mean(subset.feature_dim_after, 'omitnan')), ...
            height(subset), ...
            sum(subset.runtime_sec, 'omitnan'), ...
            'VariableNames', { ...
                'config_id', 'csp_type', 'spectral_mode', 'anova_k', ...
                'norm_mode', 'mi_window', ...
                'mean_accuracy', 'std_accuracy', 'mean_f1', ...
                'feature_dim', 'n_subjects', 'total_runtime_sec'});

        summary_table = [summary_table; row]; %#ok<AGROW>
    end

    summary_table = sortrows(summary_table, 'mean_accuracy', 'descend');

    % ----- Print ranked table --------------------------------------------
    fprintf('\n================================================================\n');
    fprintf('  RANKED CONFIGURATION SUMMARY\n');
    fprintf('================================================================\n');
    fprintf('%-5s %-6s %-10s %-6s %-5s %-8s %-10s %8s %8s %5s\n', ...
        'Rank', 'ID', 'CSP', 'Bands', 'AnoK', 'Norm', 'Window', ...
        'MeanAcc', 'StdAcc', 'Dim');
    fprintf('%s\n', repmat('-', 1, 82));

    for r = 1:height(summary_table)
        fprintf('%-5d %-6s %-10s %-6s %-5d %-8s %-10s %7.2f%% %7.2f%% %5d\n', ...
            r, ...
            char(summary_table.config_id(r)), ...
            char(summary_table.csp_type(r)), ...
            char(summary_table.spectral_mode(r)), ...
            summary_table.anova_k(r), ...
            char(summary_table.norm_mode(r)), ...
            char(summary_table.mi_window(r)), ...
            summary_table.mean_accuracy(r) * 100, ...
            summary_table.std_accuracy(r) * 100, ...
            summary_table.feature_dim(r));
    end

    % ----- Baseline comparison -------------------------------------------
    baseline_mask = summary_table.config_id == "Hist";
    if any(baseline_mask)
        baseline_acc = summary_table.mean_accuracy(baseline_mask);
        fprintf('\n  Historical Baseline (Hist): %.2f%%\n', baseline_acc * 100);

        for r = 1:height(summary_table)
            if summary_table.config_id(r) == "Hist"
                continue;
            end
            delta = (summary_table.mean_accuracy(r) - baseline_acc) * 100;
            if delta >= 0
                sign_str = '+';
            else
                sign_str = '';
            end
            fprintf('  %s vs Hist baseline: %s%.2f pp\n', ...
                char(summary_table.config_id(r)), sign_str, delta);
        end

        % Log best result comparison
        best_acc = summary_table.mean_accuracy(1);
        best_id  = summary_table.config_id(1);
        delta_pp = (best_acc - baseline_acc) * 100;
        log_experiment(log_file, sprintf( ...
            'Best config: %s (%.2f%%) vs Hist baseline (%.2f%%): %+.2f pp', ...
            best_id, best_acc * 100, baseline_acc * 100, delta_pp));
    end

    results.summary_table = summary_table;
    if height(summary_table) > 0
        results.best_config = summary_table(1, :);
    end
end


%% ========================================================================
%  UTILITIES
%  ========================================================================

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
