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
cfg.fs = 160;
cfg.band = [8 30];
cfg.trial_duration_sec = 4;
cfg.mi_window_sec = [0 4];
cfg.csp_components = 4;
cfg.channel_mode = 'all';
cfg.spectral = struct('bands', [4 8; 8 12; 12 16; 16 20; 20 24; 24 30]);
cfg.fbcsp_mi = struct( ...
    'bands', [4 8; 8 12; 12 16; 16 20; 20 24; 24 30], ...
    'top_k', 20, ...
    'mi_bins', 8, ...
    'filter_order', 4);
cfg.temporal = struct( ...
    'sample_entropy_m', 2, ...
    'sample_entropy_r', 0.2, ...
    'sample_entropy_stride', 8, ...
    'sample_entropy_max_points', 80);
cfg.run_ids = [4 8 12];
cfg.min_balanced_trials = 10;
cfg.common_filter_order = 4;
cfg.butterworth_order = 4;
cfg.ho = struct('agents', 10, 'iterations', 30, 'threshold', 0.5);
cfg.eval = struct('holdout', 0.2, 'ho_folds', 5, 'positive_label', 2);
cfg.classifier = struct('svm_kernel', 'linear', 'rf_cycles', 100, 'rf_min_leaf', 1);
cfg.riemannian = struct('epsilon', 1e-6, 'top_k', 0);
cfg.runtime = struct('enable_intermediate_cache', true, ...
    'fast_debug_mode', false);
end
