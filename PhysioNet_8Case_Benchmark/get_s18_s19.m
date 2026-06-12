addpath(fullfile(pwd, 'benchmark_framework', 'functions'));
cfg = build_default_config();
cfg.workspace_dir = pwd;
case_cfgs = build_case_configs(cfg);
case4_cfg = case_cfgs(4);

% Setup temporary output folders so save commands don't crash
case4_cfg.case_root = fullfile(pwd, 'temp_run_case04');
ensure_dir(fullfile(case4_cfg.case_root, 'curves'));
ensure_dir(fullfile(case4_cfg.case_root, 'mat'));

disp('--------------------------------------------------');
disp('Processing Subject 18 for Case 04 (Riemannian+RF+Common)...');
res18 = run_subject_case(18, case4_cfg);
fprintf('Subject 18 -> Acc: %.3f | F1: %.3f\n', res18.Accuracy, res18.F1);
disp(res18);

disp('--------------------------------------------------');
disp('Processing Subject 19 for Case 04 (Riemannian+RF+Common)...');
res19 = run_subject_case(19, case4_cfg);
fprintf('Subject 19 -> Acc: %.3f | F1: %.3f\n', res19.Accuracy, res19.F1);
disp(res19);
disp('--------------------------------------------------');
