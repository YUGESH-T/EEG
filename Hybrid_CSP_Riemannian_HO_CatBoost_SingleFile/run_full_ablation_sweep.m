% run_full_ablation_sweep.m
% Execute the full-dataset ablation study for the isolated experimental pipeline

datasets = {'cho_full52', 'physionet109'};
csps = [14, 16];

% The 8 combinations
feature_configs = {
    'CSP only',                 struct('s', false, 't', false, 'h', false);
    'CSP + Spectral',           struct('s', true,  't', false, 'h', false);
    'CSP + Temporal',           struct('s', false, 't', true,  'h', false);
    'CSP + HO',                 struct('s', false, 't', false, 'h', true);
    'CSP + Spectral + Temporal',struct('s', true,  't', true,  'h', false);
    'CSP + Spectral + HO',      struct('s', true,  't', false, 'h', true);
    'CSP + Temporal + HO',      struct('s', false, 't', true,  'h', true);
    'CSP + Spectral + Temporal + HO', struct('s', true, 't', true, 'h', true)
};

% Results storage
results_table_data = [];
results_table_meta = {};

fprintf('=========================================================================\n');
fprintf('STARTING FULL ABLATION SWEEP (32 EXPERIMENTS)\n');
fprintf('=========================================================================\n\n');

for d_idx = 1:length(datasets)
    dataset = datasets{d_idx};
    
    for c_idx = 1:length(csps)
        csp_count = csps(c_idx);
        
        for f_idx = 1:size(feature_configs, 1)
            combo_name = feature_configs{f_idx, 1};
            f_flags = feature_configs{f_idx, 2};
            
            % Calculate Expected Feature Dimension
            feat_dim = csp_count; % base CSP
            if f_flags.s, feat_dim = feat_dim + (csp_count * 3); end
            if f_flags.t, feat_dim = feat_dim + (csp_count * 7); end
            if f_flags.h, feat_dim = feat_dim + (csp_count * 2); end
            
            fprintf('==================================================\n');
            fprintf('Dataset: %s\n', dataset);
            fprintf('CSP Components: %d\n', csp_count);
            fprintf('Active Features: %s\n', combo_name);
            fprintf('Expected Feature Dimension: %d\n', feat_dim);
            fprintf('===========================\n');
            
            % Setup configuration
            cfg = struct();
            cfg.results.resume = true; % safely continue interrupted experiments
            cfg.features.csp_components = csp_count;
            cfg.features.use_spectral = f_flags.s;
            cfg.features.use_temporal = f_flags.t;
            cfg.features.use_ho = f_flags.h;
            
            % We will override the output CSV/MAT inside the core to separate results
            % Or the core will automatically create files based on config.
            % We don't want the core plain_csp_catboostcopy to overwrite different combinations
            % if they share the same CSV file!
            % Wait, plain_csp_catboostcopy hardcodes CSV names like 'csp_catboost_physionet_results.csv'
            % If we use resume=true, different combinations might conflict in the same CSV!
            % To avoid conflicts, we MUST dynamically change the output_dir or csv_file name.
            % The prompt says: "Do NOT alter original CSP baselines... Do NOT modify plain_csp_catboostcopy.m"
            % So we can set cfg.results.output_dir to keep them separate.
            combo_folder = strrep(strrep(combo_name, ' + ', '_'), ' ', '_');
            cfg.results.output_dir = fullfile('results_ablation', sprintf('%s_csp%d', combo_folder, csp_count));
            
            mean_acc = NaN;
            std_acc = NaN;
            mean_f1 = NaN;
            
            t_start = tic;
            try
                res = plain_csp_catboostcopy(dataset, cfg);
                if isfield(res, 'summary')
                    mean_acc = res.summary.MeanAccuracy;
                    std_acc = res.summary.StdAccuracy;
                    mean_f1 = res.summary.MeanF1;
                end
            catch ME
                fprintf('ERROR: Experiment failed: %s\n', ME.message);
            end
            runtime = toc(t_start);
            
            fprintf('---\n');
            fprintf('Mean Accuracy: %.2f%%\n', mean_acc * 100);
            fprintf('Std Accuracy: %.2f%%\n', std_acc * 100);
            fprintf('Mean F1: %.4f\n', mean_f1);
            fprintf('Runtime: %.2f seconds\n', runtime);
            fprintf('--------\n\n');
            
            % Store results
            results_table_meta(end+1, :) = {dataset, combo_name};
            results_table_data(end+1, :) = [csp_count, f_flags.s, f_flags.t, f_flags.h, ...
                                            feat_dim, mean_acc, std_acc, mean_f1, runtime];
        end
    end
end

%% Collect into a structured MATLAB table
Dataset = string(results_table_meta(:, 1));
FeatureCombinationName = string(results_table_meta(:, 2));
CSPComponents = results_table_data(:, 1);
UseSpectral = logical(results_table_data(:, 2));
UseTemporal = logical(results_table_data(:, 3));
UseHO = logical(results_table_data(:, 4));
FeatureDimension = results_table_data(:, 5);
MeanAccuracy = results_table_data(:, 6);
StdAccuracy = results_table_data(:, 7);
MeanF1 = results_table_data(:, 8);
RuntimeSeconds = results_table_data(:, 9);

final_table = table(Dataset, CSPComponents, UseSpectral, UseTemporal, UseHO, ...
    FeatureCombinationName, FeatureDimension, MeanAccuracy, StdAccuracy, MeanF1, RuntimeSeconds);

% Save to CSV and MAT
writetable(final_table, 'full_ablation_results.csv');
save('full_ablation_results.mat', 'final_table');

fprintf('\nSaved full_ablation_results.csv and full_ablation_results.mat\n');

%% FINAL SUMMARY TABLE
fprintf('\n=========================================================================\n');
fprintf('FINAL SUMMARY TABLE (Sorted by Mean Accuracy)\n');
fprintf('=========================================================================\n');

% Sort the table
sorted_table = sortrows(final_table, {'Dataset', 'MeanAccuracy'}, {'ascend', 'descend'});

fprintf('%-12s | %-4s | %-32s | %-8s | %-8s | %-8s\n', ...
    'Dataset', 'CSP', 'Feature Combination', 'FeatDim', 'MeanAcc', 'StdAcc');
fprintf('----------------------------------------------------------------------------------------\n');

for i = 1:height(sorted_table)
    r = sorted_table(i,:);
    fprintf('%-12s | %-4d | %-32s | %-8d | %5.2f%%   | %5.2f%%\n', ...
        r.Dataset, r.CSPComponents, r.FeatureCombinationName, ...
        r.FeatureDimension, r.MeanAccuracy * 100, r.StdAccuracy * 100);
end
fprintf('=========================================================================\n');

fprintf('\nAll 32 experiments finished.\n');
