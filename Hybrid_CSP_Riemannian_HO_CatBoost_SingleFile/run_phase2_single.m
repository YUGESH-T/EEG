% run_phase2_single.m
% Execute Phase 2 single-feature sweeps

modes = {'cho_pilot5', 'pilot10'};
csps = [14, 16];
feature_configs = {
    'Spectral', struct('use_spectral', true, 'use_temporal', false, 'use_ho', false),
    'Temporal', struct('use_spectral', false, 'use_temporal', true, 'use_ho', false),
    'HO',       struct('use_spectral', false, 'use_temporal', false, 'use_ho', true)
};

fprintf('=========================================================================\n');
fprintf('%-12s | %-4s | %-15s | %-8s | %-8s | %-8s\n', 'Dataset', 'CSP', 'Features', 'FeatDim', 'MeanAcc', 'StdAcc');
fprintf('=========================================================================\n');

% We will store results in an array to print a summary table at the end as well
results_table = {};

for m = 1:length(modes)
    mode = modes{m};
    
    for c = 1:length(csps)
        csp_val = csps(c);
        
        for f = 1:size(feature_configs, 1)
            f_name = feature_configs{f, 1};
            f_cfg = feature_configs{f, 2};
            
            cfg = struct();
            cfg.results.resume = false;
            cfg.features.csp_components = csp_val;
            cfg.features.use_spectral = f_cfg.use_spectral;
            cfg.features.use_temporal = f_cfg.use_temporal;
            cfg.features.use_ho = f_cfg.use_ho;
            
            % To avoid printing too much output, we can capture evalc if needed,
            % but let's just let it print and we'll summarize.
            res = plain_csp_catboostcopy(mode, cfg);
            
            % We can extract the mean accuracy from the summary table
            if isfield(res, 'summary')
                mean_acc = res.summary.MeanAccuracy;
                std_acc = res.summary.StdAccuracy;
            else
                mean_acc = NaN;
                std_acc = NaN;
            end
            
            % We don't have csp_feature_dim in the final summary directly, 
            % but we can grab it from the first row of the raw rows if needed, 
            % or we can just print the theoretical dimension.
            feat_dim = 0;
            if isfield(res, 'rows') && height(res.rows) > 0
                feat_dim = res.rows.csp_feature_dim(1);
            end
            
            results_table(end+1, :) = {mode, csp_val, ['CSP + ', f_name], feat_dim, mean_acc*100, std_acc*100};
        end
    end
end

% Print Summary Table
fprintf('\n\n');
fprintf('--- FINAL COMPARISON TABLE ---\n');
fprintf('=========================================================================\n');
fprintf('%-12s | %-4s | %-15s | %-8s | %-8s | %-8s\n', 'Dataset', 'CSP', 'Features', 'FeatDim', 'MeanAcc', 'StdAcc');
fprintf('=========================================================================\n');
for i = 1:size(results_table, 1)
    fprintf('%-12s | %-4d | %-15s | %-8d | %5.2f%%   | %5.2f%%\n', ...
        results_table{i, 1}, results_table{i, 2}, results_table{i, 3}, ...
        results_table{i, 4}, results_table{i, 5}, results_table{i, 6});
end
fprintf('=========================================================================\n');
