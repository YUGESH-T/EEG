function ho_result = run_ho_feature_selection(train_X, train_Y, case_cfg, subject_num)
candidate_idx = 1:size(train_X, 2);
X_ho = train_X;

if strcmpi(case_cfg.feature_type, 'riemannian') && case_cfg.riemannian.top_k > 0
    candidate_idx = anova_rank_features(train_X, train_Y, case_cfg.riemannian.top_k);
    X_ho = train_X(:, candidate_idx);
end

rng(case_cfg.seed + 1000 + subject_num, 'twister');
cv = cvpartition(train_Y, 'KFold', case_cfg.eval.ho_folds);
fitness_cache = containers.Map('KeyType', 'char', 'ValueType', 'double');
fitness_fn = @cached_mask_fitness;
[best_score, best_pos, ho_curve] = hippopotamus_optimization(size(X_ho, 2), fitness_fn, case_cfg);
mask = best_pos > case_cfg.ho.threshold;
if ~any(mask)
    mask = true(1, size(X_ho, 2));
end

selected_idx = candidate_idx(mask);
ho_result = struct();
ho_result.selected_idx = selected_idx;
ho_result.selected_count = numel(selected_idx);
ho_result.best_score = best_score;
ho_result.best_pos = best_pos;
ho_result.ho_curve = ho_curve;

    function error_val = cached_mask_fitness(position)
        mask_local = position > case_cfg.ho.threshold;
        if ~any(mask_local)
            error_val = 1;
            return;
        end

        mask_key = char(mask_local + '0');
        if isKey(fitness_cache, mask_key)
            error_val = fitness_cache(mask_key);
            return;
        end

        error_val = evaluate_mask_fitness(mask_local, X_ho, train_Y, ...
            case_cfg.classifier_type, case_cfg, subject_num, cv);
        fitness_cache(mask_key) = error_val;
    end
end
