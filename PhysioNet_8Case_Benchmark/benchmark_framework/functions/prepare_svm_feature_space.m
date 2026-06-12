function [train_X_out, test_X_out, info] = prepare_svm_feature_space(train_X, test_X, cfg)
train_X_out = train_X;
test_X_out = test_X;

info = struct();
info.original_feature_count = size(train_X, 2);
info.zero_variance_removed = 0;
info.duplicate_removed = 0;
info.auto_pca_applied = false;
info.final_feature_count = size(train_X, 2);

if isempty(train_X_out)
    return;
end

variance_mask = var(train_X_out, 0, 1) > 0;
info.zero_variance_removed = size(train_X_out, 2) - sum(variance_mask);

if any(variance_mask)
    train_X_out = train_X_out(:, variance_mask);
    test_X_out = test_X_out(:, variance_mask);
else
    train_X_out = zeros(size(train_X_out, 1), 1);
    test_X_out = zeros(size(test_X_out, 1), 1);
    info.final_feature_count = 1;
    return;
end

try
    [~, unique_idx] = unique(train_X_out', 'rows', 'stable');
    unique_idx = sort(unique_idx);
    info.duplicate_removed = size(train_X_out, 2) - numel(unique_idx);
    train_X_out = train_X_out(:, unique_idx);
    test_X_out = test_X_out(:, unique_idx);
catch
    info.duplicate_removed = 0;
end

if size(train_X_out, 2) > size(train_X_out, 1) && ...
        size(train_X_out, 2) > 1 && size(train_X_out, 1) > 1
    [coeff, score, ~, ~, ~, mu] = pca(train_X_out);
    n_components = max(1, min(size(train_X_out, 1) - 1, size(score, 2)));
    if isfield(cfg, 'runtime') && isfield(cfg.runtime, 'fast_debug_mode') && ...
            cfg.runtime.fast_debug_mode
        n_components = min(n_components, 12);
    end
    train_X_out = score(:, 1:n_components);
    test_X_out = (test_X_out - mu) * coeff(:, 1:n_components);
    info.auto_pca_applied = true;
end

info.final_feature_count = size(train_X_out, 2);
end
