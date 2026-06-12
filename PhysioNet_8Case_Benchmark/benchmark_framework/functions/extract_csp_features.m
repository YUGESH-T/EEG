function feature_bundle = extract_csp_features(X_left, X_right, split, cfg)
X_all = cat(3, X_left, X_right);
labels = [ones(size(X_left, 3), 1); 2 * ones(size(X_right, 3), 1)];

train_left = X_all(:, :, split.train_idx & labels == 1);
train_right = X_all(:, :, split.train_idx & labels == 2);
[filters, ~] = compute_csp_filters(train_left, train_right);
n_filters = size(filters, 1);
n_components = 4;
if isfield(cfg, 'csp_components') && ~isempty(cfg.csp_components)
    n_components = cfg.csp_components;
end
n_components = max(2, min(n_filters, round(n_components)));
n_pairs = max(1, floor(n_components / 2));
selected_filter_idx = unique(max(1, [1:n_pairs, (n_filters - n_pairs + 1):n_filters]));
W = filters(selected_filter_idx, :);

features = zeros(size(X_all, 3), numel(selected_filter_idx));
for i = 1:size(X_all, 3)
    projected = W * squeeze(X_all(:, :, i));
    var_features = var(projected, 0, 2);
    features(i, :) = log(var_features ./ sum(var_features));
end

feature_bundle = struct();
feature_bundle.train_X = features(split.train_idx, :);
feature_bundle.test_X = features(split.test_idx, :);
feature_bundle.train_Y = labels(split.train_idx);
feature_bundle.test_Y = labels(split.test_idx);
feature_bundle.selected_filter_idx = selected_filter_idx;
end
