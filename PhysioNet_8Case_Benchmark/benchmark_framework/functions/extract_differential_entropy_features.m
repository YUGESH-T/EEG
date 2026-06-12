function feature_bundle = extract_differential_entropy_features(X_left, X_right, split, cfg)
X_all = cat(3, X_left, X_right);
labels = [ones(size(X_left, 3), 1); 2 * ones(size(X_right, 3), 1)];

features = zeros(size(X_all, 3), size(X_all, 1));
for i = 1:size(X_all, 3)
    epoch = squeeze(X_all(:, :, i))';
    channel_var = var(epoch, 0, 1) + eps;
    features(i, :) = 0.5 * log(2 * pi * exp(1) * channel_var);
end

feature_bundle = struct();
feature_bundle.train_X = features(split.train_idx, :);
feature_bundle.test_X = features(split.test_idx, :);
feature_bundle.train_Y = labels(split.train_idx);
feature_bundle.test_Y = labels(split.test_idx);
feature_bundle.feature_type = 'differential_entropy';
end
