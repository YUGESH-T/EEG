function feature_bundle = extract_riemannian_features(X_left, X_right, split, cfg)
n_trials = size(X_left, 3);
n_channels = size(X_left, 1);
epsilon = cfg.riemannian.epsilon;
diag_mask = triu(true(n_channels), 0);
offdiag_mask = triu(true(n_channels), 1);

X_all = cat(3, X_left, X_right);
labels = [ones(n_trials, 1); 2 * ones(n_trials, 1)];
num_total = size(X_all, 3);

cov_matrices = zeros(n_channels, n_channels, num_total);
for i = 1:num_total
    trial = squeeze(X_all(:, :, i));
    cov_matrices(:, :, i) = (trial * trial') / trace(trial * trial') + epsilon * eye(n_channels);
end

train_cov = cov_matrices(:, :, split.train_idx);
mean_cov = riemannian_mean_spd(train_cov);
P_half = sqrtm(mean_cov);
P_half_inv = P_half \ eye(size(P_half));

n_features = sum(diag_mask(:)) + sum(offdiag_mask(:));
features = zeros(num_total, n_features);
for i = 1:num_total
    tangent_matrix = logm(P_half_inv * cov_matrices(:, :, i) * P_half_inv);
    diag_terms = tangent_matrix(diag_mask);
    offdiag_terms = sqrt(2) * tangent_matrix(offdiag_mask);
    features(i, :) = [diag_terms; offdiag_terms].';
end

feature_bundle = struct();
feature_bundle.train_X = features(split.train_idx, :);
feature_bundle.test_X = features(split.test_idx, :);
feature_bundle.train_Y = labels(split.train_idx);
feature_bundle.test_Y = labels(split.test_idx);
end
