function feature_bundle = extract_spatial_spectral_features(X_left, X_right, split, cfg)
csp_bundle = extract_csp_features(X_left, X_right, split, cfg);
X_all = cat(3, X_left, X_right);
spectral_features = extract_spectral_features(X_all, cfg);

feature_bundle = struct();
feature_bundle.train_X = [csp_bundle.train_X, spectral_features(split.train_idx, :)];
feature_bundle.test_X = [csp_bundle.test_X, spectral_features(split.test_idx, :)];
feature_bundle.train_Y = csp_bundle.train_Y;
feature_bundle.test_Y = csp_bundle.test_Y;
feature_bundle.selected_filter_idx = csp_bundle.selected_filter_idx;
feature_bundle.feature_blocks = {'csp', 'spectral'};
feature_bundle.spectral_feature_count = size(spectral_features, 2);
end
