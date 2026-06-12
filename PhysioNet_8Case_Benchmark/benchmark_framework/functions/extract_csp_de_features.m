function feature_bundle = extract_csp_de_features(X_left, X_right, split, cfg)
csp_bundle = extract_csp_features(X_left, X_right, split, cfg);
de_bundle = extract_differential_entropy_features(X_left, X_right, split, cfg);

feature_bundle = struct();
feature_bundle.train_X = [csp_bundle.train_X, de_bundle.train_X];
feature_bundle.test_X = [csp_bundle.test_X, de_bundle.test_X];
feature_bundle.train_Y = csp_bundle.train_Y;
feature_bundle.test_Y = csp_bundle.test_Y;
feature_bundle.selected_filter_idx = csp_bundle.selected_filter_idx;
feature_bundle.feature_blocks = {'csp', 'differential_entropy'};
end
