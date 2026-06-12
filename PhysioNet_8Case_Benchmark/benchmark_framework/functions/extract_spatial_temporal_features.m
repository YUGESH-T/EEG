function feature_bundle = extract_spatial_temporal_features(X_left, X_right, split, cfg)
feature_bundle = extract_csp_temporal_features(X_left, X_right, split, cfg);
feature_bundle.feature_blocks = {'csp', 'temporal'};
end
