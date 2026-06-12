function feature_bundle = extract_csp_fbcsp_features(X_left, X_right, split, cfg)
single_band_cfg = cfg;
single_band_cfg.band = resolve_single_csp_band(cfg);

csp_bundle = extract_csp_features(X_left, X_right, split, single_band_cfg);
fbcsp_bundle = extract_fbcsp_features(X_left, X_right, split, cfg);

feature_bundle = struct();
feature_bundle.train_X = [csp_bundle.train_X, fbcsp_bundle.train_X];
feature_bundle.test_X = [csp_bundle.test_X, fbcsp_bundle.test_X];
feature_bundle.train_Y = csp_bundle.train_Y;
feature_bundle.test_Y = csp_bundle.test_Y;
feature_bundle.selected_filter_idx = csp_bundle.selected_filter_idx;
feature_bundle.feature_blocks = {'csp', 'fbcsp'};
feature_bundle.selected_bands = fbcsp_bundle.selected_bands;
end

function band = resolve_single_csp_band(cfg)
if isfield(cfg, 'band') && numel(cfg.band) == 2
    band = cfg.band;
else
    band = [8 30];
end
end
