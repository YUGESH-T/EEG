function feature_bundle = extract_fbcsp_mi_features(X_left, X_right, split, cfg)
fbcsp_cfg = resolve_fbcsp_mi_cfg(cfg);
bands = fbcsp_cfg.bands;

band_train_blocks = cell(1, size(bands, 1));
band_test_blocks = cell(1, size(bands, 1));
band_ranges = cell(1, size(bands, 1));
band_feature_labels = cell(1, size(bands, 1));

feature_offset = 0;
labels = [ones(size(X_left, 3), 1); 2 * ones(size(X_right, 3), 1)];

for band_idx = 1:size(bands, 1)
    band = bands(band_idx, :);
    [X_left_band, X_right_band] = bandpass_trials(X_left, X_right, cfg.fs, band, fbcsp_cfg.filter_order);

    band_cfg = cfg;
    band_cfg.band = band;
    band_cfg.csp_components = resolve_csp_components(cfg);

    band_bundle = extract_csp_features(X_left_band, X_right_band, split, band_cfg);
    band_train_blocks{band_idx} = band_bundle.train_X;
    band_test_blocks{band_idx} = band_bundle.test_X;

    band_feature_count = size(band_bundle.train_X, 2);
    band_ranges{band_idx} = feature_offset + (1:band_feature_count);
    feature_offset = feature_offset + band_feature_count;
    band_feature_labels{band_idx} = sprintf('[%g %g]', band(1), band(2));
end

train_X = cat(2, band_train_blocks{:});
test_X = cat(2, band_test_blocks{:});
train_Y = labels(split.train_idx);
test_Y = labels(split.test_idx);

if isempty(train_X)
    error('FBCSP-MI feature extraction produced no features.');
end

[selected_idx, mi_scores, ranked_idx] = select_mutual_information_features( ...
    train_X, ...
    train_Y, ...
    fbcsp_cfg.top_k, ...
    fbcsp_cfg.mi_bins);

feature_bundle = struct();
feature_bundle.train_X = train_X(:, selected_idx);
feature_bundle.test_X = test_X(:, selected_idx);
feature_bundle.train_Y = train_Y;
feature_bundle.test_Y = test_Y;
feature_bundle.raw_train_X = train_X;
feature_bundle.raw_test_X = test_X;
feature_bundle.selected_feature_idx = selected_idx;
feature_bundle.ranked_feature_idx = ranked_idx;
feature_bundle.mi_scores = mi_scores;
feature_bundle.raw_feature_count = size(train_X, 2);
feature_bundle.selected_feature_count = numel(selected_idx);
feature_bundle.band_feature_ranges = band_ranges;
feature_bundle.selected_bands = bands;
feature_bundle.band_labels = band_feature_labels;
feature_bundle.feature_blocks = {'fbcsp_mi'};
feature_bundle.fbcsp_mi = fbcsp_cfg;
end

function cfg_out = resolve_fbcsp_mi_cfg(cfg)
cfg_out = struct();

cfg_out.bands = [4 8; 8 12; 12 16; 16 20; 20 24; 24 30];
cfg_out.top_k = 20;
cfg_out.mi_bins = 8;
cfg_out.filter_order = 4;

if isfield(cfg, 'fbcsp_mi') && ~isempty(cfg.fbcsp_mi)
    if isfield(cfg.fbcsp_mi, 'bands') && ~isempty(cfg.fbcsp_mi.bands)
        cfg_out.bands = cfg.fbcsp_mi.bands;
    end
    if isfield(cfg.fbcsp_mi, 'top_k') && ~isempty(cfg.fbcsp_mi.top_k)
        cfg_out.top_k = cfg.fbcsp_mi.top_k;
    end
    if isfield(cfg.fbcsp_mi, 'mi_bins') && ~isempty(cfg.fbcsp_mi.mi_bins)
        cfg_out.mi_bins = cfg.fbcsp_mi.mi_bins;
    end
    if isfield(cfg.fbcsp_mi, 'filter_order') && ~isempty(cfg.fbcsp_mi.filter_order)
        cfg_out.filter_order = cfg.fbcsp_mi.filter_order;
    end
end

cfg_out.top_k = max(1, round(cfg_out.top_k));
cfg_out.mi_bins = max(4, round(cfg_out.mi_bins));
cfg_out.filter_order = max(1, round(cfg_out.filter_order));
end

function n_components = resolve_csp_components(cfg)
if isfield(cfg, 'csp_components') && ~isempty(cfg.csp_components)
    n_components = cfg.csp_components;
elseif isfield(cfg, 'features') && isfield(cfg.features, 'csp_components') && ~isempty(cfg.features.csp_components)
    n_components = cfg.features.csp_components;
else
    n_components = 4;
end
end

function [X_left_f, X_right_f] = bandpass_trials(X_left, X_right, fs, band, filter_order)
[b, a] = butter(filter_order, band / (fs / 2), 'bandpass');
X_left_f = zeros(size(X_left));
X_right_f = zeros(size(X_right));

for trial_idx = 1:size(X_left, 3)
    X_left_f(:, :, trial_idx) = filtfilt(b, a, double(X_left(:, :, trial_idx)') )';
end
for trial_idx = 1:size(X_right, 3)
    X_right_f(:, :, trial_idx) = filtfilt(b, a, double(X_right(:, :, trial_idx)') )';
end
end
