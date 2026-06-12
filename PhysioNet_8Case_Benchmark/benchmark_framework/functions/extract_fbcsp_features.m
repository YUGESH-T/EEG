function feature_bundle = extract_fbcsp_features(X_left, X_right, split, cfg)
bands = resolve_fbcsp_bands(cfg);
if size(bands, 1) <= 1
    band_cfg = cfg;
    band_cfg.band = bands(1, :);
    feature_bundle = extract_csp_features(X_left, X_right, split, band_cfg);
    feature_bundle.selected_bands = bands;
    return;
end

labels = [ones(size(X_left, 3), 1); 2 * ones(size(X_right, 3), 1)];
train_labels = labels(split.train_idx);

band_train_blocks = cell(1, size(bands, 1));
band_test_blocks = cell(1, size(bands, 1));
band_scores = zeros(1, size(bands, 1));
band_feature_ranges = cell(1, size(bands, 1));

for band_idx = 1:size(bands, 1)
    band_cfg = cfg;
    band_cfg.band = bands(band_idx, :);
    band_bundle = extract_csp_features(X_left, X_right, split, band_cfg);

    band_train_blocks{band_idx} = band_bundle.train_X;
    band_test_blocks{band_idx} = band_bundle.test_X;
    band_scores(band_idx) = mean(fisher_scores_for_features(band_bundle.train_X, train_labels));
end

for band_idx = 1:size(bands, 1)
    if band_idx == 1
        start_col = 1;
    else
        start_col = band_feature_ranges{band_idx - 1}(end) + 1;
    end
    stop_col = start_col + size(band_train_blocks{band_idx}, 2) - 1;
    band_feature_ranges{band_idx} = start_col:stop_col;
end

[~, ranked_bands] = sort(band_scores, 'descend');
n_selected_bands = min(size(bands, 1), resolve_selected_band_count(cfg, size(bands, 1)));
selected_bands = sort(ranked_bands(1:n_selected_bands));
selected_idx = [];
for i = 1:numel(selected_bands)
    selected_idx = [selected_idx, band_feature_ranges{selected_bands(i)}]; %#ok<AGROW>
end

train_X = [];
test_X = [];
for band_idx = 1:size(bands, 1)
    train_X = [train_X, band_train_blocks{band_idx}]; %#ok<AGROW>
    test_X = [test_X, band_test_blocks{band_idx}]; %#ok<AGROW>
end

feature_bundle = struct();
feature_bundle.train_X = train_X(:, selected_idx);
feature_bundle.test_X = test_X(:, selected_idx);
feature_bundle.train_Y = labels(split.train_idx);
feature_bundle.test_Y = labels(split.test_idx);
feature_bundle.selected_bands = bands(selected_bands, :);
feature_bundle.selected_filter_idx = selected_idx;
end

function bands = resolve_fbcsp_bands(cfg)
if isfield(cfg, 'fbcsp_bands') && ~isempty(cfg.fbcsp_bands)
    bands = cfg.fbcsp_bands;
    return;
end
if isfield(cfg, 'features') && isfield(cfg.features, 'csp_bands') && ~isempty(cfg.features.csp_bands)
    bands = cfg.features.csp_bands;
    return;
end
if isfield(cfg, 'band') && numel(cfg.band) == 2
    low = cfg.band(1);
    high = cfg.band(2);
else
    low = 8;
    high = 30;
end
bands = [low 12; 12 16; 16 22; 22 high];
bands(:, 1) = max(bands(:, 1), low);
bands(:, 2) = min(bands(:, 2), high);
bands = bands(bands(:, 2) > bands(:, 1), :);
if isempty(bands)
    bands = [low high];
end
end

function n_selected = resolve_selected_band_count(cfg, n_bands)
if isfield(cfg, 'fbcsp_band_count') && ~isempty(cfg.fbcsp_band_count)
    n_selected = max(1, round(cfg.fbcsp_band_count));
else
    n_selected = min(3, n_bands);
end
end

function scores = fisher_scores_for_features(features, labels)
class_labels = unique(labels);
if numel(class_labels) ~= 2
    scores = zeros(1, size(features, 2));
    return;
end

scores = zeros(1, size(features, 2));
class1 = features(labels == class_labels(1), :);
class2 = features(labels == class_labels(2), :);
mu1 = mean(class1, 1);
mu2 = mean(class2, 1);
var1 = var(class1, 0, 1);
var2 = var(class2, 0, 1);
scores = (mu1 - mu2) .^ 2 ./ (var1 + var2 + eps);
end
