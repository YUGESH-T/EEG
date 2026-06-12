function [selected_idx, scores, ranked_idx] = select_mutual_information_features(X, Y, top_k, n_bins)
if nargin < 3 || isempty(top_k)
    top_k = 20;
end
if nargin < 4 || isempty(n_bins)
    n_bins = 8;
end

X = double(X);
Y = double(Y(:));
class_labels = unique(Y);
if numel(class_labels) ~= 2
    error('Mutual information feature selection requires exactly two classes.');
end

top_k = max(1, round(top_k));
n_bins = max(4, round(n_bins));

scores = zeros(1, size(X, 2));
for feature_idx = 1:size(X, 2)
    scores(feature_idx) = feature_mutual_information(X(:, feature_idx), Y, class_labels, n_bins);
end

[scores_sorted, ranked_idx] = sort(scores, 'descend'); %#ok<ASGLU>
selected_idx = ranked_idx(1:min(top_k, numel(ranked_idx)));
end

function mi = feature_mutual_information(feature_values, labels, class_labels, n_bins)
feature_values = double(feature_values(:));
mi = 0;

if isempty(feature_values) || all(feature_values == feature_values(1))
    return;
end

finite_mask = isfinite(feature_values);
feature_values = feature_values(finite_mask);
labels = labels(finite_mask);
if isempty(feature_values)
    return;
end

min_val = min(feature_values);
max_val = max(feature_values);
if min_val == max_val
    return;
end

edges = linspace(min_val, max_val, n_bins + 1);
edges(1) = edges(1) - eps(edges(1));
edges(end) = edges(end) + eps(edges(end));

bin_ids = discretize(feature_values, edges);
bin_ids(isnan(bin_ids)) = 1;

n_samples = numel(feature_values);
if n_samples == 0
    return;
end

for bin_idx = 1:n_bins
    p_x = sum(bin_ids == bin_idx) / n_samples;
    if p_x == 0
        continue;
    end

    for class_idx = 1:numel(class_labels)
        p_y = sum(labels == class_labels(class_idx)) / n_samples;
        if p_y == 0
            continue;
        end

        p_xy = sum(bin_ids == bin_idx & labels == class_labels(class_idx)) / n_samples;
        if p_xy > 0
            mi = mi + p_xy * log2(p_xy / (p_x * p_y));
        end
    end
end
end
