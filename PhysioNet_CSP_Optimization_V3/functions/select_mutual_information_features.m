function [X_train_sel, X_test_sel, selected_idx, scores] = select_mutual_information_features(X_train, y_train, X_test, top_k)
%SELECT_MUTUAL_INFORMATION_FEATURES Mutual information feature selection.
%
%   [X_train_sel, X_test_sel, selected_idx, scores] = ...
%       select_mutual_information_features(X_train, y_train, X_test, top_k)

    n_features = size(X_train, 2);

    if nargin < 4 || isempty(top_k) || top_k <= 0 || top_k >= n_features
        X_train_sel  = X_train;
        X_test_sel   = X_test;
        selected_idx = 1:n_features;
        scores       = zeros(1, n_features);
        return;
    end

    n_bins = 8;
    X_train = double(X_train);
    y_train = double(y_train(:));
    class_labels = unique(y_train);

    if numel(class_labels) ~= 2
        error('Mutual information feature selection requires exactly two classes.');
    end

    top_k = max(1, round(top_k));

    scores = zeros(1, size(X_train, 2));
    for feature_idx = 1:size(X_train, 2)
        scores(feature_idx) = feature_mutual_information(X_train(:, feature_idx), y_train, class_labels, n_bins);
    end

    [~, rank_order] = sort(scores, 'descend');
    selected_idx = sort(rank_order(1:top_k));

    X_train_sel = X_train(:, selected_idx);
    X_test_sel  = X_test(:, selected_idx);

    X_train_sel(~isfinite(X_train_sel)) = 0;
    X_test_sel(~isfinite(X_test_sel))   = 0;
end

function mi = feature_mutual_information(feature_values, labels, class_labels, n_bins)
    feature_values = double(feature_values(:));
    mi = 0;

    finite_mask = isfinite(feature_values);
    feature_values = feature_values(finite_mask);
    labels = labels(finite_mask);

    if isempty(feature_values) || all(feature_values == feature_values(1))
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
