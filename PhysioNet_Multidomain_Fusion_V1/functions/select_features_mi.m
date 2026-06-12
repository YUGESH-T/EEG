function [X_train_sel, X_test_sel, selected_idx, mi_scores] = ...
        select_features_mi(X_train, y_train, X_test, top_k)
%SELECT_FEATURES_MI Mutual-information feature selection.
%
%   [X_train_sel, X_test_sel, selected_idx, mi_scores] = ...
%       select_features_mi(X_train, y_train, X_test, top_k)
%
%   Estimates MI(feature, class) using histogram binning (Sturges' rule).
%   Selects top_k features by descending MI score.
%   No Statistics Toolbox dependency.
%
%   LEAKAGE-SAFE: MI scores computed exclusively from X_train, y_train.
%   The same column indices are applied to X_test without inspecting it.

    n_features = size(X_train, 2);
    n_train    = size(X_train, 1);

    % --- Pass-through if top_k >= n_features or disabled ------------------
    if top_k <= 0 || top_k >= n_features
        X_train_sel  = X_train;
        X_test_sel   = X_test;
        selected_idx = 1:n_features;
        mi_scores    = zeros(1, n_features);
        return;
    end

    y_train = y_train(:);
    labels  = unique(y_train);

    if numel(labels) ~= 2
        error('select_features_mi:classCount', ...
            'MI selection requires exactly 2 classes, got %d.', numel(labels));
    end

    % Number of bins (Sturges' rule, floor ≥ 5)
    n_bins = max(floor(1 + log2(n_train)), 5);

    % Class probabilities
    n_classes = numel(labels);
    p_c = zeros(n_classes, 1);
    for ci = 1:n_classes
        p_c(ci) = sum(y_train == labels(ci)) / n_train;
    end

    mi_scores = zeros(1, n_features);

    for d = 1:n_features
        col = X_train(:, d);

        % Guard: constant or non-finite columns
        if ~any(isfinite(col)) || var(col) < eps
            mi_scores(d) = 0;
            continue;
        end

        % Bin the feature
        x_min = min(col);
        x_max = max(col);
        if x_max - x_min < eps
            mi_scores(d) = 0;
            continue;
        end

        bin_edges = linspace(x_min - eps, x_max + eps, n_bins + 1);
        x_binned  = discretize_safe(col, bin_edges);

        % Compute MI = Σ p(x,y) log[ p(x,y) / (p(x) p(y)) ]
        mi = 0;
        for b = 1:n_bins
            b_mask = x_binned == b;
            p_x    = sum(b_mask) / n_train;
            if p_x < eps, continue; end

            for ci = 1:n_classes
                p_xy = sum(b_mask & y_train == labels(ci)) / n_train;
                if p_xy > eps
                    mi = mi + p_xy * log(p_xy / (p_x * p_c(ci)));
                end
            end
        end

        mi_scores(d) = max(mi, 0);   % MI is non-negative
    end

    % --- Select top-K by descending MI score ------------------------------
    [~, rank_order] = sort(mi_scores, 'descend');
    selected_idx = sort(rank_order(1:top_k));   % preserve column order

    X_train_sel = X_train(:, selected_idx);
    X_test_sel  = X_test(:, selected_idx);

    % Post-selection NaN/Inf guard
    X_train_sel(~isfinite(X_train_sel)) = 0;
    X_test_sel(~isfinite(X_test_sel))   = 0;
end


%% =========================================================================

function bins = discretize_safe(x, edges)
%DISCRETIZE_SAFE Bin assignment with NaN-safe fallback.
    try
        bins = discretize(x, edges);
    catch
        % Manual binning fallback for older MATLAB versions
        n_bins = numel(edges) - 1;
        bins   = ones(size(x));
        for i = 1:numel(x)
            for b = 1:n_bins
                if x(i) >= edges(b) && x(i) < edges(b + 1)
                    bins(i) = b;
                    break;
                end
            end
            if x(i) >= edges(end)
                bins(i) = n_bins;
            end
        end
    end
    bins(isnan(bins)) = 1;
end
