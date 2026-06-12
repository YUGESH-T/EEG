function [X_train_sel, X_test_sel, selected_idx] = select_top_k_features( ...
        X_train, y_train, X_test, method, top_k)
% SELECT_TOP_K_FEATURES  Rank-and-select top-K features for classification.
%
%   [X_train_sel, X_test_sel, selected_idx] = select_top_k_features( ...
%       X_train, y_train, X_test, method, top_k)
%
%   Inputs:
%     X_train  - [n_train × n_features] training feature matrix
%     y_train  - [n_train × 1] training labels (exactly 2 classes)
%     X_test   - [n_test  × n_features] test feature matrix
%     method   - 'fisher' | 'mutual_information'
%     top_k    - number of features to keep (clamped to n_features)
%
%   Outputs:
%     X_train_sel  - [n_train × top_k] selected training features
%     X_test_sel   - [n_test  × top_k] selected test features
%     selected_idx - [1 × top_k] column indices of selected features
%
%   Feature ranking is computed ONLY from X_train and y_train.
%   The same column indices are applied to X_test (no leakage).
%
%   Supports two ranking methods:
%     'fisher'               — Fisher discriminant ratio (fast, closed-form)
%     'mutual_information'   — Histogram-based MI estimate (non-parametric)

    n_features = size(X_train, 2);

    % --- Clamp top_k to available features -----------------------------------
    if top_k >= n_features
        top_k = n_features;
    end
    if top_k < 1
        top_k = 1;
    end

    % --- NaN / Inf guard on inputs -------------------------------------------
    nan_mask_train = any(~isfinite(X_train), 1);
    nan_mask_test  = any(~isfinite(X_test), 1);
    bad_cols = nan_mask_train | nan_mask_test;
    if any(bad_cols)
        warning('select_top_k_features:nanInf', ...
            '%d feature columns contain NaN/Inf; zeroing them before ranking.', ...
            sum(bad_cols));
        X_train(:, bad_cols) = 0;
        X_test(:, bad_cols)  = 0;
    end

    % --- Compute per-feature scores (train only) -----------------------------
    method = lower(string(method));
    switch method
        case "fisher"
            scores = compute_fisher_scores(X_train, y_train);
        case "mutual_information"
            scores = compute_mi_scores(X_train, y_train);
        otherwise
            error('select_top_k_features:unknownMethod', ...
                'Unknown feature selection method: %s. Use ''fisher'' or ''mutual_information''.', ...
                method);
    end

    % --- Select top-K by descending score ------------------------------------
    [~, rank_order] = sort(scores, 'descend');
    selected_idx = sort(rank_order(1:top_k));  % keep original column order

    X_train_sel = X_train(:, selected_idx);
    X_test_sel  = X_test(:, selected_idx);

    % --- Debug prints --------------------------------------------------------
    fprintf('    Feature selection method: %s\n', method);
    fprintf('    Selected features: %d/%d\n', top_k, n_features);

    % --- Post-selection NaN/Inf check ----------------------------------------
    if any(~isfinite(X_train_sel(:))) || any(~isfinite(X_test_sel(:)))
        error('select_top_k_features:postNaN', ...
            'NaN or Inf detected in features after selection.');
    end
end

%% ========================================================================
%  FISHER DISCRIMINANT RATIO
%  ========================================================================

function scores = compute_fisher_scores(X, y)
% Fisher score: (mu1 - mu2)^2 / (var1 + var2)
% Higher = more discriminative between the two classes.

    labels = unique(y);
    if numel(labels) ~= 2
        error('compute_fisher_scores:classCount', ...
            'Fisher score requires exactly 2 classes, got %d.', numel(labels));
    end

    n_features = size(X, 2);
    scores = zeros(1, n_features);

    mask1 = y == labels(1);
    mask2 = y == labels(2);

    for f = 1:n_features
        c1 = X(mask1, f);
        c2 = X(mask2, f);
        scores(f) = (mean(c1) - mean(c2))^2 / (var(c1) + var(c2) + eps);
    end
end

%% ========================================================================
%  MUTUAL INFORMATION (histogram-based estimate)
%  ========================================================================

function scores = compute_mi_scores(X, y)
% Estimates I(X_f ; Y) for each feature f using histogram binning.
% Uses Sturges' rule for bin count: ceil(1 + log2(n)).

    labels = unique(y);
    if numel(labels) ~= 2
        error('compute_mi_scores:classCount', ...
            'MI score requires exactly 2 classes, got %d.', numel(labels));
    end

    n_samples  = size(X, 1);
    n_features = size(X, 2);
    scores = zeros(1, n_features);

    n_bins = max(ceil(1 + log2(n_samples)), 4);  % Sturges' rule, floor at 4

    % Class prior probabilities
    p_y = zeros(1, numel(labels));
    for c = 1:numel(labels)
        p_y(c) = sum(y == labels(c)) / n_samples;
    end

    for f = 1:n_features
        col = X(:, f);

        % Determine bin edges from training data
        col_min = min(col);
        col_max = max(col);
        if col_max == col_min
            % Zero-variance feature → zero MI
            scores(f) = 0;
            continue;
        end

        edges = linspace(col_min, col_max, n_bins + 1);
        % Ensure the last edge captures the maximum value
        edges(end) = col_max + eps;

        % Discretize
        bin_idx = discretize(col, edges);
        if any(isnan(bin_idx))
            % Fallback for edge cases
            bin_idx(isnan(bin_idx)) = 1;
        end

        mi = 0;
        for b = 1:n_bins
            b_mask = bin_idx == b;
            p_x = sum(b_mask) / n_samples;
            if p_x < eps
                continue;
            end

            for c = 1:numel(labels)
                joint_mask = b_mask & (y == labels(c));
                p_xy = sum(joint_mask) / n_samples;
                if p_xy < eps
                    continue;
                end
                mi = mi + p_xy * log2(p_xy / (p_x * p_y(c) + eps) + eps);
            end
        end

        scores(f) = max(mi, 0);  % MI is non-negative; clamp numerical noise
    end
end
