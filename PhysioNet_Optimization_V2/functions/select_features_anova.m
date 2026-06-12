function [X_train_sel, X_test_sel, selected_idx, f_scores] = ...
        select_features_anova(X_train, y_train, X_test, top_k)
%SELECT_FEATURES_ANOVA One-way ANOVA F-test feature selection.
%
%   [X_train_sel, X_test_sel, selected_idx, f_scores] = ...
%       select_features_anova(X_train, y_train, X_test, top_k)
%
%   Computes F = SS_between / SS_within for each feature using training
%   data only. Selects top_k features by descending F-score.
%
%   LEAKAGE-SAFE: F-statistics computed exclusively from X_train, y_train.
%   The same column indices are applied to X_test without inspecting it.

    n_features = size(X_train, 2);

    % --- Pass-through if top_k >= n_features or disabled -----------------
    if top_k <= 0 || top_k >= n_features
        X_train_sel  = X_train;
        X_test_sel   = X_test;
        selected_idx = 1:n_features;
        f_scores     = zeros(1, n_features);
        return;
    end

    y_train = y_train(:);
    labels  = unique(y_train);

    if numel(labels) ~= 2
        error('select_features_anova:classCount', ...
            'ANOVA selection requires exactly 2 classes, got %d.', numel(labels));
    end

    idx1 = y_train == labels(1);
    idx2 = y_train == labels(2);
    n1   = sum(idx1);
    n2   = sum(idx2);

    f_scores = zeros(1, n_features);

    for d = 1:n_features
        col = X_train(:, d);

        % Guard: zero-variance or non-finite columns
        if ~any(isfinite(col)) || var(col) < eps
            f_scores(d) = 0;
            continue;
        end

        mu_all = mean(col);
        mu1    = mean(col(idx1));
        mu2    = mean(col(idx2));
        var1   = var(col(idx1));
        var2   = var(col(idx2));

        ss_between = n1 * (mu1 - mu_all)^2 + n2 * (mu2 - mu_all)^2;
        ss_within  = max((n1 - 1) * var1 + (n2 - 1) * var2, 0);

        if ss_within > eps
            f_scores(d) = ss_between / ss_within;
        else
            f_scores(d) = 0;
        end
    end

    % --- Select top-K by descending F-score ------------------------------
    [~, rank_order] = sort(f_scores, 'descend');
    selected_idx = sort(rank_order(1:top_k));  % preserve column order

    X_train_sel = X_train(:, selected_idx);
    X_test_sel  = X_test(:, selected_idx);

    % --- Post-selection NaN/Inf guard ------------------------------------
    X_train_sel(~isfinite(X_train_sel)) = 0;
    X_test_sel(~isfinite(X_test_sel))   = 0;
end
