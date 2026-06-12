function [X_train_norm, X_test_norm, stats] = normalize_features(X_train, X_test, norm_mode)
%NORMALIZE_FEATURES Train-fold-only feature normalization.
%
%   [X_train_norm, X_test_norm, stats] = normalize_features(X_train, X_test, norm_mode)
%
%   Modes:
%     'zscore' - Standard z-score (mean / std)
%     'robust' - Robust scaling (median / IQR)
%     'none'   - No normalization (pass-through)
%
%   LEAKAGE-SAFE: all statistics computed exclusively from X_train.

    if nargin < 3 || isempty(norm_mode)
        norm_mode = 'zscore';
    end

    norm_mode = lower(string(norm_mode));
    stats = struct('mode', char(norm_mode));

    switch norm_mode
        case "zscore"
            mu = mean(X_train, 1);
            sigma = std(X_train, 0, 1);
            sigma(sigma < eps) = 1;  % Guard zero-variance columns

            X_train_norm = (X_train - mu) ./ sigma;
            X_test_norm  = (X_test  - mu) ./ sigma;

            stats.mu    = mu;
            stats.sigma = sigma;

        case "robust"
            med     = median(X_train, 1);
            q1      = prctile(X_train, 25, 1);
            q3      = prctile(X_train, 75, 1);
            iqr_val = q3 - q1;
            iqr_val(iqr_val < eps) = 1;  % Guard zero-IQR columns

            X_train_norm = (X_train - med) ./ iqr_val;
            X_test_norm  = (X_test  - med) ./ iqr_val;

            stats.median_val = med;
            stats.iqr        = iqr_val;

        case "none"
            X_train_norm = X_train;
            X_test_norm  = X_test;

        otherwise
            error('normalize_features:unknownMode', ...
                'Unknown normalization mode: %s. Use ''zscore'', ''robust'', or ''none''.', ...
                norm_mode);
    end

    % Post-normalization NaN/Inf guard
    X_train_norm(~isfinite(X_train_norm)) = 0;
    X_test_norm(~isfinite(X_test_norm))   = 0;
end
