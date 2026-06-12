function [train_features, test_features] = extract_minimal_fbcsp_features( ...
        X_train, y_train, X_test, cfg)
% EXTRACT_MINIMAL_FBCSP_FEATURES  3-band Filter-Bank CSP feature extraction.
%
%   [train_features, test_features] = extract_minimal_fbcsp_features( ...
%       X_train, y_train, X_test, cfg)
%
%   Applies independent CSP per frequency band, concatenates log-variance
%   features across bands.  Uses the SAME regularized covariance and CSP
%   projection logic as the validated plain CSP path.
%
%   Default bands (configurable via cfg.features.fbcsp_bands):
%     Band 1:  8–12 Hz  (mu rhythm)
%     Band 2: 12–20 Hz  (low beta)
%     Band 3: 20–30 Hz  (high beta)
%
%   Feature dimension = n_bands × csp_components (default 3×4 = 12).
%
%   NO PCA, NO feature selection, NO normalization — only concatenation.

    fs = cfg.fs;

    % --- Band definitions ---------------------------------------------------
    if isfield(cfg.features, 'fbcsp_bands') && ~isempty(cfg.features.fbcsp_bands)
        bands = cfg.features.fbcsp_bands;
    else
        bands = [8 12; 12 20; 20 30];
    end

    n_bands = size(bands, 1);

    % --- CSP component count (clamped to channel count) ----------------------
    n_components = min(cfg.features.csp_components, ...
        2 * floor(size(X_train, 1) / 2));

    labels = unique(y_train);
    if numel(labels) ~= 2
        error('extract_minimal_fbcsp_features:classCount', ...
            'CSP requires exactly two classes, got %d.', numel(labels));
    end

    % --- Resolve regularization lambda ------------------------------------
    if isfield(cfg.features, 'csp_regularization')
        lambda = cfg.features.csp_regularization;
    else
        lambda = 1e-3;
    end

    % --- Per-band CSP --------------------------------------------------------
    band_train = cell(1, n_bands);
    band_test  = cell(1, n_bands);

    fprintf('    Regularized CSP lambda: %.6f\n', lambda);

    for b = 1:n_bands
        low_f  = bands(b, 1);
        high_f = bands(b, 2);

        % Nyquist guard
        if high_f >= fs / 2
            high_f = fs / 2 - 1;
        end
        if low_f <= 0
            low_f = 0.5;
        end

        [bf, af] = butter(4, [low_f high_f] / (fs / 2), 'bandpass');

        X_train_f = filter_epochs_fbcsp(X_train, bf, af);
        X_test_f  = filter_epochs_fbcsp(X_test,  bf, af);

        W = compute_csp_projection_fbcsp(X_train_f, y_train, n_components, labels, lambda);

        band_train{b} = project_csp_fbcsp(X_train_f, W);
        band_test{b}  = project_csp_fbcsp(X_test_f,  W);

        fprintf('    FBCSP band: %d-%d Hz | features: %d\n', ...
            low_f, high_f, size(band_train{b}, 2));
    end

    % --- Concatenate across bands (no PCA, no selection) ---------------------
    train_features = horzcat(band_train{:});
    test_features  = horzcat(band_test{:});

    fprintf('    Total FBCSP feature dimension: %d\n', size(train_features, 2));

    % --- Dimension sanity check ----------------------------------------------
    n_train = size(X_train, 3);
    n_test  = size(X_test, 3);
    expected_dim = n_bands * n_components;
    if size(train_features, 1) ~= n_train || size(train_features, 2) ~= expected_dim
        error('extract_minimal_fbcsp_features:trainDim', ...
            'Train feature matrix [%d×%d] does not match expected [%d×%d].', ...
            size(train_features, 1), size(train_features, 2), ...
            n_train, expected_dim);
    end
    if size(test_features, 1) ~= n_test || size(test_features, 2) ~= expected_dim
        error('extract_minimal_fbcsp_features:testDim', ...
            'Test feature matrix [%d×%d] does not match expected [%d×%d].', ...
            size(test_features, 1), size(test_features, 2), ...
            n_test, expected_dim);
    end
end

%% ========================================================================
%  LOCAL HELPERS — exact copies of the validated plain CSP logic
%  ========================================================================

function X_f = filter_epochs_fbcsp(X, b, a)
    X_f = zeros(size(X));
    for e = 1:size(X, 3)
        X_f(:, :, e) = filtfilt(b, a, X(:, :, e)')';
    end
end

function W = compute_csp_projection_fbcsp(X, y, n_components, labels, lambda)
    cov_1 = avg_norm_cov_fbcsp(X(:, :, y == labels(1)), lambda);
    cov_2 = avg_norm_cov_fbcsp(X(:, :, y == labels(2)), lambda);
    global_cov = (cov_1 + cov_2) / 2;
    beta = 0.15;
    cov_1 = (1 - beta) * cov_1 + beta * global_cov;
    cov_2 = (1 - beta) * cov_2 + beta * global_cov;

    composite = cov_1 + cov_2;

    % Fallback: detect singular/unstable composite before eigendecomposition
    if rcond(composite) < eps
        warning('compute_csp_projection_fbcsp:instability', ...
            'Covariance instability detected; applying additional diagonal stabilization.');
        composite = composite + lambda * 10 * eye(size(composite, 1));
    end

    [V, D] = eig(cov_1, composite);
    d = diag(D);

    % Guard against NaN/Inf eigenvalues from near-singular matrices
    if any(~isfinite(d))
        warning('compute_csp_projection_fbcsp:nanEig', ...
            'Non-finite eigenvalues detected; clamping to [0,1].');
        d(~isfinite(d)) = 0.5;
        D = diag(d); %#ok<NASGU>
    end

    [~, order] = sort(d, 'descend');
    V = V(:, order);
    half = n_components / 2;
    W = [V(:, 1:half), V(:, end-half+1:end)];
end

function cov_avg = avg_norm_cov_fbcsp(X, lambda)
    n_ch = size(X, 1);
    cov_avg = zeros(n_ch);
    n_epochs = size(X, 3);
    for e = 1:n_epochs
        trial = squeeze(X(:, :, e));
        trial = trial - mean(trial, 2);
        c = (trial * trial') / size(trial, 2);
        alpha = 0.12;
        target = trace(c) / n_ch * eye(n_ch);
        c = (1 - alpha) * c + alpha * target;
        c = c + lambda * eye(n_ch);
        c = (c + c') / 2;
        cov_avg = cov_avg + c / max(trace(c), eps);
    end
    cov_avg = cov_avg / max(n_epochs, 1);
    cov_avg = (cov_avg + cov_avg') / 2;
end

function features = project_csp_fbcsp(X, W)
    n_epochs = size(X, 3);
    features = zeros(n_epochs, size(W, 2));
    for e = 1:n_epochs
        Z = W' * X(:, :, e);
        v = var(Z, 0, 2);
        features(e, :) = log(v / sum(v) + eps)';
    end
end
