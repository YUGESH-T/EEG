function [W, eigenvalues] = compute_csp_projection(X_train, y_train, ...
        n_components, csp_type, csp_beta)
%COMPUTE_CSP_PROJECTION Compute CSP spatial filters (standard or shrinkage).
%
%   [W, eigenvalues] = compute_csp_projection(X_train, y_train, n_components, csp_type, csp_beta)
%
%   Inputs:
%     X_train      - [n_channels × n_time × n_train] filtered training epochs
%     y_train      - [n_train × 1] labels (exactly 2 classes)
%     n_components - total CSP components (even; takes top/bottom halves)
%     csp_type     - 'standard' (fixed α=0.12) or 'shrinkage' (Ledoit-Wolf OAS)
%     csp_beta     - class-level regularization toward global cov (default 0.15)
%
%   Outputs:
%     W            - [n_channels × n_components] spatial filter matrix
%     eigenvalues  - sorted generalized eigenvalues
%
%   CSP filters are computed ONLY from X_train (leakage-safe).
%   Standard mode replicates the baseline's fixed regularization.
%   Shrinkage mode replaces the per-trial fixed α with data-adaptive
%   Oracle Approximating Shrinkage (Chen et al., 2010).

    if nargin < 4 || isempty(csp_type), csp_type = 'standard'; end
    if nargin < 5 || isempty(csp_beta), csp_beta = 0.15; end

    n_channels = size(X_train, 1);
    y_train    = y_train(:);
    labels     = unique(y_train);

    if numel(labels) ~= 2
        error('compute_csp_projection:classCount', ...
            'CSP requires exactly 2 classes, got %d.', numel(labels));
    end

    % Ensure even component count, clamped to channel count
    n_components = min(n_components, 2 * floor(n_channels / 2));
    n_components = max(n_components, 2);
    half = floor(n_components / 2);

    % ----- Compute class covariance matrices -----------------------------
    csp_type_str = lower(string(csp_type));
    switch csp_type_str
        case "standard"
            cov_1 = avg_cov_standard(X_train(:, :, y_train == labels(1)));
            cov_2 = avg_cov_standard(X_train(:, :, y_train == labels(2)));
        case "shrinkage"
            cov_1 = avg_cov_shrinkage(X_train(:, :, y_train == labels(1)));
            cov_2 = avg_cov_shrinkage(X_train(:, :, y_train == labels(2)));
        otherwise
            error('compute_csp_projection:unknownType', ...
                'Unknown CSP type: %s. Use ''standard'' or ''shrinkage''.', csp_type);
    end

    % ----- Class-level regularization (blend toward global cov) ----------
    global_cov = (cov_1 + cov_2) / 2;
    cov_1 = (1 - csp_beta) * cov_1 + csp_beta * global_cov;
    cov_2 = (1 - csp_beta) * cov_2 + csp_beta * global_cov;

    % ----- Ensure SPD ----------------------------------------------------
    cov_1 = ensure_spd(cov_1, n_channels);
    cov_2 = ensure_spd(cov_2, n_channels);

    % ----- Generalized eigendecomposition --------------------------------
    composite = cov_1 + cov_2;
    composite = ensure_spd(composite, n_channels);

    try
        [V, D] = eig(cov_1, composite);
    catch
        % Fallback: add larger jitter
        composite = composite + 1e-5 * eye(n_channels);
        cov_1_reg = cov_1 + 1e-5 * eye(n_channels);
        [V, D] = eig(cov_1_reg, composite);
    end

    eigenvalues = real(diag(D));
    [eigenvalues, order] = sort(eigenvalues, 'descend');
    V = real(V(:, order));

    % Select top half (class 1 extreme) and bottom half (class 2 extreme)
    top_idx    = 1:half;
    bottom_idx = (numel(eigenvalues) - half + 1):numel(eigenvalues);
    W          = [V(:, top_idx), V(:, bottom_idx)];
    eigenvalues = [eigenvalues(top_idx); eigenvalues(bottom_idx)];
end


%% ========================================================================
%  STANDARD COVARIANCE (fixed alpha = 0.12, matching baseline)
%  ========================================================================

function cov_avg = avg_cov_standard(X)
%AVG_COV_STANDARD Average trace-normalized covariance with fixed shrinkage.
%   Replicates the baseline regularization from plain_csp_catboost.m.
    n_ch     = size(X, 1);
    n_epochs = size(X, 3);
    cov_avg  = zeros(n_ch);
    alpha    = 0.12;  % Fixed shrinkage intensity (baseline value)

    for e = 1:n_epochs
        trial = squeeze(X(:, :, e));
        trial = trial - mean(trial, 2);  % zero-mean per channel

        c      = (trial * trial') / size(trial, 2);
        target = trace(c) / n_ch * eye(n_ch);
        c      = (1 - alpha) * c + alpha * target;
        c      = c + 1e-4 * eye(n_ch);
        c      = (c + c') / 2;  % ensure symmetry

        cov_avg = cov_avg + c / max(trace(c), eps);
    end

    cov_avg = cov_avg / max(n_epochs, 1);
    cov_avg = (cov_avg + cov_avg') / 2;
end


%% ========================================================================
%  SHRINKAGE COVARIANCE (Ledoit-Wolf OAS, data-adaptive)
%  ========================================================================

function cov_avg = avg_cov_shrinkage(X)
%AVG_COV_SHRINKAGE Average trace-normalized covariance with Ledoit-Wolf shrinkage.
%   Replaces the fixed α=0.12 with per-trial analytically-optimal α.
    n_ch     = size(X, 1);
    n_epochs = size(X, 3);
    cov_avg  = zeros(n_ch);

    for e = 1:n_epochs
        trial = squeeze(X(:, :, e));
        trial = trial - mean(trial, 2);  % zero-mean per channel

        c = ledoit_wolf_oas(trial, n_ch);
        c = c + 1e-4 * eye(n_ch);
        c = (c + c') / 2;

        cov_avg = cov_avg + c / max(trace(c), eps);
    end

    cov_avg = cov_avg / max(n_epochs, 1);
    cov_avg = (cov_avg + cov_avg') / 2;
end


function cov_shrunk = ledoit_wolf_oas(trial_data, p)
%LEDOIT_WOLF_OAS Oracle Approximating Shrinkage covariance estimator.
%   Chen, Wiesel, Eldar, Hero (2010).
%
%   trial_data: [p × n] data matrix (p channels, n time points)
%   p: number of channels (passed explicitly for clarity)
%
%   Returns shrunk covariance toward scaled identity: α*μI + (1-α)*S

    n = size(trial_data, 2);

    % Sample covariance
    S  = (trial_data * trial_data') / n;
    mu = trace(S) / p;

    % OAS estimator for optimal shrinkage intensity
    tr_S2    = trace(S * S);
    tr_S_sq  = trace(S)^2;

    rho_num = (1 - 2 / p) * tr_S2 + tr_S_sq;
    rho_den = (n + 1 - 2 / p) * (tr_S2 - tr_S_sq / p);

    if abs(rho_den) < eps
        alpha_est = 1.0;
    else
        alpha_est = max(0, min(1, rho_num / rho_den));
    end

    cov_shrunk = (1 - alpha_est) * S + alpha_est * mu * eye(p);
end


%% ========================================================================
%  UTILITIES
%  ========================================================================

function M = ensure_spd(M, n)
%ENSURE_SPD Force matrix to symmetric positive-definite.
    M = (M + M') / 2;

    try
        chol(M);
        return;  % Already SPD
    catch
        % Not SPD — fix eigenvalues
    end

    [V, D] = eig(M);
    d = real(diag(D));
    d(d < eps) = eps;
    M = V * diag(d) * V';
    M = (M + M') / 2;

    % Final fallback: add small ridge
    if any(~isfinite(M(:)))
        M = eye(n) * eps;
    end
end
