function [features, ref_matrix] = extract_riemannian_features(epochs, ref_matrix_in)
%EXTRACT_RIEMANNIAN_FEATURES Riemannian tangent-space features from EEG.
%
%   [features, ref_matrix] = extract_riemannian_features(epochs)
%   [features, ref_matrix] = extract_riemannian_features(epochs, ref_matrix_in)
%
%   Training mode (no ref_matrix_in):
%     Computes Riemannian geometric mean from epoch covariances as reference.
%   Test mode (with ref_matrix_in):
%     Uses the provided reference matrix (fitted on training data).
%
%   Pipeline:
%     1. Regularised covariance per epoch (Ledoit-Wolf shrinkage)
%     2. Riemannian geometric mean of training covariances (Fréchet mean)
%     3. Tangent-space projection: logm(Ref^{-½} C Ref^{-½})
%     4. Upper-triangle vectorisation (sqrt(2) off-diagonal scaling)
%
%   Inputs:
%     epochs        - [n_channels × n_time × n_epochs]
%     ref_matrix_in - (optional) [n_ch × n_ch] SPD reference from training
%
%   Outputs:
%     features   - [n_epochs × n_ch*(n_ch+1)/2]
%     ref_matrix - [n_ch × n_ch] SPD reference matrix
%
%   LEAKAGE-SAFE: reference matrix computed from training epochs only.

    if nargin < 2, ref_matrix_in = []; end

    n_channels = size(epochs, 1);
    n_time     = size(epochs, 2);
    n_epochs   = size(epochs, 3);

    % === Step 1: Regularised covariance per epoch =========================
    covs = zeros(n_channels, n_channels, n_epochs);
    for e = 1:n_epochs
        trial = double(epochs(:, :, e));
        trial = trial - mean(trial, 2);           % zero-mean per channel

        c = (trial * trial') / n_time;
        c = lw_shrinkage_riem(c, n_channels, n_time);
        c = ensure_spd_riem(c, n_channels);
        covs(:, :, e) = c;
    end

    % === Step 2: Reference matrix =========================================
    if isempty(ref_matrix_in)
        ref_matrix = riemannian_mean(covs, n_channels);
    else
        ref_matrix = ref_matrix_in;
    end

    % === Step 3 & 4: Tangent space projection + vectorisation =============
    n_tangent = n_channels * (n_channels + 1) / 2;
    features  = zeros(n_epochs, n_tangent);

    ref_sqrt_inv = matrix_power_riem(ref_matrix, -0.5);

    if any(~isfinite(ref_sqrt_inv(:)))
        ref_sqrt_inv = eye(n_channels);
    end

    for e = 1:n_epochs
        C = covs(:, :, e);

        % Transport to tangent space
        S = ref_sqrt_inv * C * ref_sqrt_inv';
        S = (S + S') / 2;
        S = ensure_spd_riem(S, n_channels);

        % Matrix logarithm
        T = real(logm(S));
        T = (T + T') / 2;

        % Vectorise upper triangle
        features(e, :) = tangent_vectorize(T, n_channels);
    end

    % NaN / Inf guard
    features(~isfinite(features)) = 0;
end


%% =========================================================================
%  LEDOIT-WOLF SHRINKAGE
%  =========================================================================

function c = lw_shrinkage_riem(c, p, n)
%LW_SHRINKAGE_RIEM Ledoit-Wolf OAS shrinkage for covariance matrix.
    mu     = trace(c) / p;
    target = mu * eye(p);

    tr_c2   = trace(c * c);
    tr_c_sq = trace(c)^2;

    rho_num = (1 - 2 / p) * tr_c2 + tr_c_sq;
    rho_den = (n + 1 - 2 / p) * (tr_c2 - tr_c_sq / p);

    if abs(rho_den) < eps
        alpha = 1.0;
    else
        alpha = max(0, min(1, rho_num / rho_den));
    end

    c = (1 - alpha) * c + alpha * target;
end


%% =========================================================================
%  RIEMANNIAN GEOMETRIC MEAN (FRÉCHET MEAN ON SPD MANIFOLD)
%  =========================================================================

function M = riemannian_mean(covs, n_ch)
%RIEMANNIAN_MEAN Iterative geometric mean of SPD matrices.
%   Fréchet mean using tangent-space gradient descent.
    n_epochs = size(covs, 3);
    max_iter = 50;
    tol      = 1e-8;

    % Initialise with arithmetic mean
    M = mean(covs, 3);
    M = ensure_spd_riem(M, n_ch);

    for iter = 1:max_iter
        M_sqrt_inv = matrix_power_riem(M, -0.5);
        M_sqrt     = matrix_power_riem(M,  0.5);

        if any(~isfinite(M_sqrt_inv(:))) || any(~isfinite(M_sqrt(:)))
            break;
        end

        S_sum = zeros(n_ch);
        for e = 1:n_epochs
            S = M_sqrt_inv * covs(:, :, e) * M_sqrt_inv';
            S = (S + S') / 2;
            S = ensure_spd_riem(S, n_ch);
            T = real(logm(S));
            T = (T + T') / 2;
            S_sum = S_sum + T;
        end

        S_mean = S_sum / n_epochs;

        % Convergence check
        if norm(S_mean, 'fro') < tol
            break;
        end

        % Gradient step: M ← M^{½} expm(S̄) M^{½}
        M = M_sqrt * expm(S_mean) * M_sqrt';
        M = (M + M') / 2;
        M = ensure_spd_riem(M, n_ch);
    end
end


%% =========================================================================
%  MATRIX POWER VIA EIGENDECOMPOSITION
%  =========================================================================

function M_pow = matrix_power_riem(M, p)
%MATRIX_POWER_RIEM M^p via eigendecomposition of SPD matrix.
    [V, D] = eig(M);
    d = real(diag(D));
    d(d < eps) = eps;
    M_pow = real(V * diag(d.^p) * V');
    M_pow = (M_pow + M_pow') / 2;
end


%% =========================================================================
%  TANGENT-SPACE VECTORISATION
%  =========================================================================

function v = tangent_vectorize(T, n)
%TANGENT_VECTORIZE Upper-triangle vectorisation of symmetric matrix.
%   Off-diagonal elements scaled by sqrt(2) to preserve Frobenius norm.
    k = 0;
    v = zeros(1, n * (n + 1) / 2);
    for i = 1:n
        for j = i:n
            k = k + 1;
            if i == j
                v(k) = T(i, j);
            else
                v(k) = T(i, j) * sqrt(2);
            end
        end
    end
end


%% =========================================================================
%  SPD ENFORCEMENT
%  =========================================================================

function M = ensure_spd_riem(M, n)
%ENSURE_SPD_RIEM Force matrix to symmetric positive-definite.
    M = (M + M') / 2;

    try
        chol(M);
        return;   % already SPD
    catch
    end

    [V, D] = eig(M);
    d = real(diag(D));
    d(d < eps) = eps;
    M = V * diag(d) * V';
    M = (M + M') / 2;

    if any(~isfinite(M(:)))
        M = eye(n) * eps;
    end
end
