function [feat_train, feat_test] = extract_fbcsp_features( ...
        X_train_ep, y_train, X_test_ep, cfg)
%EXTRACT_FBCSP_FEATURES Filter Bank CSP (5-band) feature extraction.
%
%   [feat_train, feat_test] = extract_fbcsp_features(X_train_ep, y_train, X_test_ep, cfg)
%
%   Filter banks: [8,12], [12,16], [16,20], [20,24], [24,30] Hz
%   Per band:
%     1. Bandpass filter train and test epochs
%     2. Compute CSP spatial filters (TRAIN ONLY)
%     3. Project both sets through CSP
%     4. Extract log-variance features
%   Concatenate across all bands.
%
%   Inputs:
%     X_train_ep - [n_ch × n_time × n_train] preprocessed training epochs
%     y_train    - [n_train × 1] class labels
%     X_test_ep  - [n_ch × n_time × n_test] preprocessed test epochs
%     cfg        - struct with .csp_components, .csp_beta, .fs
%
%   Outputs:
%     feat_train - [n_train × (n_bands * n_csp_components)]
%     feat_test  - [n_test  × (n_bands * n_csp_components)]
%
%   LEAKAGE-SAFE: CSP computed per-band from X_train only.

    bands = [8 12; 12 16; 16 20; 20 24; 24 30];
    n_bands = size(bands, 1);

    n_components = cfg.csp_components;
    csp_beta     = cfg.csp_beta;
    fs           = cfg.fs;
    nyquist      = fs / 2;

    feat_train = [];
    feat_test  = [];

    for b = 1:n_bands
        band = bands(b, :);

        % Guard: skip bands that violate Nyquist
        if band(2) > nyquist
            continue;
        end

        % Design bandpass filter — lower order for narrow bands
        bw = band(2) - band(1);
        if bw < 5
            filt_order = 3;
        else
            filt_order = 4;
        end

        try
            [bf, af] = butter(filt_order, band / nyquist, 'bandpass');
        catch
            [bf, af] = butter(2, band / nyquist, 'bandpass');
        end

        % Filter epochs
        X_train_f = filter_epochs_fb(X_train_ep, bf, af);
        X_test_f  = filter_epochs_fb(X_test_ep,  bf, af);

        % Compute CSP from TRAINING data only
        try
            [W, ~] = compute_csp_projection( ...
                X_train_f, y_train, n_components, 'shrinkage', csp_beta);
        catch
            % CSP may fail for narrow bands with rank deficiency — skip band
            continue;
        end

        % Project through CSP filters
        Z_train = project_csp_fb(X_train_f, W);
        Z_test  = project_csp_fb(X_test_f,  W);

        % Log-variance features
        lv_train = extract_logvar_fb(Z_train);
        lv_test  = extract_logvar_fb(Z_test);

        feat_train = [feat_train, lv_train]; %#ok<AGROW>
        feat_test  = [feat_test,  lv_test];  %#ok<AGROW>
    end

    % Edge case: all bands failed
    if isempty(feat_train)
        feat_train = zeros(size(X_train_ep, 3), 0);
        feat_test  = zeros(size(X_test_ep,  3), 0);
        return;
    end

    % NaN / Inf guard
    feat_train(~isfinite(feat_train)) = 0;
    feat_test(~isfinite(feat_test))   = 0;
end


%% =========================================================================
%  LOCAL HELPERS (self-contained — no dependency on driver locals)
%  =========================================================================

function X_f = filter_epochs_fb(X, b, a)
%FILTER_EPOCHS_FB Bandpass filter each epoch [ch × time × epochs].
    X_f = zeros(size(X));
    for e = 1:size(X, 3)
        X_f(:, :, e) = filtfilt(b, a, X(:, :, e)')';
    end
end


function Z = project_csp_fb(X, W)
%PROJECT_CSP_FB Project epochs through CSP spatial filters.
    n_comp   = size(W, 2);
    n_time   = size(X, 2);
    n_epochs = size(X, 3);
    Z = zeros(n_comp, n_time, n_epochs);
    for e = 1:n_epochs
        Z(:, :, e) = W' * X(:, :, e);
    end
end


function features = extract_logvar_fb(Z)
%EXTRACT_LOGVAR_FB CSP log-normalised-variance features.
    n_epochs = size(Z, 3);
    n_comp   = size(Z, 1);
    features = zeros(n_epochs, n_comp);
    for e = 1:n_epochs
        v       = var(Z(:, :, e), 0, 2);
        total_v = sum(v) + eps;
        features(e, :) = log(v / total_v + eps)';
    end
end
