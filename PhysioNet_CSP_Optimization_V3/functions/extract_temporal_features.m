function features = extract_temporal_features(epochs, fs, ar_order)
%EXTRACT_TEMPORAL_FEATURES Channel-level temporal features from EEG epochs.
%
%   features = extract_temporal_features(epochs, fs)
%   features = extract_temporal_features(epochs, fs, ar_order)
%
%   Computes per-channel:
%     1. Hjorth Activity   — variance of signal
%     2. Hjorth Mobility   — sqrt(var(diff(x)) / var(x))
%     3. Hjorth Complexity — mobility(diff(x)) / mobility(x)
%     4. RMS               — root mean square
%     5. Waveform Length   — sum(|diff(x)|)
%     6. Mean Abs Value    — mean(|x|)
%     7. AR coefficients   — Levinson-Durbin (order = ar_order)
%
%   Inputs:
%     epochs   - [n_channels × n_time × n_epochs] bandpassed EEG
%     fs       - sampling frequency (Hz), reserved for future use
%     ar_order - AR model order (default 4)
%
%   Output:
%     features - [n_epochs × n_channels*(6 + ar_order)]
%
%   No train-only fitting required (stateless extraction).

    if nargin < 2 || isempty(fs), fs = 160; end
    if nargin < 3 || isempty(ar_order), ar_order = 4; end

    n_channels = size(epochs, 1);
    n_time     = size(epochs, 2);
    n_epochs   = size(epochs, 3);
    n_base     = 6;   % Hjorth(3) + RMS + WL + MAV
    n_feat_ch  = n_base + ar_order;
    n_features = n_channels * n_feat_ch;
    features   = zeros(n_epochs, n_features);

    for e = 1:n_epochs
        for c = 1:n_channels
            x = double(epochs(c, :, e));
            x = x(:);

            dx  = diff(x);
            ddx = diff(dx);

            var_x   = var(x)   + eps;
            var_dx  = var(dx)  + eps;
            var_ddx = var(ddx) + eps;

            % --- Hjorth parameters ----------------------------------------
            activity   = var_x;
            mobility_x = sqrt(var_dx / var_x);
            mobility_dx = sqrt(var_ddx / var_dx);
            complexity = mobility_dx / (mobility_x + eps);

            % --- RMS ------------------------------------------------------
            rms_val = sqrt(mean(x.^2) + eps);

            % --- Waveform Length ------------------------------------------
            wl = sum(abs(dx));

            % --- Mean Absolute Value --------------------------------------
            mav = mean(abs(x));

            % --- AR coefficients (Levinson-Durbin) ------------------------
            ar_coeffs = levinson_durbin_ar(x, ar_order);

            % --- Pack into feature vector ---------------------------------
            offset = (c - 1) * n_feat_ch;
            features(e, offset + 1) = activity;
            features(e, offset + 2) = mobility_x;
            features(e, offset + 3) = complexity;
            features(e, offset + 4) = rms_val;
            features(e, offset + 5) = wl;
            features(e, offset + 6) = mav;
            features(e, offset + 7 : offset + 6 + ar_order) = ar_coeffs(:)';
        end
    end

    % NaN / Inf guard
    features(~isfinite(features)) = 0;
end


%% =========================================================================
%  AR ESTIMATION VIA LEVINSON-DURBIN (no Signal Processing Toolbox needed)
%  =========================================================================

function a = levinson_durbin_ar(x, order)
%LEVINSON_DURBIN_AR Autoregressive coefficients via Levinson-Durbin.
%   x     — column vector signal
%   order — AR model order
%   a     — [order × 1] AR coefficients (a1 … ap)
    x = x(:);
    n = length(x);

    if n <= order + 1
        a = zeros(order, 1);
        return;
    end

    % --- Biased autocorrelation -------------------------------------------
    r = zeros(order + 1, 1);
    for k = 0:order
        r(k + 1) = sum(x(1:n - k) .* x(k + 1:n)) / n;
    end

    if abs(r(1)) < eps
        a = zeros(order, 1);
        return;
    end

    % --- Levinson-Durbin recursion ----------------------------------------
    a = zeros(order, 1);
    e = r(1);

    for m = 1:order
        lambda = r(m + 1);
        for j = 1:m - 1
            lambda = lambda - a(j) * r(m + 1 - j);
        end

        if abs(e) < eps
            break;
        end

        lambda = lambda / e;

        a_prev = a;
        a(m)   = lambda;
        for j = 1:m - 1
            a(j) = a_prev(j) - lambda * a_prev(m - j);
        end

        e = e * (1 - lambda^2);
        if e <= 0
            break;
        end
    end
end
