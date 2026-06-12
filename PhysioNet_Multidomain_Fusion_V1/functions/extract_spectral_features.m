function feats = extract_spectral_features(epochs, fs)
%EXTRACT_SPECTRAL_FEATURES Channel-level spectral features via Welch PSD.
%
%   feats = extract_spectral_features(epochs, fs)
%
%   Computes per-channel using Welch's method:
%     1. Broadband PSD  — log mean power in 8-30 Hz
%     2. Mu-band power  — log mean power in 8-12 Hz
%     3. Beta-band power — log mean power in 13-30 Hz
%     4. Spectral entropy — Shannon entropy of normalised PSD
%
%   Inputs:
%     epochs - [n_channels × n_time × n_epochs] bandpassed EEG
%     fs     - sampling frequency (Hz)
%
%   Output struct fields (each [n_epochs × n_channels]):
%     feats.psd     — broadband log-power
%     feats.mu      — mu-band log-power
%     feats.beta    — beta-band log-power
%     feats.entropy — spectral entropy
%
%   No train-only fitting required (stateless extraction).

    if nargin < 2 || isempty(fs), fs = 160; end

    n_channels = size(epochs, 1);
    n_time     = size(epochs, 2);
    n_epochs   = size(epochs, 3);

    feats.psd     = zeros(n_epochs, n_channels);
    feats.mu      = zeros(n_epochs, n_channels);
    feats.beta    = zeros(n_epochs, n_channels);
    feats.entropy = zeros(n_epochs, n_channels);

    % Welch parameters — power-of-two FFT length, 50 % overlap
    nfft = min(256, 2^floor(log2(n_time)));
    if nfft < 16
        nfft = n_time;
    end
    win      = hamming(nfft);
    noverlap = floor(nfft / 2);

    for e = 1:n_epochs
        for c = 1:n_channels
            signal = double(epochs(c, :, e));
            signal = signal(:);

            % --- Welch PSD estimate ---------------------------------------
            [pxx, f] = welch_safe(signal, win, noverlap, nfft, fs);
            pxx = pxx + eps;   % guard log(0)

            % --- Broadband 8-30 Hz ----------------------------------------
            bb_mask = f >= 8 & f <= 30;
            if any(bb_mask)
                feats.psd(e, c) = log(mean(pxx(bb_mask)));
            end

            % --- Mu band 8-12 Hz ------------------------------------------
            mu_mask = f >= 8 & f <= 12;
            if any(mu_mask)
                feats.mu(e, c) = log(mean(pxx(mu_mask)));
            end

            % --- Beta band 13-30 Hz ---------------------------------------
            beta_mask = f >= 13 & f <= 30;
            if any(beta_mask)
                feats.beta(e, c) = log(mean(pxx(beta_mask)));
            end

            % --- Spectral entropy -----------------------------------------
            pxx_norm = pxx / (sum(pxx) + eps);
            feats.entropy(e, c) = -sum(pxx_norm .* log(pxx_norm + eps));
        end
    end

    % NaN / Inf guard
    feats.psd(~isfinite(feats.psd))         = 0;
    feats.mu(~isfinite(feats.mu))           = 0;
    feats.beta(~isfinite(feats.beta))       = 0;
    feats.entropy(~isfinite(feats.entropy)) = 0;
end


%% =========================================================================
%  SAFE WELCH WRAPPER
%  =========================================================================

function [pxx, f] = welch_safe(signal, win, noverlap, nfft, fs)
%WELCH_SAFE pwelch with periodogram fallback.
    try
        [pxx, f] = pwelch(signal, win, noverlap, nfft, fs);
    catch
        % Fallback: manual periodogram
        n   = length(signal);
        w   = hamming(n);
        xw  = signal .* w;
        X   = fft(xw, nfft);
        pxx = (abs(X).^2) / (fs * n);
        pxx = pxx(1:floor(nfft / 2) + 1);
        f   = (0:floor(nfft / 2))' * fs / nfft;
        % One-sided scaling (double non-DC/Nyquist bins)
        pxx(2:end-1) = 2 * pxx(2:end-1);
    end
end
