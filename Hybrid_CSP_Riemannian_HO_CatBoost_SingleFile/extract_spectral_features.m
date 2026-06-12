function features = extract_spectral_features(Z, fs)
% EXTRACT_SPECTRAL_FEATURES Extracts spectral features from CSP-filtered signals.
%
%   features = extract_spectral_features(Z, fs)
%
%   Inputs:
%     Z  - [n_components × n_time × n_epochs] spatially filtered signals
%     fs - Sampling frequency
%
%   Outputs:
%     features - [n_epochs × (n_components * 3)] spectral feature matrix.
%                Bands: 8-12 Hz, 12-20 Hz, 20-30 Hz (log bandpower).

    n_components = size(Z, 1);
    n_epochs = size(Z, 3);
    
    bands = [8 12; 12 20; 20 30];
    n_bands = size(bands, 1);
    
    features = zeros(n_epochs, n_components * n_bands);
    
    for e = 1:n_epochs
        for c = 1:n_components
            signal = Z(c, :, e)';
            
            % Use bandpower for each band
            for b = 1:n_bands
                bp = bandpower(signal, fs, bands(b, :));
                idx = (c - 1) * n_bands + b;
                features(e, idx) = log(bp + eps);
            end
        end
    end
end
