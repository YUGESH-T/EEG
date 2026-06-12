function features = extract_ho_features(Z)
% EXTRACT_HO_FEATURES Extracts higher-order features from CSP-filtered signals.
%
%   features = extract_ho_features(Z)
%
%   Inputs:
%     Z - [n_components × n_time × n_epochs] spatially filtered signals
%
%   Outputs:
%     features - [n_epochs × (n_components * 2)] HO feature matrix.
%                Features: skewness, kurtosis.

    n_components = size(Z, 1);
    n_epochs = size(Z, 3);
    
    n_features_per_comp = 2;
    features = zeros(n_epochs, n_components * n_features_per_comp);
    
    for e = 1:n_epochs
        for c = 1:n_components
            signal = Z(c, :, e)';
            
            f_skew = skewness(signal);
            f_kurt = kurtosis(signal);
            
            idx_start = (c - 1) * n_features_per_comp + 1;
            idx_end = c * n_features_per_comp;
            
            features(e, idx_start:idx_end) = [f_skew, f_kurt];
        end
    end
end
