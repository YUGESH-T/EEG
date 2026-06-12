function features = extract_temporal_features(Z)
% EXTRACT_TEMPORAL_FEATURES Extracts temporal features from CSP-filtered signals.
%
%   features = extract_temporal_features(Z)
%
%   Inputs:
%     Z - [n_components × n_time × n_epochs] spatially filtered signals
%
%   Outputs:
%     features - [n_epochs × (n_components * 7)] temporal feature matrix.
%                Features: mean, variance, RMS, zero-crossing rate, 
%                Hjorth Activity, Mobility, Complexity.

    n_components = size(Z, 1);
    n_time = size(Z, 2);
    n_epochs = size(Z, 3);
    
    n_features_per_comp = 7;
    features = zeros(n_epochs, n_components * n_features_per_comp);
    
    for e = 1:n_epochs
        for c = 1:n_components
            signal = Z(c, :, e)';
            
            % 1. Mean
            f_mean = mean(signal);
            % 2. Variance
            f_var = var(signal);
            % 3. RMS
            f_rms = rms(signal);
            % 4. Zero-crossing rate
            f_zcr = sum(abs(diff(signal > 0))) / n_time;
            
            % Hjorth parameters
            d1 = diff(signal);
            d2 = diff(d1);
            
            % 5. Activity (variance of signal)
            f_activity = f_var;
            
            % 6. Mobility
            var_d1 = var(d1);
            if f_activity > 0
                f_mobility = sqrt(var_d1 / f_activity);
            else
                f_mobility = 0;
            end
            
            % 7. Complexity
            var_d2 = var(d2);
            if var_d1 > 0 && f_mobility > 0
                f_complexity = sqrt(var_d2 / var_d1) / f_mobility;
            else
                f_complexity = 0;
            end
            
            idx_start = (c - 1) * n_features_per_comp + 1;
            idx_end = c * n_features_per_comp;
            
            features(e, idx_start:idx_end) = [f_mean, f_var, f_rms, f_zcr, f_activity, f_mobility, f_complexity];
        end
    end
end
