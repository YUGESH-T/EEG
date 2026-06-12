function fused_features = fuse_feature_blocks(feature_blocks)
% FUSE_FEATURE_BLOCKS Concatenates multiple feature blocks and handles NaNs.
%
%   fused_features = fuse_feature_blocks(feature_blocks)
%
%   Inputs:
%     feature_blocks - Cell array of [n_epochs × n_features_i] matrices
%
%   Outputs:
%     fused_features - [n_epochs × sum(n_features_i)] matrix with NaNs/Infs
%                      replaced by 0.

    if isempty(feature_blocks)
        fused_features = [];
        return;
    end
    
    fused_features = horzcat(feature_blocks{:});
    
    % Handle NaNs and Infs
    bad_mask = ~isfinite(fused_features);
    if any(bad_mask(:))
        warning('fuse_feature_blocks:nanInf', ...
            '%d NaN/Inf values detected in fused features. Replacing with 0.', sum(bad_mask(:)));
        fused_features(bad_mask) = 0;
    end
end
