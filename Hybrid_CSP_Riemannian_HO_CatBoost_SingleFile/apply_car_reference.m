function data = apply_car_reference(data)
% APPLY_CAR_REFERENCE  Common Average Reference preprocessing for EEG.
%
%   data = apply_car_reference(data)
%
%   Input:
%     data  - [channels x samples] EEG matrix (double).
%
%   Output:
%     data  - [channels x samples] CAR-referenced EEG matrix.
%             Dimensions are guaranteed unchanged.
%
%   For each time point, subtracts the mean voltage across all channels.
%   This removes global/volume-conducted artifacts that are common to every
%   electrode, improving the spatial specificity of CSP filters.
%
%   Must be called AFTER bandpass filtering and BEFORE epoching/CSP.

    if isempty(data)
        return;
    end

    dims_before = size(data);

    % CAR: subtract channel-wise mean at every time point
    data = data - mean(data, 1);

    % Dimension integrity check
    dims_after = size(data);
    if ~isequal(dims_before, dims_after)
        error('apply_car_reference:dimMismatch', ...
            'CAR changed data dimensions from [%s] to [%s].', ...
            num2str(dims_before), num2str(dims_after));
    end

    fprintf('    CAR preprocessing applied\n');
end
