function y_pred = classify_catboost(X_train, y_train, X_test, cfg)
%CLASSIFY_CATBOOST CatBoost classifier with MATLAB ensemble fallback.
%
%   y_pred = classify_catboost(X_train, y_train, X_test, cfg)
%
%   Uses Python CatBoost bridge. Falls back to MATLAB AdaBoostM1 if the
%   Python bridge is unavailable or fails.
%
%   cfg fields:
%     cfg.catboost.iterations    - boosting rounds (default 200)
%     cfg.catboost.depth         - tree depth (default 6)
%     cfg.catboost.learning_rate - learning rate (default 0.08)
%     cfg.seed                   - random seed (default 42)

    iterations    = 200;
    depth         = 6;
    learning_rate = 0.08;
    seed_val      = 42;

    if isfield(cfg, 'catboost')
        if isfield(cfg.catboost, 'iterations'),    iterations    = cfg.catboost.iterations; end
        if isfield(cfg.catboost, 'depth'),         depth         = cfg.catboost.depth; end
        if isfield(cfg.catboost, 'learning_rate'), learning_rate = cfg.catboost.learning_rate; end
    end
    if isfield(cfg, 'seed'), seed_val = cfg.seed; end

    % --- NaN/Inf guard on inputs -----------------------------------------
    X_train(~isfinite(X_train)) = 0;
    X_test(~isfinite(X_test))   = 0;

    try
        catboost_mod = py.importlib.import_module('catboost');
        np           = py.importlib.import_module('numpy');

        label_offset = min(y_train);
        y_zero       = y_train(:) - label_offset;

        model = catboost_mod.CatBoostClassifier(pyargs( ...
            'iterations',    int32(iterations), ...
            'depth',         int32(depth), ...
            'learning_rate', learning_rate, ...
            'loss_function', 'Logloss', ...
            'verbose',       false, ...
            'random_seed',   int32(seed_val)));

        model.fit(np.array(X_train), np.array(y_zero));

        py_pred = model.predict(np.array(X_test));
        y_pred  = double(py_pred);
        y_pred  = y_pred(:) + label_offset;

    catch ME
        % --- MATLAB fallback: AdaBoostM1 --------------------------------
        fprintf('    CatBoost unavailable (%s). Using AdaBoostM1.\n', ...
            truncate_msg(ME.message, 60));
        try
            t   = templateTree('MaxNumSplits', 20);
            mdl = fitcensemble(X_train, y_train, ...
                'Method',            'AdaBoostM1', ...
                'Learners',          t, ...
                'NumLearningCycles', min(iterations, 100));
        catch
            mdl = fitctree(X_train, y_train, 'MaxNumSplits', 20);
        end
        y_pred = predict(mdl, X_test);
    end

    y_pred = y_pred(:);
end

function msg = truncate_msg(msg, max_len)
    msg = regexprep(msg, '[\r\n]+', ' ');
    if numel(msg) > max_len
        msg = [msg(1:max_len-3) '...'];
    end
end
