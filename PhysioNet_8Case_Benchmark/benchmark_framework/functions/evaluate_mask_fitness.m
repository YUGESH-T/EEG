function error_val = evaluate_mask_fitness(position, X, Y, classifier_type, cfg, subject_num, cv)
if islogical(position)
    mask = position;
else
    mask = position > cfg.ho.threshold;
end
if ~any(mask)
    error_val = 1;
    return;
end
X_sel = X(:, mask);

if nargin < 7 || isempty(cv)
    rng(cfg.seed + 1000 + subject_num, 'twister');
    cv = cvpartition(Y, 'KFold', cfg.eval.ho_folds);
end

accs = zeros(1, cv.NumTestSets);
rf_tree = [];
if strcmpi(classifier_type, 'rf')
    rf_tree = templateTree('MinLeafSize', cfg.classifier.rf_min_leaf);
end

for k = 1:cv.NumTestSets
    X_train = X_sel(training(cv, k), :);
    Y_train = Y(training(cv, k));
    X_test = X_sel(test(cv, k), :);
    Y_test = Y(test(cv, k));
    switch lower(classifier_type)
        case 'svm'
            [X_train, X_test] = prepare_svm_feature_space(X_train, X_test, cfg);
            model = fitcsvm(X_train, Y_train, 'KernelFunction', cfg.classifier.svm_kernel);
        case 'rf'
            model = fitcensemble(X_train, categorical(Y_train), 'Method', 'Bag', ...
                'Learners', rf_tree, 'NumLearningCycles', cfg.classifier.rf_cycles);
        otherwise
            error('Unknown classifier_type: %s', classifier_type);
    end
    pred = predict(model, X_test);
    if iscategorical(pred)
        pred = double(pred);
    end
    accs(k) = mean(pred == Y_test);
end
error_val = 1 - mean(accs);
end
