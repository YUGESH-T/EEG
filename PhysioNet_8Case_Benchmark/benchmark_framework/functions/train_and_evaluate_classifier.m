function eval_result = train_and_evaluate_classifier(train_X, train_Y, test_X, test_Y, case_cfg)
switch lower(case_cfg.classifier_type)
    case 'svm'
        if isfield(case_cfg, 'feature_type') && strcmpi(string(case_cfg.feature_type), 'fbcsp_mi')
            model = fitcsvm(train_X, train_Y, 'KernelFunction', case_cfg.classifier.svm_kernel);
            pred = predict(model, test_X);
            [metrics, conf_mat] = compute_binary_metrics(test_Y, pred, case_cfg.eval.positive_label);
            eval_result = metrics;
            eval_result.conf_mat = conf_mat;
            return;
        end
        [train_X, test_X] = prepare_svm_feature_space(train_X, test_X, case_cfg);
        model = fitcsvm(train_X, train_Y, 'KernelFunction', case_cfg.classifier.svm_kernel);
    case 'rf'
        tree = templateTree('MinLeafSize', case_cfg.classifier.rf_min_leaf);
        model = fitcensemble(train_X, categorical(train_Y), 'Method', 'Bag', ...
            'Learners', tree, 'NumLearningCycles', case_cfg.classifier.rf_cycles);
    otherwise
        error('Unknown classifier_type: %s', case_cfg.classifier_type);
end

pred = predict(model, test_X);
if iscategorical(pred)
    pred = double(pred);
end

[metrics, conf_mat] = compute_binary_metrics(test_Y, pred, case_cfg.eval.positive_label);
eval_result = metrics;
eval_result.conf_mat = conf_mat;
end
