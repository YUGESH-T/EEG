function case_cfgs = build_case_configs(cfg)
case_matrix = {
    'csp',        'svm', 'common'
    'riemannian', 'svm', 'common'
    'csp',        'rf',  'common'
    'riemannian', 'rf',  'common'
    'csp',        'svm', 'butterworth'
    'riemannian', 'svm', 'butterworth'
    'csp',        'rf',  'butterworth'
    'riemannian', 'rf',  'butterworth'
};

case_cfgs = repmat(cfg, 1, size(case_matrix, 1));
for idx = 1:size(case_matrix, 1)
    feature_type = case_matrix{idx, 1};
    classifier_type = case_matrix{idx, 2};
    filter_type = case_matrix{idx, 3};

    case_cfgs(idx).feature_type = feature_type;
    case_cfgs(idx).classifier_type = classifier_type;
    case_cfgs(idx).filter_type = filter_type;
    case_cfgs(idx).case_index = idx;
    case_cfgs(idx).case_id = sprintf('case%02d_%s_%s_%s', idx, feature_type, classifier_type, filter_type);
end
end
