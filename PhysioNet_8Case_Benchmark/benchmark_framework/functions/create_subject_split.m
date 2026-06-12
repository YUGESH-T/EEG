function split = create_subject_split(labels, subject_num, cfg)
rng(cfg.seed + subject_num, 'twister');
cv = cvpartition(labels, 'HoldOut', cfg.eval.holdout);
split = struct();
split.train_idx = training(cv);
split.test_idx = test(cv);
end
