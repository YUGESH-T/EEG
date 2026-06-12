% Baseline test script
disp('=== Cho2017 Smoke CSP=14 ===');
cfg = struct();
cfg.features.csp_components = 14;
cfg.features.use_spectral = false;
cfg.features.use_temporal = false;
cfg.features.use_ho = false;
res = plain_csp_catboostcopy('cho_smoke', cfg);

disp('=== Cho2017 Smoke CSP=16 ===');
cfg.features.csp_components = 16;
res = plain_csp_catboostcopy('cho_smoke', cfg);

disp('=== PhysioNet Smoke CSP=14 ===');
cfg.features.csp_components = 14;
res = plain_csp_catboostcopy('smoke', cfg);

disp('=== PhysioNet Smoke CSP=16 ===');
cfg.features.csp_components = 16;
res = plain_csp_catboostcopy('smoke', cfg);

disp('All smoke tests finished.');
