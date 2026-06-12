% Pilot baseline test script (forcing recomputation)
cfg = struct();
cfg.results.resume = false;
cfg.features.use_spectral = false;
cfg.features.use_temporal = false;
cfg.features.use_ho = false;

disp('=== Cho2017 Pilot5 CSP=14 ===');
cfg.features.csp_components = 14;
res = plain_csp_catboostcopy('cho_pilot5', cfg);

disp('=== Cho2017 Pilot5 CSP=16 ===');
cfg.features.csp_components = 16;
res = plain_csp_catboostcopy('cho_pilot5', cfg);

disp('=== PhysioNet Pilot10 CSP=14 ===');
cfg.features.csp_components = 14;
res = plain_csp_catboostcopy('pilot10', cfg);

disp('=== PhysioNet Pilot10 CSP=16 ===');
cfg.features.csp_components = 16;
res = plain_csp_catboostcopy('pilot10', cfg);

disp('All pilot tests finished.');
