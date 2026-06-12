function [best_score, best_pos, ho_curve] = hippopotamus_optimization(dim, fitness_fn, cfg)
SearchAgents = cfg.ho.agents;
Max_iterations = cfg.ho.iterations;
lb = zeros(1, dim);
ub = ones(1, dim);

X = lb + rand(SearchAgents, dim) .* (ub - lb);
fit = zeros(1, SearchAgents);
for i = 1:SearchAgents
    fit(i) = fitness_fn(X(i, :));
end

fbest = inf;
Xbest = X(1, :);
ho_curve = zeros(1, Max_iterations);
for t = 1:Max_iterations
    [best, loc] = min(fit);
    if t == 1 || best < fbest
        fbest = best;
        Xbest = X(loc, :);
    end
    for i = 1:SearchAgents
        r = rand(1, dim);
        X_new = X(i, :) + r .* (Xbest - X(i, :));
        X_new = min(max(X_new, lb), ub);
        f_new = fitness_fn(X_new);
        if f_new < fit(i)
            X(i, :) = X_new;
            fit(i) = f_new;
        end
    end
    ho_curve(t) = fbest;
end

best_score = fbest;
best_pos = Xbest;
end
