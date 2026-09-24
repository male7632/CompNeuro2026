clear;
clc;
close all;

rng(1);


mu = 1;
theta = 1;

sigma2Vals = [0.25 1 4];

dt = 5e-4;

Tburn = 50;        % discard initial transient
Tcollect = 2000;   % collect stationary statistics
Ttotal = Tburn + Tcollect;

nBurn = round(Tburn/dt);
nCollect = round(Tcollect/dt);
nTotal = nBurn + nCollect;

nCases = length(sigma2Vals);


rate_meas = zeros(nCases,1);
rate_theory = zeros(nCases,1);

fracBelow_meas = zeros(nCases,1);
fracBelow_theory = zeros(nCases,1);

for j = 1:nCases

    sigma2 = sigma2Vals(j);

    u = 0;

    U = zeros(nCollect,1);

    spikeCount = 0;
    sampleIndex = 0;

    for n = 1:nTotal

        u = u ...
            + mu*dt ...
            + sqrt(2*sigma2*dt)*randn;

        if u >= theta

            if n > nBurn
                spikeCount = spikeCount + 1;
            end

            u = 0;

        end

        if n > nBurn

            sampleIndex = sampleIndex + 1;
            U(sampleIndex) = u;

        end

    end



    rate_meas(j) = spikeCount / Tcollect;

    fracBelow_meas(j) = mean(U < 0);


    nu = mu/theta;

    rate_theory(j) = nu;

    fracBelow_theory(j) = ...
        sigma2/(mu*theta) * ...
        (1 - exp(-mu*theta/sigma2));


    uMin = min(U);

    x = linspace(uMin, theta, 1000);

    p = zeros(size(x));

    neg = x < 0;

    pos = (x >= 0) & (x <= theta);

    p(neg) = ...
        (nu/mu) .* ...
        (1 - exp(-mu*theta/sigma2)) .* ...
        exp(mu*x(neg)/sigma2);

    p(pos) = ...
        (nu/mu) .* ...
        (1 - exp(mu*(x(pos)-theta)/sigma2));



    figure;

    histogram(U, 120, ...
        'Normalization', 'pdf');

    hold on;

    plot(x, p, ...
        'LineWidth', 2);

    xlabel('Subthreshold voltage, u');
    ylabel('Probability density');

    title(sprintf( ...
        'Noisy Integrator: \\mu = 1, \\theta = 1, \\sigma^2 = %.2g', ...
        sigma2));

    legend('Simulation', ...
           'Stationary theory', ...
           'Location', 'best');

    grid on;

end


Results = table( ...
    sigma2Vals(:), ...
    rate_meas, ...
    rate_theory, ...
    fracBelow_meas, ...
    fracBelow_theory, ...
    'VariableNames', { ...
    'sigma2', ...
    'Rate_Measured', ...
    'Rate_Theory', ...
    'FractionBelow_Measured', ...
    'FractionBelow_Theory'});

disp(Results);