clear;
clc;
close all;

rng(1);                    

tau = 0.020;              
muTarget = 10;           
qVals = [2 20 200];       

dt = 0.0002;              
Tburn = 2;                
Tcollect = 200;           
Ttotal = Tburn + Tcollect;

nCases = length(qVals);

mean_meas = zeros(nCases,1);
var_meas  = zeros(nCases,1);
skew_meas = zeros(nCases,1);

mean_theory = zeros(nCases,1);
var_theory  = zeros(nCases,1);
skew_theory = zeros(nCases,1);

nuVals = zeros(nCases,1);
wVals  = zeros(nCases,1);

Vsave = cell(nCases,1);

for j = 1:nCases

    q = qVals(j);

  
    nu = q/tau;


    w = muTarget/q;

    nuVals(j) = nu;
    wVals(j) = w;

    V = simulateShotNoise(nu, w, tau, Ttotal, dt);

    i0 = floor(Tburn/dt) + 1;
    Vss = V(i0:end);

    Vsave{j} = Vss;

    m = mean(Vss);
    centered = Vss - m;

    mean_meas(j) = m;
    var_meas(j) = mean(centered.^2);
    skew_meas(j) = mean(centered.^3) / ...
                   var_meas(j)^(3/2);

    mean_theory(j) = nu*w*tau;
    var_theory(j) = nu*w^2*tau/2;

    skew_theory(j) = ...
        2*sqrt(2)/(3*sqrt(nu*tau));

end


Results = table( ...
    qVals(:), ...
    nuVals, ...
    wVals, ...
    mean_meas, ...
    mean_theory, ...
    var_meas, ...
    var_theory, ...
    skew_meas, ...
    skew_theory, ...
    'VariableNames', { ...
    'nuTau', ...
    'nu_Hz', ...
    'w_mV', ...
    'Mean_Measured', ...
    'Mean_Theory', ...
    'Variance_Measured', ...
    'Variance_Theory', ...
    'Skewness_Measured', ...
    'Skewness_Theory'});

disp(Results);


j = 1;

mu = mean_theory(j);
sig2 = var_theory(j);
sig = sqrt(sig2);

x = linspace(mu - 4*sig, mu + 5*sig, 500);

gaussianPDF = ...
    1./sqrt(2*pi*sig2) .* ...
    exp(-(x-mu).^2./(2*sig2));

figure;

histogram(Vsave{j}, 100, ...
    'Normalization', 'pdf');

hold on;

plot(x, gaussianPDF, ...
    'LineWidth', 2);

xlabel('Voltage, V (mV)');
ylabel('Probability density (1/mV)');

title(sprintf( ...
    'Shot Noise: \\nu\\tau = %g, \\nu = %.0f Hz, w = %.2f mV', ...
    qVals(j), nuVals(j), wVals(j)));

legend('Simulated shot noise', ...
       'Matched Gaussian', ...
       'Location', 'best');

grid on;



j = 3;

mu = mean_theory(j);
sig2 = var_theory(j);
sig = sqrt(sig2);

x = linspace(mu - 4*sig, mu + 4*sig, 500);

gaussianPDF = ...
    1./sqrt(2*pi*sig2) .* ...
    exp(-(x-mu).^2./(2*sig2));

figure;

histogram(Vsave{j}, 100, ...
    'Normalization', 'pdf');

hold on;

plot(x, gaussianPDF, ...
    'LineWidth', 2);

xlabel('Voltage, V (mV)');
ylabel('Probability density (1/mV)');

title(sprintf( ...
    'Shot Noise: \\nu\\tau = %g, \\nu = %.0f Hz, w = %.3f mV', ...
    qVals(j), nuVals(j), wVals(j)));

legend('Simulated shot noise', ...
       'Matched Gaussian', ...
       'Location', 'best');

grid on;


function V = simulateShotNoise(nu, w, tau, T, dt)

    N = floor(T/dt);
    Tend = N*dt;

    expectedEvents = nu*Tend;

    M = ceil(expectedEvents ...
        + 10*sqrt(expectedEvents) + 100);

    isi = -log(rand(M,1))/nu;
    eventTimes = cumsum(isi);

    while eventTimes(end) < Tend

        Mextra = max(1000, ceil(0.05*M));

        extraISI = -log(rand(Mextra,1))/nu;

        extraTimes = ...
            eventTimes(end) + cumsum(extraISI);

        eventTimes = [eventTimes; extraTimes];

    end

    eventTimes = eventTimes(eventTimes <= Tend);
    bin = ceil(eventTimes/dt);

    sampleTime = bin*dt;

    eventContribution = ...
        w .* exp(-(sampleTime-eventTimes)/tau);

    input = accumarray( ...
        bin, ...
        eventContribution, ...
        [N 1], ...
        @sum, ...
        0);

    r = exp(-dt/tau);

    V = filter(1, [1 -r], input);

end