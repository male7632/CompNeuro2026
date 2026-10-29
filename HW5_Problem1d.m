%% APPM 5370 Homework 5 - Problem 1(d)


clear; clc; close all;

%% Model parameters
eta  = 4;
tauE = 5;      % ms
wEI  = 10;
wIE  = 10;
Ebar = 0.1;
Ibar = 0.1;

% From Problem 1(b), in units of 1/ms:
B = eta^2*wEI*wIE*Ebar*(1-Ebar)*Ibar*(1-Ibar)/tauE;

%% 1(d)(i): Change epsilon at tauI = 10 ms
tauI = 10;
epsList = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2];
periods = zeros(size(epsList));
amplitudes = zeros(size(epsList));
lags = zeros(size(epsList));

fprintf('\nPART (i): tauI = 10 ms\n');
fprintf(' epsilon    period (ms)    max(E)-min(E)    I lag (deg)\n');

for k = 1:numel(epsList)
    result = runWC(tauI, epsList(k), eta, tauE, wEI, wIE, Ebar, Ibar);
    periods(k) = result.period;
    amplitudes(k) = result.amplitude;
    lags(k) = result.lagDeg;
    fprintf(' %7.3f    %10.4f      %10.5f      %10.3f\n', ...
        epsList(k), periods(k), amplitudes(k), lags(k));
end

coeffs = polyfit(log(epsList), log(amplitudes), 1);
slope = coeffs(1);
fprintf('Log-log amplitude fit: slope = %.4f\n', slope);

figure(1);
loglog(epsList, amplitudes, 'ko', 'MarkerFaceColor', 'k', ...
    'MarkerSize', 7); hold on;
epsFine = logspace(log10(min(epsList)), log10(max(epsList)), 200);
fitAmp = exp(polyval(coeffs, log(epsFine)));
loglog(epsFine, fitAmp, 'b-', 'LineWidth', 1.6);
grid on;
xlabel('\epsilon');
ylabel('Oscillation amplitude: max(E) - min(E)');
title('Problem 1(d)(i): Amplitude scaling near Hopf');
legend('Simulated amplitudes', sprintf('Power-law fit, slope = %.3f', slope), ...
    'Location', 'northwest');
exportgraphics(gcf, 'P1d_amplitude_vs_epsilon.png', 'Resolution', 300);

%% 1(d)(ii): Change tauI at epsilon = 0.02
tauIList = [2, 3, 5, 7.5, 10, 15, 20];
epsilon = 0.02;
simFreq = zeros(size(tauIList));
linearFreq = zeros(size(tauIList));

fprintf('\nPART (ii): epsilon = 0.02\n');
fprintf(' tauI (ms)    simulated (Hz)    linear theory (Hz)    difference (%%)\n');
for k = 1:numel(tauIList)
    tauI = tauIList(k);
    result = runWC(tauI, epsilon, eta, tauE, wEI, wIE, Ebar, Ibar);
    simFreq(k) = 1000/result.period;
    omega = sqrt(B/tauI - 1/tauI^2);  % rad/ms, from Problem 1(b)
    linearFreq(k) = 1000*omega/(2*pi);
    pctDiff = 100*(simFreq(k)-linearFreq(k))/linearFreq(k);
    fprintf(' %8.1f       %10.4f        %10.4f        %+8.3f\n', ...
        tauI, simFreq(k), linearFreq(k), pctDiff);
end

figure(2);
plot(tauIList, simFreq, 'o-', 'LineWidth', 1.5, ...
    'MarkerSize', 6); hold on;
plot(tauIList, linearFreq, 's--', 'LineWidth', 1.5, ...
    'MarkerSize', 6);
grid on;
xlabel('\tau_I (ms)');
ylabel('Frequency (Hz)');
title('Problem 1(d)(ii): Simulated vs. linear frequency');
legend('Nonlinear simulation', 'Trace-zero linear theory', ...
    'Location', 'northeast');
exportgraphics(gcf, 'P1d_frequency_comparison.png', 'Resolution', 300);

%% 1(d)(iii): Time series for tauI = 10 ms, epsilon = 0.1
tauI = 10;
epsilon = 0.1;
result = runWC(tauI, epsilon, eta, tauE, wEI, wIE, Ebar, Ibar);

% Show approximately the final eight oscillations after transients.
keep = result.t >= result.t(end) - 8*result.period;
tPlot = result.t(keep);
tPlot = tPlot - tPlot(1);

figure(3);
plot(tPlot, result.y(keep,1), 'b-', 'LineWidth', 1.5); hold on;
plot(tPlot, result.y(keep,2), 'r-', 'LineWidth', 1.5);
grid on;
xlabel('Time after transient (ms)');
ylabel('Population activity');
title('Problem 1(d)(iii): E(t) and I(t), \tau_I=10 ms, \epsilon=0.1');
legend('E(t)', 'I(t)', 'Location', 'best');
exportgraphics(gcf, 'P1d_EI_time_series.png', 'Resolution', 300);

fprintf('\nPART (iii): tauI = 10 ms, epsilon = 0.1\n');
fprintf('Period = %.4f ms, frequency = %.4f Hz, peak lag = %.3f deg\n', ...
    result.period, 1000/result.period, result.lagDeg);


function result = runWC(tauI, epsilon, eta, tauE, wEI, wIE, Ebar, Ibar)
    fPrimeE = eta*Ebar*(1-Ebar);
    wEEhopf = (1 + tauE/tauI)/fPrimeE;
    wEE = (1 + epsilon)*wEEhopf;

    % Recalculate inputs to keep (Ebar, Ibar) as the equilibrium.
    invfE = log(Ebar/(1-Ebar))/eta;
    invfI = log(Ibar/(1-Ibar))/eta;
    pE = invfE - wEE*Ebar + wEI*Ibar;
    pI = invfI - wIE*Ebar;

    % Linear growth rate above Hopf; use it to allow longer transients
    % for very small epsilon.
    growthRate = epsilon*(1 + tauE/tauI)/(2*tauE); % 1/ms
    tEnd = max(5000, 8/growthRate);                  % ms
    dt = 0.2;                                       % output interval, ms
    tspan = 0:dt:tEnd;
    initialState = [Ebar+0.001; Ibar];
    opts = odeset('RelTol', 1e-9, 'AbsTol', 1e-11, 'MaxStep', 0.5);

    [t, y] = ode45(@(t,y) wcRHS(t, y, tauE, tauI, ...
        eta, wEE, wEI, wIE, pE, pI), tspan, initialState, opts);

    % Measure the final 25% of the integration, after transients.
    use = (t >= 0.75*t(end));
    ts = t(use);
    Es = y(use,1);
    Is = y(use,2);
    tEpeaks = localPeakTimes(ts, Es);
    tIpeaks = localPeakTimes(ts, Is);
    if numel(tEpeaks) < 5 || numel(tIpeaks) < 5
        error('Too few peaks detected: extend the simulation.');
    end

    period = mean(diff(tEpeaks));
    amplitude = max(Es)-min(Es);

    % For each excitatory peak, find the following inhibitory peak.
    % Quadratic peak interpolation reduces sampling-grid timing error.
    delays = [];
    for j = 1:numel(tEpeaks)
        nextI = find(tIpeaks > tEpeaks(j), 1, 'first');
        if ~isempty(nextI)
            lagTime = tIpeaks(nextI)-tEpeaks(j);
            if lagTime < 0.5*period
                delays(end+1) = lagTime; %#ok<AGROW>
            end
        end
    end
    if isempty(delays)
        error('No usable E-to-I peak pairs found.');
    end
    lagDeg = 360*mean(delays)/period;

    result = struct('period', period, 'amplitude', amplitude, ...
        'lagDeg', lagDeg, 't', t, 'y', y, 'wEE', wEE, 'pE', pE, 'pI', pI);
end

function dy = wcRHS(~, y, tauE, tauI, eta, wEE, wEI, wIE, pE, pI)
    E = y(1);
    I = y(2);
    inputE = wEE*E-wEI*I+pE;
    inputI = wIE*E+pI;
    fE = 1/(1+exp(-eta*inputE));
    fI = 1/(1+exp(-eta*inputI));
    dy = [(-E+fE)/tauE; (-I+fI)/tauI];
end

function peakT = localPeakTimes(t, y)
    % Local maxima by neighboring-sample comparison (no extra toolbox).
    k = find(y(2:end-1) >= y(1:end-2) & ...
             y(2:end-1) >  y(3:end)) + 1;
    peakT = zeros(numel(k),1);
    for n = 1:numel(k)
        j = k(n);
        denom = y(j-1)-2*y(j)+y(j+1);
        fractionalStep = 0;
        if abs(denom) > 1e-14
            fractionalStep = 0.5*(y(j-1)-y(j+1))/denom;
        end
        fractionalStep = min(1, max(-1, fractionalStep));
        peakT(n) = t(j) + fractionalStep*(t(j+1)-t(j));
    end
end
