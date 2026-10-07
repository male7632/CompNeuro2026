%% Problem 4(c): Gap-Junction Coupled QIF Neurons

clear;
clc;
close all;

%% Parameters

L = 200;
phi0 = 0.4;

epsilonValues = [0.005, 0.02, 0.05];

% Long enough to measure the slowest decay
tEnd = 400;

fitRates = zeros(size(epsilonValues));

%% ============================================================
% Simulate each coupling strength
% ============================================================

for m = 1:length(epsilonValues)

    epsilon = epsilonValues(m);

    fprintf('\n----------------------------------------\n');
    fprintf('epsilon = %.4f\n', epsilon);

    %% Initial conditions

    % Neuron 1 has just fired and is at reset
    v1_0 = -L;

    % Finite-cutoff uncoupled period
    Tcut = 2*atan(L);

    % Neuron 2 is phi0 of a cycle behind neuron 1.
    % Therefore it has phi0*Tcut remaining until its next spike.
    %
    % State on the finite-cutoff uncoupled trajectory:
    v2_0 = tan((1 - 2*phi0)*atan(L));

    x0 = [v1_0; v2_0];

    %% Simulate hybrid voltage system

    [spikes1, spikes2] = simulateQIF( ...
        epsilon, L, tEnd, x0);

    % Neuron 1 fired at t = 0 before being reset,
    % so include t = 0 as its first spike time.
    spikes1 = [0; spikes1(:)];
    spikes2 = spikes2(:);

    %% Measure lag once per neuron-1 cycle

    lagTimes = [];
    phiLag = [];

    for k = 1:length(spikes1)-1

        t1 = spikes1(k);
        t1next = spikes1(k+1);

        % Find neuron-2 spike occurring during this neuron-1 cycle
        idx = find( ...
            spikes2 >= t1 & spikes2 < t1next, ...
            1, 'first');

        if ~isempty(idx)

            t2 = spikes2(idx);

            phi_k = (t2 - t1)/(t1next - t1);

            % Keep valid lags
            if phi_k > 0 && phi_k < 0.5
                lagTimes(end+1) = t1; %#ok<SAGROW>
                phiLag(end+1) = phi_k; %#ok<SAGROW>
            end

        end
    end

    %% ========================================================
    % Fit decay rate of tan(pi*phi_k)
    % =========================================================

    y = tan(pi*phiLag);

    % Avoid points extremely close to synchrony, where numerical
    % spike-time errors become important.
    fitMask = ...
        phiLag > 0.03 & ...
        phiLag < 0.38 & ...
        isfinite(y) & ...
        y > 0;

    tFit = lagTimes(fitMask);
    logY = log(y(fitMask));

    if length(tFit) >= 3

        p = polyfit(tFit, logY, 1);

        % log(tan(pi*phi)) = constant - rate*t
        fitRates(m) = -p(1);

    else

        fitRates(m) = NaN;

    end

    fprintf('Fitted decay rate = %.5f\n', fitRates(m));
    fprintf('Theory 2*epsilon  = %.5f\n', 2*epsilon);

    %% ========================================================
    % Plot lag vs theory for epsilon = 0.02
    % =========================================================

    if abs(epsilon - 0.02) < 1e-12

        phiTheory = (1/pi) * atan( ...
            tan(pi*phi0) .* exp(-2*epsilon*lagTimes));

        figure;

        plot(lagTimes, phiLag, 'o', ...
            'MarkerSize', 4);

        hold on;

        plot(lagTimes, phiTheory, ...
            'LineWidth', 1.5);

        xlabel('Time');
        ylabel('Phase lag \phi');

        title( ...
            'Gap-Junction QIF Synchronization, \epsilon = 0.02');

        legend( ...
            'Measured voltage simulation', ...
            'Phase-model prediction', ...
            'Location', 'northeast');

        grid on;

    end

end

%% ============================================================
% Display decay-rate comparison
% ============================================================

fprintf('\n========================================\n');
fprintf('Decay Rate Comparison\n');
fprintf('========================================\n');

fprintf(' epsilon      fitted rate      2*epsilon\n');

for m = 1:length(epsilonValues)

    fprintf(' %.4f       %.5f          %.5f\n', ...
        epsilonValues(m), ...
        fitRates(m), ...
        2*epsilonValues(m));

end

%% ============================================================
% Plot fitted rates against theoretical rates
% ============================================================

figure;

plot(2*epsilonValues, fitRates, ...
    'o-', ...
    'LineWidth', 1.5, ...
    'MarkerSize', 7);

hold on;

plot(2*epsilonValues, 2*epsilonValues, ...
    '--', ...
    'LineWidth', 1.5);

xlabel('Theoretical decay rate 2\epsilon');
ylabel('Fitted decay rate');

title('Measured vs Predicted Synchronization Rate');

legend( ...
    'Voltage simulation', ...
    'Perfect agreement', ...
    'Location', 'northwest');

grid on;


%% ============================================================
% Local function: simulate coupled QIF neurons
% ============================================================

function [spikes1, spikes2] = simulateQIF( ...
    epsilon, L, tEnd, x0)

    tCurrent = 0;
    xCurrent = x0;

    spikes1 = [];
    spikes2 = [];

    % Coupled QIF equations
    qif = @(t,x) [ ...
        1 + x(1)^2 + epsilon*(x(2) - x(1)); ...
        1 + x(2)^2 + epsilon*(x(1) - x(2)) ];

    opts = odeset( ...
        'RelTol', 1e-9, ...
        'AbsTol', 1e-11, ...
        'MaxStep', 0.02, ...
        'Events', @(t,x) spikeEvent(t,x,L));

    while tCurrent < tEnd

        [~,x,te,xe,ie] = ode45( ...
            qif, ...
            [tCurrent tEnd], ...
            xCurrent, ...
            opts);

        % No more threshold crossings
        if isempty(te)
            break
        end

        % Event time and state
        tEvent = te(end);
        xEvent = xe(end,:)';

        % Which neuron fired?
        fired = unique(ie);

        if any(fired == 1)
            spikes1(end+1,1) = tEvent; %#ok<AGROW>
            xEvent(1) = -L;
        end

        if any(fired == 2)
            spikes2(end+1,1) = tEvent; %#ok<AGROW>
            xEvent(2) = -L;
        end

        % Move slightly beyond event time to prevent immediate
        % repeated event detection.
        tCurrent = tEvent + 1e-10;
        xCurrent = xEvent;

    end

end


%% ============================================================
% Local function: threshold events
% ============================================================

function [value,isterminal,direction] = spikeEvent(~,x,L)

    % Threshold is +L for both neurons
    value = [x(1) - L;
             x(2) - L];

    % Stop integration whenever either neuron spikes
    isterminal = [1; 1];

    % Only upward threshold crossings
    direction = [1; 1];

end