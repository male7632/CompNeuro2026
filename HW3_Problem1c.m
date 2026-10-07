%% Problem 1(c): Periodically Forced LIF Model

clear;
clc;
close all;

%% Parameters

epsilon = 0.5;

Ivalues = linspace(0.9, 3.0, 500);

nTransientPeriods = 100;
nMeasurePeriods   = 200;

firingNumber = zeros(size(Ivalues));

%% Sweep over I

for k = 1:length(Ivalues)

    I = Ivalues(k);

    % Start at reset at t = 0
    tSpike = 0;
    spikeTimes = [];

    tEnd = nTransientPeriods + nMeasurePeriods;

    % Generate spikes until the end of the simulation
    while tSpike < tEnd

        tNext = findNextSpike(tSpike, I, epsilon);

        % If no more spike is found, stop
        if isnan(tNext)
            break
        end

        spikeTimes(end+1) = tNext; %#ok<SAGROW>
        tSpike = tNext;

    end

    %% Count spikes after transient

    measurementStart = nTransientPeriods;
    measurementEnd   = nTransientPeriods + nMeasurePeriods;

    measuredSpikes = spikeTimes( ...
        spikeTimes >= measurementStart & ...
        spikeTimes < measurementEnd);

    firingNumber(k) = length(measuredSpikes) / nMeasurePeriods;

end

%% Unforced firing number

unforcedNumber = zeros(size(Ivalues));

mask = Ivalues > 1;

unforcedNumber(mask) = ...
    1 ./ log(Ivalues(mask) ./ (Ivalues(mask) - 1));

%% 1:1 tongue boundaries

Istar = 1 / (1 - exp(-1));

tongueHalfWidth = epsilon / sqrt(1 + 4*pi^2);

Ileft  = Istar - tongueHalfWidth;
Iright = Istar + tongueHalfWidth;

fprintf('I* = %.4f\n', Istar);
fprintf('1:1 tongue boundaries: %.4f and %.4f\n', ...
    Ileft, Iright);

%% Plot firing number

figure;

plot(Ivalues, firingNumber, 'LineWidth', 1.5);
hold on;

plot(Ivalues, unforcedNumber, '--', 'LineWidth', 1.5);

xline(Ileft, ':', '1:1 boundary', ...
    'LabelVerticalAlignment', 'bottom');

xline(Iright, ':', '1:1 boundary', ...
    'LabelVerticalAlignment', 'bottom');

xlabel('I');
ylabel('Spikes per forcing period');

title('\epsilon = 0.5 Periodically Forced LIF Model');

legend('Forced model', ...
       'Unforced model', ...
       'Location', 'northwest');

grid on;

%% Spike phase at I = 1.6

I = 1.6;

tSpike = 0;
spikeTimes = [];

tEnd = 150;

while tSpike < tEnd

    tNext = findNextSpike(tSpike, I, epsilon);

    if isnan(tNext)
        break
    end

    spikeTimes(end+1) = tNext; %#ok<SAGROW>
    tSpike = tNext;

end

% Remove transient spikes
steadySpikes = spikeTimes(spikeTimes >= 100);

% Convert spike times to forcing phase
phases = mod(steadySpikes, 1);

fprintf('\nSpike phases at I = 1.6:\n');
disp(phases);

fprintf('Mean phase = %.4f\n', mean(phases));

%% ---------------------------------------------------------------
% Local helper function
% ---------------------------------------------------------------

function tNext = findNextSpike(t0, I, epsilon)

    % Define G(t)
    G = @(t) I + epsilon/(1 + 4*pi^2) .* ...
        (sin(2*pi*t) - 2*pi*cos(2*pi*t));

    % Exact voltage trajectory after reset at t0
    %
    % v(t) = G(t) + [v(t0) - G(t0)] exp(-(t-t0))
    %
    % Since v(t0) = 0 after reset:
    %
    % v(t) = G(t) - G(t0) exp(-(t-t0))

    v = @(t) G(t) - G(t0).*exp(-(t - t0));

    % Threshold equation
    F = @(t) v(t) - 1;

    %% Search forward for the FIRST threshold crossing

    dtSearch = 0.002;

    t1 = t0 + 1e-8;
    f1 = F(t1);

    % Search up to 20 forcing periods ahead
    maxSearchTime = t0 + 20;

    tNext = NaN;

    while t1 < maxSearchTime

        t2 = t1 + dtSearch;
        f2 = F(t2);

        % Detect an upward crossing of v = 1
        if f1 < 0 && f2 >= 0

            % Refine crossing time accurately
            tNext = fzero(F, [t1 t2]);
            return

        end

        t1 = t2;
        f1 = f2;

    end
end