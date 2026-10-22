
%% APPM 4370/5370 - Homework 4
% Problem 1(c): Ocular Dominance Stripes

clear;
clc;
close all;

%% Simulation parameters

N = 200;                   % Number of cortical sites
sigma_e = 1;               % Excitatory width
sigma_i_values = [2, 3];   % Inhibitory widths

dt = 0.02;                 % Time step
t_final = 40;              % Final simulation time
numSteps = round(t_final/dt);

numTrials = 20;            % Initial conditions per case
initialStd = 0.01;         % Initial weight standard deviation

rng(4370);                 % Reproducible results

%% Initialize arrays

positions = 0:N-1;

stripePeriods = NaN(numTrials, 2);
finalPatterns = zeros(2, N);

theoreticalPeriods = zeros(1, 2);
discretePeriods = zeros(1, 2);
fastestModes = zeros(1, 2);

%% Run simulations

for j = 1:length(sigma_i_values)

    sigma_i = sigma_i_values(j);

    % Periodic distance on ring
    distance = min(positions, N - positions);

    % Mexican-hat interaction kernel
    K = exp(-distance.^2/(2*sigma_e^2)) ...
        - (sigma_e/sigma_i) ...
        * exp(-distance.^2/(2*sigma_i^2));

    % Fourier transform of the interaction kernel
    Khat = real(fft(K));

    %% Continuous theoretical prediction

    k_star = sqrt( ...
        4*log(sigma_i/sigma_e) / ...
        (sigma_i^2 - sigma_e^2));

    theoreticalPeriods(j) = 2*pi/k_star;

    %% Fastest-growing Fourier mode on the ring

    % Only consider positive, nonzero frequencies
    [~, modeIndex] = max(Khat(2:N/2+1));

    % MATLAB index offset gives Fourier mode number
    n_star = modeIndex;

    fastestModes(j) = n_star;

    % Period of that discrete Fourier mode
    discretePeriods(j) = N/n_star;

    %% Simulate 20 random initial conditions

    for trial = 1:numTrials

        % Random initial ocular dominance weights
        w = initialStd * randn(1, N);

        % Forward Euler time integration
        for step = 1:numSteps

            % Periodic convolution using FFT
            convolution = real(ifft(Khat .* fft(w)));

            % Euler update
            w = w + dt * convolution;

            % Clip weights to [-1, 1]
            w = max(-1, min(1, w));

        end

        %% Measure stripe period

        % Right-eye-dominant sites have positive w
        rightEye = (w > 0);

        % Count sign transitions around the entire ring
        transitions = sum( ...
            rightEye ~= circshift(rightEye, [0, 1]));

        % Two transitions correspond to one R/L pair
        numPairs = transitions / 2;

        if numPairs > 0
            stripePeriods(trial, j) = N/numPairs;
        end

        % Save one final pattern for plotting
        if trial == 1
            finalPatterns(j, :) = w;
        end

    end

end

%% Calculate statistics across 20 trials

meanPeriods = zeros(1, 2);
stdPeriods = zeros(1, 2);

for j = 1:2
    valid = stripePeriods(~isnan(stripePeriods(:,j)),j);

    meanPeriods(j) = mean(valid);
    stdPeriods(j) = std(valid);
end

%% Print numerical results

fprintf('\nPROBLEM 1(c) RESULTS\n');
fprintf('----------------------------------------\n');

for j = 1:2

    fprintf('\nsigma_i = %.0f\n', sigma_i_values(j));

    fprintf('Measured period = %.4f +/- %.4f\n', ...
        meanPeriods(j), stdPeriods(j));

    fprintf('Continuous theory = %.4f\n', ...
        theoreticalPeriods(j));

    fprintf('Fastest ring mode = %d\n', ...
        fastestModes(j));

    fprintf('Discrete-mode period = %.4f\n', ...
        discretePeriods(j));

end

%% Figure 1: Final ocular dominance patterns

figure(1);

for j = 1:2

    subplot(2,1,j);

    plot(positions, finalPatterns(j,:), ...
        'LineWidth', 1.2);

    hold on;
    yline(0, 'k--');

    xlabel('Cortical position (sites)');
    ylabel('w_R - w_L');

    title(sprintf( ...
        'Final Ocular Dominance Pattern, sigma_i = %g', ...
        sigma_i_values(j)));

    xlim([0 N-1]);
    ylim([-1.1 1.1]);
    grid on;

end

%% Figure 2: Stripe period comparison

figure(2);

errorbar(sigma_i_values, meanPeriods, stdPeriods, ...
    'ko-', 'LineWidth', 1.5, 'MarkerSize', 7);

hold on;

plot(sigma_i_values, theoreticalPeriods, ...
    'bs--', 'LineWidth', 1.5, 'MarkerSize', 7);

plot(sigma_i_values, discretePeriods, ...
    'rd--', 'LineWidth', 1.5, 'MarkerSize', 7);

xlabel('Inhibitory width sigma_i');
ylabel('Stripe period (sites)');

title('Measured vs. Predicted Stripe Period');

legend('Simulation mean +/- SD (20 trials)', ...
       'Continuous theory', ...
       'Fastest discrete Fourier mode', ...
       'Location', 'best');

xticks(sigma_i_values);
grid on;

%% Save figures

print(figure(1), 'HW4_P1c_Patterns.png', '-dpng', '-r300');
print(figure(2), 'HW4_P1c_Periods.png', '-dpng', '-r300');
