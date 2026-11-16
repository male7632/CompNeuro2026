
%% APPM 5370 - Homework 6
% Problem 1(c): Optimal Patch Leaving

clear; clc; close all;
rng(6);

%% Parameters

rho = 0.5;
pH = 0.5;
tau = 5;

N = 4e5;  % Number of patch visits per waiting time

%% Theoretical food intake rate

R = @(T) pH .* (1-exp(-rho.*T)) ./ ...
    ((pH/rho).*(1-exp(-rho.*T)) + ...
    (1-pH).*T + tau);

% Belief-weighted marginal food rate
lambda = @(T) rho*pH.*exp(-rho.*T) ./ ...
    (pH.*exp(-rho.*T) + 1-pH);

% Known-patch intake rate
Rknown = pH/(pH/rho + tau);

% Find optimal leaving time
Tstar = fzero(@(T) R(T)-lambda(T), [0.1, 20]);
Rstar = R(Tstar);

fprintf('Optimal waiting time T* = %.6f\n', Tstar);
fprintf('Optimal intake rate R* = %.6f\n', Rstar);
fprintf('Known-patch rate = %.6f\n\n', Rknown);

%% Part 1(c)(i): Monte Carlo simulation

Tvals = [2, Tstar, 10];

Rsim = zeros(size(Tvals));
Rtheory = R(Tvals);

% Generate 400,000 patch types
isHigh = rand(N,1) < pH;

% Food discovery times
% Low patches never yield food
foodTime = inf(N,1);

% Exponential discovery times for high patches
foodTime(isHigh) = -log(rand(sum(isHigh),1))/rho;

for i = 1:length(Tvals)

    T = Tvals(i);

    % Did the animal find food before leaving?
    foundFood = foodTime <= T;

    % Waiting time is the lesser of discovery time
    % and the maximum allowed waiting time
    waitTime = min(foodTime, T);

    % Total time including travel
    totalTime = sum(waitTime + tau);

    % Total food collected
    totalFood = sum(foundFood);

    % Simulated intake rate
    Rsim(i) = totalFood / totalTime;

end

% Compare simulation to theory
percentError = 100 * (Rsim-Rtheory)./Rtheory;

Results = table(Tvals', Rtheory', Rsim', ...
    percentError', 'VariableNames', ...
    {'T','Theory','Simulation','PercentError'});

disp('Simulation Results:')
disp(Results)

%% Part 1(c)(ii): Plot theoretical rates

Tgrid = linspace(0, 12, 1000);

figure;

plot(Tgrid, R(Tgrid), 'b-', ...
    'LineWidth', 2);
hold on;

plot(Tgrid, lambda(Tgrid), 'r-', ...
    'LineWidth', 2);

yline(Rknown, 'k--', 'LineWidth', 1.5);

% Mark optimal point
plot(Tstar, Rstar, 'ko', ...
    'MarkerFaceColor', 'k', ...
    'MarkerSize', 8);

% Mark optimal leaving time
xline(Tstar, 'k:', 'LineWidth', 1.2);

xlabel('Waiting Time T');
ylabel('Food Intake Rate');
title('Problem 1(c): Optimal Patch Leaving');

legend('R(T)', 'Marginal Rate', 'R known', ...
    'Optimal Point', 'T*', ...
    'Location', 'northeast');

grid on;
xlim([0, 12]);
ylim([0, 0.26]);

hold off;
