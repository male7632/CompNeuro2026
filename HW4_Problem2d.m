%% APPM 4370/5370 Homework 4 - Problem 2(d)
% Competitive circuit: accuracy and decision time
% Simulates the ORIGINAL TWO-POPULATION MODEL with independent noise.
% Records first threshold crossings from the SAME trajectories, which
% avoids rerunning 10,000 trials separately for every threshold.

clear; clc; close all;
rng(4370);                    % Reproducible random numbers

%% Parameters
mu = 0.2;
sigma = 0.5;
I0 = 5;
dt = 0.002;
nTrials = 10000;
thetaGrid = (0.5:0.5:4)';
lambdaValues = [0, 0.2];

%% Theoretical accuracy and mean decision time
% For lambda=0, use the closed-form results from Problem 2(b).
pTheory0 = 1 ./ (1 + exp(-mu*thetaGrid/sigma^2));
tTheory0 = (thetaGrid/mu) .* tanh(mu*thetaGrid/(2*sigma^2));

% For lambda>0, use Problem 2(c) for accuracy, and solve the backward
% equation numerically for mean decision time.
lambdaU = lambdaValues(2);
Phi = @(z) 0.5*(1 + erf(z/sqrt(2)));
z = @(x) (lambdaU*x + mu)/(sigma*sqrt(lambdaU));
pTheoryU = zeros(size(thetaGrid));
tTheoryU = zeros(size(thetaGrid));
for j = 1:numel(thetaGrid)
    theta = thetaGrid(j);
    pTheoryU(j) = (Phi(z(0)) - Phi(z(-theta))) / ...
                  (Phi(z(theta)) - Phi(z(-theta)));
    tTheoryU(j) = meanDecisionTimeFD(theta, lambdaU, mu, sigma);
end

%% Perfect-integrator threshold with matching theoretical accuracy
pTarget = pTheoryU(end);         % Unstable integrator: lambda=0.2, theta=4
thetaMatch = (sigma^2/mu)*log(pTarget/(1-pTarget));
tMatchTheory = (thetaMatch/mu)*tanh(mu*thetaMatch/(2*sigma^2));

% Include the matched threshold in the lambda=0 simulation to compare
% directly with the unstable circuit at theta=4.
theta0 = sort([thetaGrid; thetaMatch]);
[~, idxGrid0] = ismember(thetaGrid, theta0);
[~, idxMatch] = min(abs(theta0-thetaMatch));

%% Monte Carlo simulations
fprintf('Simulating lambda = 0 (10,000 trials)...\n');
[hitTime0, hitRight0] = simulateCircuit(0, theta0, nTrials, ...
                                       dt, mu, sigma, I0);
fprintf('Simulating lambda = 0.2 (10,000 trials)...\n');
[hitTimeU, hitRightU] = simulateCircuit(lambdaU, thetaGrid, nTrials, ...
                                       dt, mu, sigma, I0);

% The decision at +theta (r1 > r2) is correct, as mu > 0.
pSim0 = mean(hitRight0(:,idxGrid0),1)';
tSim0 = mean(hitTime0(:,idxGrid0),1)';
pSimU = mean(hitRightU,1)';
tSimU = mean(hitTimeU,1)';
pMatchSim = mean(hitRight0(:,idxMatch));
tMatchSim = mean(hitTime0(:,idxMatch));

%% Numerical output
fprintf('\nPROBLEM 2(d) RESULTS\n');
fprintf('lambda  theta  P_sim   P_theory  T_sim    T_theory\n');
fprintf('--------------------------------------------------\n');
for j = 1:numel(thetaGrid)
    fprintf('%5.1f  %5.1f  %6.4f   %6.4f    %7.4f  %7.4f\n', ...
        0, thetaGrid(j), pSim0(j), pTheory0(j), tSim0(j), tTheory0(j));
end
for j = 1:numel(thetaGrid)
    fprintf('%5.1f  %5.1f  %6.4f   %6.4f    %7.4f  %7.4f\n', ...
        lambdaU, thetaGrid(j), pSimU(j), pTheoryU(j), tSimU(j), tTheoryU(j));
end
fprintf('\nMATCHED-ACCURACY COMPARISON\n');
fprintf('Unstable integrator: lambda=0.2, theta=4\n');
fprintf('  Theory accuracy   = %.4f\n', pTarget);
fprintf('  Sim accuracy      = %.4f\n', pSimU(end));
fprintf('  Theory mean time  = %.4f\n', tTheoryU(end));
fprintf('  Sim mean time     = %.4f\n', tSimU(end));
fprintf('Perfect integrator: lambda=0, theta=%.4f\n', thetaMatch);
fprintf('  Theory accuracy   = %.4f\n', pTarget);
fprintf('  Sim accuracy      = %.4f\n', pMatchSim);
fprintf('  Theory mean time  = %.4f\n', tMatchTheory);
fprintf('  Sim mean time     = %.4f\n', tMatchSim);
fprintf('  Unstable model time / perfect model time (theory) = %.4f\n', ...
    tTheoryU(end)/tMatchTheory);

%% Figure 1 - Accuracy
figure(1); clf;
h1 = plot(thetaGrid, pTheory0, 'k-', 'LineWidth',1.6); hold on;
h2 = plot(thetaGrid, pTheoryU, 'b-', 'LineWidth',1.6);
h3 = plot(thetaGrid, pSim0, 'ko', 'MarkerSize',6, 'LineWidth',1.2);
h4 = plot(thetaGrid, pSimU, 'bs', 'MarkerSize',6, 'LineWidth',1.2);
h5 = plot(thetaMatch, pMatchSim, 'md', 'MarkerSize',7, ...
    'LineWidth',1.4);
xlabel('Decision threshold \theta');
ylabel('Probability correct');
title('Problem 2(d): Decision accuracy vs. threshold');
legend([h1,h2,h3,h4,h5], ...
    {'Theory, \lambda=0', 'Theory, \lambda=0.2', ...
     'Simulation, \lambda=0', 'Simulation, \lambda=0.2', ...
     'Matched threshold, \lambda=0'}, 'Location','southeast');
axis([0.4 4.1 0.45 1.02]); grid on;
print(gcf, 'HW4_P2d_Accuracy.png','-dpng','-r300');

%% Figure 2 - Mean decision time
figure(2); clf;
h1 = plot(thetaGrid, tTheory0, 'k-', 'LineWidth',1.6); hold on;
h2 = plot(thetaGrid, tTheoryU, 'b-', 'LineWidth',1.6);
h3 = plot(thetaGrid, tSim0, 'ko', 'MarkerSize',6, 'LineWidth',1.2);
h4 = plot(thetaGrid, tSimU, 'bs', 'MarkerSize',6, 'LineWidth',1.2);
h5 = plot(thetaMatch, tMatchSim, 'md', 'MarkerSize',7, ...
    'LineWidth',1.4);
xlabel('Decision threshold \theta');
ylabel('Mean decision time (model time units)');
title('Problem 2(d): Mean decision time vs. threshold');
legend([h1,h2,h3,h4,h5], ...
    {'Theory, \lambda=0', 'Theory, \lambda=0.2', ...
     'Simulation, \lambda=0', 'Simulation, \lambda=0.2', ...
     'Matched threshold, \lambda=0'}, 'Location','northwest');
grid on;
print(gcf, 'HW4_P2d_DecisionTime.png','-dpng','-r300');

%% Local functions
function [hitTimes, hitCorrect] = simulateCircuit(lambda, thresholds, ...
                                                  nTrials, dt, mu, sigma, I0)
    thresholds = sort(thresholds(:));
    nThresholds = numel(thresholds);
    ws = (1+lambda)/2;
    wc = ws;

    r1 = I0*ones(nTrials,1);
    r2 = I0*ones(nTrials,1);
    nextThreshold = ones(nTrials,1);
    hitTimes = NaN(nTrials,nThresholds);
    hitCorrect = false(nTrials,nThresholds);
    noiseStep = sigma*sqrt(dt);

    maxTime = 300;                 % Generous safeguard for rare long trials
    maxSteps = ceil(maxTime/dt);
    for step = 1:maxSteps
        active = find(nextThreshold <= nThresholds);
        if isempty(active)
            break;
        end
        r1old = r1(active);
        r2old = r2(active);
        nActive = numel(active);

        % Euler-Maruyama, independent white noise for each population
        r1(active) = r1old + dt*((ws-1)*r1old - wc*r2old ...
                     + I0 + mu/2) + noiseStep*randn(nActive,1);
        r2(active) = r2old + dt*((ws-1)*r2old - wc*r1old ...
                     + I0 - mu/2) + noiseStep*randn(nActive,1);

        difference = r1(active)-r2(active);
        crossed = abs(difference) >= thresholds(nextThreshold(active));
        justHit = active(crossed);

        % Record first-passage times and choices. The loop also catches
        % extremely rare steps that cross multiple thresholds at once.
        while ~isempty(justHit)
            k = nextThreshold(justHit);
            linearIndex = sub2ind([nTrials,nThresholds],justHit,k);
            hitTimes(linearIndex) = step*dt;
            hitCorrect(linearIndex) = r1(justHit) > r2(justHit);
            nextThreshold(justHit) = k+1;

            stillPending = justHit(nextThreshold(justHit) <= nThresholds);
            justHit = stillPending(abs(r1(stillPending)-r2(stillPending)) ...
                     >= thresholds(nextThreshold(stillPending)));
        end
    end

    if any(isnan(hitTimes(:)))
        warning('Some trajectories did not reach the largest threshold by t=%.1f.', maxTime);
    end
    fprintf('  Completed %d/%d trials (up to t=%.3f).\n', ...
        sum(nextThreshold>nThresholds), nTrials, step*dt);
end

function T0 = meanDecisionTimeFD(theta, lambda, mu, sigma)
    % Solve sigma^2*T'' + (lambda*x+mu)*T' = -1 on [-theta,theta],
    % with T(-theta)=T(theta)=0, using centered finite differences.
    M = 2001;                       % Odd number; x=0 is grid point
    x = linspace(-theta,theta,M)';
    h = x(2)-x(1);
    xi = x(2:end-1);
    n = numel(xi);
    drift = lambda*xi + mu;
    low = sigma^2/h^2 - drift/(2*h);
    high = sigma^2/h^2 + drift/(2*h);
    idx = (1:n)';
    A = sparse([idx;idx(2:end);idx(1:end-1)], ...
               [idx;idx(1:end-1);idx(2:end)], ...
               [-2*sigma^2/h^2*ones(n,1);low(2:end);high(1:end-1)],n,n);
    T = A\(-ones(n,1));
    T0 = T((n+1)/2);
end
