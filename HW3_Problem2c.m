%% Problem 2(c): Numerical Phase Response Curves

clear;
clc;
close all;

%% ============================================================
% Part 1: LIF PRC
% ============================================================

tau = 1;
vr = 0;
vth = 1;
I = 1.5;

delta = 1e-3;

% Exact LIF period
Delta_LIF = tau * log((I*tau - vr)/(I*tau - vth));

fprintf('LIF period = %.4f\n', Delta_LIF);

% Phase grid
phi_LIF = linspace(0.01, 0.99, 100);

R_num_LIF = zeros(size(phi_LIF));

%% Measure LIF PRC numerically

for k = 1:length(phi_LIF)

    % Time when perturbation is applied
    tkick = phi_LIF(k) * Delta_LIF;

    % Unperturbed voltage at kick time
    vkick = I*tau + ...
        (vr - I*tau)*exp(-tkick/tau);

    % Apply voltage kick
    vkick_pert = vkick + delta;

    % Remaining time to threshold after perturbation
    Tremain = tau * log( ...
        (I*tau - vkick_pert) / ...
        (I*tau - vth));

    % Perturbed spike time
    Tprime = tkick + Tremain;

    % Numerical PRC
    R_num_LIF(k) = ...
        (Delta_LIF - Tprime) / ...
        (Delta_LIF * delta);

end

%% Analytical LIF PRC

R_exact_LIF = ...
    tau * exp(phi_LIF*Delta_LIF/tau) ./ ...
    (Delta_LIF*(I*tau - vr));

%% Plot LIF PRC

figure;

plot(phi_LIF, R_num_LIF, 'o', ...
    'MarkerSize', 4);

hold on;

plot(phi_LIF, R_exact_LIF, ...
    'LineWidth', 1.5);

xlabel('Phase \phi');
ylabel('R(\phi)');
title('LIF Phase Response Curve');

legend('Numerical perturbation', ...
       'Analytical PRC', ...
       'Location', 'northwest');

grid on;


%% ============================================================
% Part 2: FitzHugh-Nagumo PRC
% ============================================================

delta = 1e-3;

% FitzHugh-Nagumo equations
fhn = @(t,x) [ ...
    x(1) - x(1)^3/3 - x(2) + 0.5; ...
    0.08*(x(1) + 0.7 - 0.8*x(2)) ];

%% First settle onto the limit cycle

x0 = [-1; -0.5];

optsSettle = odeset( ...
    'RelTol', 1e-9, ...
    'AbsTol', 1e-11, ...
    'Events', @upcrossing);

[t,x,te,xe] = ode45( ...
    fhn, ...
    [0 700], ...
    x0, ...
    optsSettle);

% Use only late crossings
steadyMask = te > 400;

te_steady = te(steadyMask);
xe_steady = xe(steadyMask,:);

% Make sure enough crossings were found
if length(te_steady) < 3
    error('Not enough steady-state FHN crossings were detected.');
end

% Compute periods
periods = diff(te_steady);

% Average up to final 6 available periods
nPeriodsToAverage = min(6, length(periods));

Delta_FHN = mean( ...
    periods(end-nPeriodsToAverage+1:end));

fprintf('\nFHN period = %.4f\n', Delta_FHN);

%% Reference state at phase zero

% Use final steady upward crossing
x_reference = xe_steady(end,:)';

%% Phase grid

phi_FHN = linspace(0.02, 0.98, 60);

R_FHN = zeros(size(phi_FHN));

%% Compute FHN PRC numerically

for k = 1:length(phi_FHN)

    %% 1. Advance from phase zero to desired phase

    tkick = phi_FHN(k) * Delta_FHN;

    optsNoEvent = odeset( ...
        'RelTol', 1e-10, ...
        'AbsTol', 1e-12);

    [~,xpre] = ode45( ...
        fhn, ...
        [0 tkick], ...
        x_reference, ...
        optsNoEvent);

    xkick = xpre(end,:)';

    %% 2. Apply small kick to voltage

    xkick(1) = xkick(1) + delta;

    %% 3. Integrate perturbed trajectory forward several cycles

    optsCross = odeset( ...
        'RelTol', 1e-9, ...
        'AbsTol', 1e-11, ...
        'Events', @upcrossing);

    [~,~,tePert] = ode45( ...
        fhn, ...
        [tkick tkick + 8*Delta_FHN], ...
        xkick, ...
        optsCross);

    %% 4. Compare a later crossing after relaxation

    % Use the 5th upward crossing after the kick
    crossingNumber = 5;

    if length(tePert) >= crossingNumber

        tPertCross = tePert(crossingNumber);

        % After the kick at phi*Delta, the next unperturbed crossing
        % occurs at Delta, then 2*Delta, etc.
        %
        % Therefore the nth crossing after the kick corresponds to:
        tExpected = crossingNumber * Delta_FHN;

        % Phase shift in time
        timeShift = tExpected - tPertCross;

        % Numerical PRC
        R_FHN(k) = ...
            timeShift / ...
            (Delta_FHN * delta);

    else

        R_FHN(k) = NaN;

    end

end

%% Plot FHN PRC

figure;

plot(phi_FHN, R_FHN, ...
    'o-', ...
    'LineWidth', 1.2, ...
    'MarkerSize', 4);

xlabel('Phase \phi');
ylabel('R(\phi)');
title('FitzHugh-Nagumo Phase Response Curve');

grid on;


%% ============================================================
% Report FHN results
% ============================================================

valid = ~isnan(R_FHN);

Rmin = min(R_FHN(valid));
Rmax = max(R_FHN(valid));

fprintf('\nFHN PRC minimum = %.4f\n', Rmin);
fprintf('FHN PRC maximum = %.4f\n', Rmax);

negativePhases = phi_FHN(R_FHN < 0);

if ~isempty(negativePhases)

    fprintf( ...
        'R(phi) < 0 approximately for phi = %.3f to %.3f\n', ...
        min(negativePhases), ...
        max(negativePhases));

else

    fprintf('No negative PRC region detected.\n');

end


%% ============================================================
% Local Event Function
% ============================================================

function [value,isterminal,direction] = upcrossing(~,x)

    % Detect v = 0
    value = x(1);

    % Do not terminate integration
    isterminal = 0;

    % Only detect upward crossings
    direction = 1;

end