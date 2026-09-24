clear;
clc;
close all;

V0 = 16.13;           % mV
EN = 0;               % mV
gbar = 1;             % nS
Mg_values = [0 0.1 1]; % mM

V = linspace(-90, 0, 1000);   % mV

figure;
hold on;

for Mg = Mg_values

    B = 1 ./ (1 + (Mg/3.57).*exp(-V./V0));

    IN = gbar .* B .* (V - EN);   % pA

    plot(V, IN, 'LineWidth', 2, ...
        'DisplayName', sprintf('[Mg^{2+}] = %.1f mM', Mg));

end

Mg = 1;

Bfun = @(Vm) 1 ./ ...
    (1 + (Mg/3.57).*exp(-Vm./V0));

slopeFactor = @(Vm) ...
    1 + ((Vm - EN).*(1 - Bfun(Vm)))./V0;

Vstar = fzero(slopeFactor, [-90 0]);

Bstar = Bfun(Vstar);
Istar = gbar .* Bstar .* (Vstar - EN);

plot(Vstar, Istar, 'ko', ...
    'MarkerSize', 8, ...
    'LineWidth', 1.5, ...
    'DisplayName', sprintf('Zero slope: V = %.2f mV', Vstar));

xlabel('Membrane voltage, V (mV)');
ylabel('NMDA current, I_N (pA)');
title('NMDA Current-Voltage Relation');
legend('Location', 'best');
grid on;

fprintf('Zero-slope voltage for Mg = 1 mM: %.4f mV\n', Vstar);