clear;
clc;
close all;

D = 1;

a = linspace(0.001, 0.999, 2000);

c = sqrt(D) .* (1 - 2*a) ./ sqrt(a .* (1 - a));

c_passive = 2 * sqrt(D);

a_cross = (2 - sqrt(2)) / 4;

figure;
plot(a, c, 'LineWidth', 2);
hold on;

yline(c_passive, '--', 'Passive apparent speed = 2\sqrt{D}', ...
    'LineWidth', 1.5);

plot(a_cross, c_passive, 'o', ...
    'MarkerSize', 8, 'LineWidth', 1.5);

xline(0.5, ':', 'Stall point a = 0.5', ...
    'LineWidth', 1.5);

xlabel('Excitation threshold, a');
ylabel('Front speed, c');
title('Active Traveling-Front Speed, D = 1');
grid on;

ylim([-6 6]);

legend('Active front speed', ...
       'Passive apparent speed', ...
       'Active/passive crossing', ...
       'Stall threshold', ...
       'Location', 'best');