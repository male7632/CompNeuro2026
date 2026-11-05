%% APPM 5370 / 4370 - HW5 Problem 3(d): Neural-field bump dynamics


clear; clc; close all;

%% Grid and numerical parameters
xmin = -20;
xmax = 20;
dx = 0.02;
dt = 0.02;
tEnd = 120;                     % Enough to settle all requested cases
x = (xmin:dx:xmax)';
N = numel(x);
t = (0:round(tEnd/dt))' * dt;
nSteps = numel(t)-1;
mid = (N+1)/2;                 % x = 0 lies exactly on the grid

% Use the equivalent stable representation of sigmoid eta = 100:
% 1/(1+exp(-100*z)) = (1+tanh(50*z))/2.
eta = 100;

%% Zero-padded FFT kernel
% To compute a nonperiodic convolution on the finite domain, zero-pad
% activity to M >= 2*N-1. Use the positive kernel lags at the beginning
% of the FFT array and negative lags at the end. This implements the
% Toeplitz (linear) convolution on the first N output values.
M = 2^nextpow2(2*N-1);
lags = (0:N-1)' * dx;
wPositive = 2*exp(-lags)-exp(-lags/2);
wPad = zeros(M,1);
wPad(1:N) = wPositive;
wPad(end-(N-2):end) = flipud(wPositive(2:end));
wHat = fft(wPad);

% For stationary bumps: U(x) = W(x+a) - W(x-a), where W is odd.
W = @(z) 2*sign(z).*(exp(-abs(z)/2)-exp(-abs(z)));
aNarrow = @(k) -log((1+sqrt(1-2*k))/2);
aWide   = @(k) -log((1-sqrt(1-2*k))/2);
lamWide = @(k) -(sqrt(1-2*k)*(1-sqrt(1-2*k))) / ...
                 (1 + sqrt(1-2*k)*(1-sqrt(1-2*k))/2);

%% Part 3(d)(i): Four perturbed bumps for kappa = 0.3
kappa = 0.3;
aMinus = aNarrow(kappa);
aPlus  = aWide(kappa);
initialA = [aMinus-0.02, aMinus+0.02, aPlus-0.05, aPlus+0.05];
caseNames = {'narrow - 0.02 (dies)', 'narrow + 0.02 (grows)', ...
             'wide - 0.05 (returns)', 'wide + 0.05 (returns)'};

figure(1); clf;
set(gcf,'Color','w','Position',[100 100 1020 620]);
hold on;

fprintf('PART 3(d)(i): kappa = %.2f\n',kappa);
fprintf('  Narrow theoretical half-width a_- = %.6f\n',aMinus);
fprintf('  Wide theoretical half-width   a_+ = %.6f\n',aPlus);
fprintf('  Initial a0       Final numerical half-width\n');

for j = 1:4
    u0 = W(x+initialA(j))-W(x-initialA(j));
    aTime = simulateBump(u0,kappa,x,dx,dt,nSteps,mid,M,wHat,eta);
    plot(t,aTime,'LineWidth',1.8,'DisplayName',caseNames{j});
    fprintf('  %9.5f            %9.6f\n',initialA(j),aTime(end));
end

yline(aMinus,'k--','a_-  (unstable)','LabelHorizontalAlignment','left');
yline(aPlus,'k:','a_+  (stable)','LabelHorizontalAlignment','left');
xlabel('Time (model time units)');
ylabel('Active-region half-width a(t)');
title('Problem 3(d)(i): Perturbed bump half-widths, \kappa = 0.3');
legend('Location','eastoutside');
set(gca,'FontSize',11);
grid on;
xlim([0 tEnd]);
exportgraphics(gcf,'P3d_i_bump_width_trajectories.png','Resolution',200);

%% Part 3(d)(ii): Relaxation from a_+ + 0.05 for three thresholds
kappas = [0.10 0.30 0.45];
fitTmin = 2;
fitTmax = 20;

figure(2); clf;
set(gcf,'Color','w','Position',[140 60 950 850]);

fprintf('\nPART 3(d)(ii): initial a0 = a_+ + 0.05\n');
fprintf(' kappa       a_+      lambda_fit    lambda_theory    diff(%%)      final a\n');

for j = 1:numel(kappas)
    kappa = kappas(j);
    aPlus = aWide(kappa);
    u0 = W(x+aPlus+0.05)-W(x-aPlus-0.05);
    aTime = simulateBump(u0,kappa,x,dx,dt,nSteps,mid,M,wHat,eta);

    % IMPORTANT: compare with the ANALYTICAL a_+ as specified.
    % Only fit the exponential portion, prior to the small discretization
    % plateau seen at long times. Same window used for all kappas.
    delta = abs(aTime-aPlus);
    fitMask = t >= fitTmin & t <= fitTmax & delta > 1e-12;
    p = polyfit(t(fitMask),log(delta(fitMask)),1);
    lambdaFit = p(1);
    lambdaTheory = lamWide(kappa);
    diffPct = 100*(lambdaFit-lambdaTheory)/abs(lambdaTheory);

    fprintf(' %5.2f     %8.5f     %+9.5f      %+9.5f     %+7.2f     %8.5f\n', ...
            kappa,aPlus,lambdaFit,lambdaTheory,diffPct,aTime(end));

    subplot(3,1,j);
    semilogy(t,delta,'LineWidth',1.6,'DisplayName','|a(t) - a_+|');
    hold on;
    semilogy(t(fitMask),exp(polyval(p,t(fitMask))),'k--','LineWidth',1.7, ...
         'DisplayName',sprintf('Fit: slope = %.4f',lambdaFit));
    xlabel('Time (model time units)');
    ylabel('|a(t) - a_+|');
    title(sprintf('\kappa = %.2f: fitted \lambda = %.4f, theory = %.4f', ...
          kappa,lambdaFit,lambdaTheory));
    legend('Location','northeast');
    grid on;
    xlim([0 80]);
end
sgtitle('Problem 3(d)(ii): Semilog width-relaxation fits (t = 2 to 20)');
exportgraphics(gcf,'P3d_ii_relaxation_fits.png','Resolution',200);

fprintf('\nNOTE: Late-time residual plateaus are numerical-grid/sigmoid offsets.\n');
fprintf('The reported slopes are fitted only over t = %.0f to %.0f.\n',fitTmin,fitTmax);

%% Local functions
function aTime = simulateBump(u,kappa,x,dx,dt,nSteps,mid,M,wHat,eta)
    % Explicit Euler time stepping with zero-padded FFT convolution.
    N = numel(x);
    aTime = zeros(nSteps+1,1);
    aTime(1) = bumpHalfWidth(u,x,kappa,mid);
    padded = zeros(M,1);

    for n = 1:nSteps
        % Numerically stable exact equivalent of sigmoid(eta*(u-kappa))
        firing = 0.5*(1+tanh(0.5*eta*(u-kappa)));
        padded(1:N) = firing;
        convFull = real(ifft(fft(padded).*wHat));
        recurrent = dx*convFull(1:N);
        u = u + dt*(-u+recurrent);
        aTime(n+1) = bumpHalfWidth(u,x,kappa,mid);
    end
end

function a = bumpHalfWidth(u,x,kappa,mid)
    % Symmetric, center-based half-width with LINEAR interpolation at edge.
    % If no active region remains at x=0, return zero (extinction).
    if u(mid) < kappa
        a = 0;
        return;
    end
    idxFirstBelow = find(u(mid:end)<kappa,1,'first');
    if isempty(idxFirstBelow)
        a = x(end);
        return;
    end
    j = mid+idxFirstBelow-2; % last point that is above threshold
    if j < mid
        a = 0;
        return;
    end
    a = x(j) + (kappa-u(j))*(x(j+1)-x(j))/(u(j+1)-u(j));
end
