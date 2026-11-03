%% APPM 4370/5370 - Homework 5, Problem 2(d)

clear; clc; close all;

%% Model parameters
par.w = 0.8;
par.beta = 1;
par.g = 2;
par.tauA = 120;
par.eta = 1000;
par.IL = 1.3;
par.IR = 1.3;

tEnd = 30 * par.tauA;               % t = 3600
initialState = [1; 0; 0; 0];        % [uL; uR; aL; aR]
threshold = 0.5;                    % |uL-uR| > 1/2 defines dominance
nDiscard = 3;                       % discard first 3 complete episodes
sampleStep = 0.1;                   % for measuring threshold crossings

% ode15s handles the near-step sigmoid (eta = 1000).
opts = odeset('RelTol',1e-8,'AbsTol',1e-10,'MaxStep',1);

%% Part (i): symmetric inputs, IL = IR = 1.3
fprintf('\nPART (i): w = 0.8, IL = IR = 1.3\n');

simEqual = simulateRivalry(par, initialState, tEnd, sampleStep, ...
    threshold, nDiscard, opts);

B = (par.IL-par.beta)/par.g;

fig = figure('Color','w','Position',[100 100 1100 680]);
tl = tiledlayout(fig,2,1,'TileSpacing','compact','Padding','compact');

ax1 = nexttile(tl);
plot(ax1,simEqual.t,simEqual.y(:,1),'b-','LineWidth',1.0); hold(ax1,'on');
plot(ax1,simEqual.t,simEqual.y(:,2),'r-','LineWidth',1.0);
ylim(ax1,[-0.05 1.05]); xlim(ax1,[0 tEnd]);
ylabel(ax1,'Population activity');
title(ax1,'Part 2(d)(i): Symmetric binocular rivalry (w = 0.8, I_L = I_R = 1.3)');
legend(ax1,{'u_L','u_R'},'Location','northeast');
grid(ax1,'on');

ax2 = nexttile(tl);
hL = plot(ax2,simEqual.t,simEqual.y(:,3),'b-','LineWidth',1.0);
hold(ax2,'on');
hR = plot(ax2,simEqual.t,simEqual.y(:,4),'r-','LineWidth',1.0);
yline(ax2,B,'k--',sprintf('B = %.2f',B));
yline(ax2,1-B,'k:',sprintf('1-B = %.2f',1-B));
ylim(ax2,[-0.05 1.05]); xlim(ax2,[0 tEnd]);
xlabel(ax2,'Time (rate-time-constant units)');
ylabel(ax2,'Adaptation');
legend(ax2,[hL,hR],{'a_L','a_R'},'Location','northeast');
grid(ax2,'on');

exportgraphics(fig,'P2d_i_symmetric_rivalry.png','Resolution',300);

fprintf('Switch threshold B = %.4f; 1-B = %.4f\n',B,1-B);
fprintf('Number of complete measured intervals after discarding first %d: %d\n', ...
    nDiscard,simEqual.nKept);
fprintf('Measured mean T_L = %.3g, T_R = %.3g\n', ...
    simEqual.meanL,simEqual.meanR);

%% Part (ii): increase IL with IR fixed at 1.3
fprintf('\nPART (ii): w = 0.8, I_R = 1.3\n');
fprintf('   IL       T_L sim   T_L theory    diff(%%)    T_R sim   T_R theory    diff(%%)\n');

ILvalues = [1.3, 1.45, 1.6];
results = struct([]);
for j = 1:numel(ILvalues)
    par.IL = ILvalues(j);
    par.IR = 1.3;

    s = simulateRivalry(par, initialState, tEnd, sampleStep, ...
        threshold, nDiscard, opts);

    BL = (par.IL-par.beta)/par.g;
    BR = (par.IR-par.beta)/par.g;
    TLtheory = par.tauA*log((1-BL)/BR);
    TRtheory = par.tauA*log((1-BR)/BL);

    diffLpercent = 100*(s.meanL-TLtheory)/TLtheory;
    diffRpercent = 100*(s.meanR-TRtheory)/TRtheory;
    fprintf('%5.2f        %7.3g      %7.3g       %+6.2f     %7.3g      %7.3g       %+6.2f\n', ...
        par.IL,s.meanL,TLtheory,diffLpercent, ...
        s.meanR,TRtheory,diffRpercent);

    results(j).IL = par.IL;
    results(j).IR = par.IR;
    results(j).meanL = s.meanL;
    results(j).meanR = s.meanR;
    results(j).TLtheory = TLtheory;
    results(j).TRtheory = TRtheory;
    results(j).diffLpercent = diffLpercent;
    results(j).diffRpercent = diffRpercent;
    results(j).nL = s.nL;
    results(j).nR = s.nR;
end

% Optional comparison figure for part (ii).
fig2 = figure('Color','w','Position',[100 100 850 540]);
ILplot = [results.IL];
simL = [results.meanL];
simR = [results.meanR];
theoryL = [results.TLtheory];
theoryR = [results.TRtheory];
plot(ILplot,theoryL,'b-','LineWidth',1.5); hold on;
plot(ILplot,simL,'bo','MarkerSize',7,'LineWidth',1.5);
plot(ILplot,theoryR,'r-','LineWidth',1.5);
plot(ILplot,simR,'rs','MarkerSize',7,'LineWidth',1.5);
xlabel('Left-eye input I_L (I_R = 1.3)');
ylabel('Mean dominance time (rate-time-constant units)');
title('Part 2(d)(ii): Simulated and theoretical dominance times');
legend({'T_L theory','T_L simulation','T_R theory','T_R simulation'}, ...
    'Location','southwest');
grid on;
exportgraphics(fig2,'P2d_ii_dominance_comparison.png','Resolution',300);

%% Part (iii): w = 0.3, outside the mutual-inhibition switching window
par.w = 0.3;
par.IL = 1.3;
par.IR = 1.3;

sLowW = simulateRivalry(par, initialState, tEnd, sampleStep, ...
    threshold, nDiscard, opts);

% For the weak-self-excitation regime the assignment supplies:
Talt = par.tauA*log((par.w+par.IL)/(par.g-par.w-par.IL));
Tmeasured = mean([sLowW.meanL,sLowW.meanR]);
diffPercent = 100*(Tmeasured-Talt)/Talt;

fprintf('\nPART (iii): w = 0.3, I_L = I_R = 1.3\n');
fprintf('Measured T_L = %.3g, T_R = %.3g\n',sLowW.meanL,sLowW.meanR);
fprintf('Measured mean T_dom = %.3g\n',Tmeasured);
fprintf('Predicted T_dom   = %.3g\n',Talt);
fprintf('Difference        = %+.3f%%\n',diffPercent);
fprintf('Negative difference means the simulation is faster than theory.\n');

% Save results for optional reuse; no need to submit this MAT file.
save('P2d_numerical_results.mat','results','sLowW','Talt','Tmeasured');

%% Local functions (MATLAB R2016b or newer allows script-local functions)
function sim = simulateRivalry(par, initialState, tEnd, sampleStep, ...
    threshold, nDiscard, opts)

    solution = ode15s(@(t,y) rivalryRhs(t,y,par), ...
        [0 tEnd],initialState,opts);

    % Sample the smooth interpolating solution on a fine time grid.
    t = (0:sampleStep:tEnd).';
    y = deval(solution,t).';
    delta = y(:,1)-y(:,2);

    % L is dominant if delta>0.5; R if delta<-0.5.
    % In between, neither exceeds the other by the requested margin.
    winner = zeros(size(t));
    winner(delta>threshold) = 1;      % L
    winner(delta<-threshold) = -1;    % R

    % Each hand-over is identified by the first time the NEW winner crosses
    % the threshold. Interpolate between samples for sub-grid precision.
    % We do NOT count the initial transient as a hand-over. Start
    % collecting entries only when a population becomes dominant later.
    entryTime = [];
    entryWinner = [];

    for k = 2:numel(t)
        if winner(k) ~= 0 && winner(k-1) ~= winner(k)
            nextWinner = winner(k);
            if ~isempty(entryWinner) && nextWinner == entryWinner(end)
                % Re-entry of the same winner is not a hand-over.
                continue;
            end
            target = nextWinner*threshold;
            if abs(delta(k)-delta(k-1)) < 1e-12
                crossTime = t(k);
            else
                alpha = (target-delta(k-1))/(delta(k)-delta(k-1));
                alpha = max(0,min(1,alpha));
                crossTime = t(k-1)+alpha*(t(k)-t(k-1));
            end
            entryTime(end+1,1) = crossTime; %#ok<AGROW>
            entryWinner(end+1,1) = nextWinner; %#ok<AGROW>
        end
    end

    % Dominance time = elapsed time from one hand-over to the next.
    % First three COMPLETE intervals are discarded as transients.
    durations = diff(entryTime);
    labels = entryWinner(1:end-1);
    valid = (1:numel(durations)).' > nDiscard;
    Ldur = durations(valid & labels==1);
    Rdur = durations(valid & labels==-1);

    if isempty(Ldur) || isempty(Rdur)
        error('Too few completed L/R dominance periods. Increase tEnd.');
    end

    sim.t = t;
    sim.y = y;
    sim.meanL = mean(Ldur);
    sim.meanR = mean(Rdur);
    sim.Ldurations = Ldur;
    sim.Rdurations = Rdur;
    sim.entryTimes = entryTime;
    sim.entryWinners = entryWinner;
    sim.nL = numel(Ldur);
    sim.nR = numel(Rdur);
    sim.nKept = sim.nL+sim.nR;
end

function dy = rivalryRhs(~,y,par)
    uL = y(1); uR = y(2);
    aL = y(3); aR = y(4);

    xL = par.w*uL-par.beta*uR-par.g*aL+par.IL;
    xR = par.w*uR-par.beta*uL-par.g*aR+par.IR;

    % Stable form of 1/(1+exp(-eta*x)); avoids exp overflow at eta=1000.
    sigmoid = @(x) 0.5*(1+tanh(par.eta*x/2));

    dy = [-uL+sigmoid(xL);
          -uR+sigmoid(xR);
          (-aL+uL)/par.tauA;
          (-aR+uR)/par.tauA];
end
