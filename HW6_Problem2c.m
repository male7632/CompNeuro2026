
%% APPM 5370 - Homework 6
% Problem 2(c): Policy-Gradient Learning

clear; clc; close all;
rng(6);

%% Parameters

eta = 0.02;
N = 4e5;

xStar = 8/17;
yStar = 9/17;

tauVals = [0, 0.05];

% Logistic policy
sigmoid = @(theta) 1./(1+exp(-theta));

%% Part 2(c)(i): Deterministic Mean-Flow Orbit

% Mean-flow equations from Problem 2(b)
meanFlow = @(t,z) [ ...
    0.85*eta*(z(1)*(1-z(1)))^2*(z(2)-yStar);
   -0.85*eta*(z(2)*(1-z(2)))^2*(z(1)-xStar)];

% Initial strategy probabilities
z0 = [0.2; 0.3];

opts = odeset('RelTol',1e-10, ...
              'AbsTol',1e-12, ...
              'MaxStep',20);

[tOrbit, zOrbit] = ode45(meanFlow, ...
                        [0 40000], z0, opts);

xOrbit = zOrbit(:,1);
yOrbit = zOrbit(:,2);

% Antiderivative for K
F = @(s,a) (1-2*a).*log(s./(1-s)) ...
    + a./s + (1-a)./(1-s);

K = @(x,y) ...
    F(x,xStar)-F(xStar,xStar) + ...
    F(y,yStar)-F(yStar,yStar);

Kvals = K(xOrbit,yOrbit);

fprintf('PART 2(c)(i): MEAN-FLOW ORBIT\n');
fprintf('x range: [%.6f, %.6f]\n', ...
    min(xOrbit),max(xOrbit));
fprintf('y range: [%.6f, %.6f]\n', ...
    min(yOrbit),max(yOrbit));
fprintf('K initial: %.8f\n', Kvals(1));
fprintf('K variation: %.6e\n\n', ...
    max(Kvals)-min(Kvals));

% Plot mean-flow orbit
figure(1);

plot(xOrbit,yOrbit,'b-', 'LineWidth',1.5);
hold on;

plot(xStar,yStar,'r*', ...
    'MarkerSize',12,'LineWidth',2);

plot(z0(1),z0(2),'ko', ...
    'MarkerFaceColor','k');

xlabel('x = P(Shoot 3)');
ylabel('y = P(Guard Paint)');
title('Problem 2(c)(i): Mean-Flow Orbit');

legend('Mean-flow orbit','Nash equilibrium', ...
    'Initial condition','Location','best');

xlim([0 1]);
ylim([0 1]);
axis square;
grid on;

%% Part 2(c)(ii): Stochastic Policy Gradients

% Rows: shoot 2, shoot 3
% Columns: guard arc (a=0), guard paint (a=1)

makeProb = [0.60, 0.40;
            0.25, 0.40];

shotPoints = [2, 3];

X = zeros(N+1,2);
Y = zeros(N+1,2);

for j = 1:length(tauVals)

    entTau = tauVals(j);

    % Initial logits
    theta1 = 0.3;
    theta2 = -0.3;

    % Running-average reward baseline
    baselinePts = 0;

    for n = 1:N

        % Current policies
        x = sigmoid(theta1);
        y = sigmoid(theta2);

        X(n,j) = x;
        Y(n,j) = y;

        % Sample offensive and defensive actions
        % aOff = 1 means shoot 3
        % aDef = 1 means guard paint

        aOff = double(rand < x);
        aDef = double(rand < y);

        % Random shot outcome
        pMake = makeProb(aOff+1,aDef+1);

        madeShot = rand < pMake;

        % Offensive reward
        points = shotPoints(aOff+1)*madeShot;

        % Defense reward is negative points

        % Policy gradients with entropy bonus
        grad1 = (points-baselinePts)*(aOff-x) ...
            - entTau*theta1*x*(1-x);

        grad2 = (-points+baselinePts)*(aDef-y) ...
            - entTau*theta2*y*(1-y);

        % Update logits
        theta1 = theta1 + eta*grad1;
        theta2 = theta2 + eta*grad2;

        % Update baseline AFTER gradient calculation
        baselinePts = baselinePts + ...
            (points-baselinePts)/n;

    end

    % Record final policy
    X(N+1,j) = sigmoid(theta1);
    Y(N+1,j) = sigmoid(theta2);

end

%% Compute Early/Late Statistics

q = N/4;

earlyIdx = 1:q;
lateIdx = (N-q+1):N;

xLastAvg = zeros(2,1);
yLastAvg = zeros(2,1);

distEarly = zeros(2,1);
distLate = zeros(2,1);

for j = 1:2

    xLastAvg(j) = mean(X(lateIdx,j));
    yLastAvg(j) = mean(Y(lateIdx,j));

    % Euclidean distance to Nash equilibrium
    d = hypot(X(:,j)-xStar, ...
              Y(:,j)-yStar);

    distEarly(j) = mean(d(earlyIdx));
    distLate(j) = mean(d(lateIdx));

end

Results = table(tauVals',xLastAvg,yLastAvg, ...
    distEarly,distLate, ...
    'VariableNames', ...
    {'EntropyTau','xLast','yLast', ...
     'DistanceEarly','DistanceLate'});

disp('PART 2(c)(ii): STOCHASTIC RESULTS')
disp(Results)

%% Plot Stochastic Trajectories

figure(2);

for j = 1:2

    subplot(1,2,j);

    % Downsample for plotting clarity
    idx = 1:50:(N+1);

    plot(X(idx,j),Y(idx,j), ...
        'b-','LineWidth',0.8);
    hold on;

    plot(xStar,yStar,'r*', ...
        'MarkerSize',12,'LineWidth',2);

    plot(X(1,j),Y(1,j),'ko', ...
        'MarkerFaceColor','k');

    xlabel('x = P(Shoot 3)');
    ylabel('y = P(Guard Paint)');

    title(sprintf('Entropy Bonus: tau = %.2f', ...
        tauVals(j)));

    legend('Learning trajectory', ...
        'Nash equilibrium','Initial strategy', ...
        'Location','best');

    xlim([0 1]);
    ylim([0 1]);
    axis square;
    grid on;

end

sgtitle('Problem 2(c)(ii): Stochastic Learning');

%% Optional: Distance From Nash Over Time

figure(3);

idx = 1:200:(N+1);

d0 = hypot(X(:,1)-xStar,Y(:,1)-yStar);
d1 = hypot(X(:,2)-xStar,Y(:,2)-yStar);

plot(idx,d0(idx),'b-', 'LineWidth',1);
hold on;

plot(idx,d1(idx),'r-', 'LineWidth',1);

xlabel('Possession Number');
ylabel('Distance From Nash Equilibrium');
title('Problem 2(c): Distance From Equilibrium');

legend('tau = 0','tau = 0.05', ...
    'Location','best');

grid on;
