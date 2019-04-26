% Cycle data

% Euler discretized discrete time double integrator
dt = 0.01; % time discretization period
Th = 10;   % steering time horizon
v  = 2;
% dynamics matrix

A = [0 0 1 0
     0 0 0 1
     13.67 0.225-1.319*v^2 -0.164*v -0.552*v
     4.857 10.81-1.125*v^2 3.621*v  -2.388*v];
B = [0 0 -0.339 7.457]';

n = size(A,1);
m = size(B,2);

G = B; % disturbance input matrix

% run dynamic programming to compute optimal controller
Q        = 0.4*eye(n) ; % state stage cost
QT       = 10*eye(n); % state terminal cost
R        = 20*eye(m); % input cost
diag_vec = n:-1:1;
W        = 0.001*diag(diag_vec); % disturbance covariance

% initialize
x0 = [0.5 0 0 0]; % tree root (initial state mean)
S0 = 0.001*eye(n);
%S0 = [0.001 0 0 0; 0 0 0 0; 0 0 0 0;0 0 0 0];
