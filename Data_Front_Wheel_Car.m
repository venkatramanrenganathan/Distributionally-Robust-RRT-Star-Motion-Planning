% Double integrator data

% Euler discretized discrete time double integrator
dt = 0.1; % time discretization period
Th = 10;  % steering time horizon
% dynamics matrix

Acts = [-2.3142 -0.9910 1.0834
        6.0084  0.3346  52651.875
        0       0       0];
Bcts = [0       0       1]';
n = size(Acts,1);
m = size(Bcts,2);
A = expm(dt*Acts);
B = dt*Bcts + 0.5*dt*Acts*Bcts + (1/6)*dt^2*Acts^2*Bcts;
% disturbance input matrix
G = B;

% run dynamic programming to compute optimal controller
Q        = 4*eye(n) ; % state stage cost
QT       = 10*eye(n); % state terminal cost
R        = 20*eye(m); % input cost
diag_vec = n:-1:1;
W        = 0.001*diag(diag_vec); % disturbance covariance

% initialize
x0 = [0.5 0 0]; % tree root (initial state mean)
S0 = [0.001 0 0; 0 0 0; 0 0 0];
