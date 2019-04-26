% Double integrator data

% Euler discretized discrete time double integrator
dt = 0.1; % time discretization period
% dynamics matrix
A = [1 0 dt 0; 
     0 1 0 dt;
     0 0 1 0;
     0 0 0 1];
% input matrix
B = [dt^2/2 0;
     0 dt^2/2;
     dt 0;
     0 dt];
C = [1 0 0 0
     0 1 0 0];
% disturbance input matrix
G = B;

% run dynamic programming to compute optimal controller
Q  = [4*eye(2) zeros(2); zeros(2) 0.1*eye(2)]; % state stage cost
QT = [100*eye(2) zeros(2); zeros(2) 0.1*eye(2)]; % state terminal cost
R  = 0.02*eye(2); % input cost

Wbar = 0.001*[2 1; 1 2];
W = [zeros(2) zeros(2); zeros(2) Wbar]; % disturbance covariance

Th = 10;  % steering time horizon

% initialize
x0 = [0.5, 0 0 0]; % tree root (initial state mean)
S0 = [0.001*eye(2) zeros(2); zeros(2) zeros(2)];