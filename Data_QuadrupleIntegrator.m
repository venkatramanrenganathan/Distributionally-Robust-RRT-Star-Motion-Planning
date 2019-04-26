% Quadruple integrator data

% Euler discretized discrete time double integrator
dt = 0.1; % time discretization period
% dynamics matrix
A = [1 0 0 0 dt 0 0 0; 
     0 1 0 0 0 dt 0 0;
     0 0 1 0 0 0 dt 0;
     0 0 0 1 0 0 0 dt;
     zeros(4) eye(4)];
% input matrix
B = [dt^4/24 0;
     0 dt^4/24;
     dt^3/6 0;
     0 dt^3/6;
     dt^2/2 0;
     0 dt^2/2;
     dt 0;
     0 dt];
% disturbance input matrix
G = B;

% run dynamic programming to compute optimal controller
Q = [1000*eye(2) zeros(2,6); 
     zeros(2) 0.1*eye(2) zeros(2,4);
     zeros(2,4) 0.01*eye(2) zeros(2);
     zeros(2,6) 0.01*eye(2)]; % state stage cost
QT = [1000*eye(2) zeros(2,6); 
     zeros(2) 0.1*eye(2) zeros(2,4);
     zeros(2,4) 0.01*eye(2) zeros(2);
     zeros(2,6) 0.01*eye(2)]; % state terminal cost
R = 0.5*eye(2); % input cost

Wbar = 0.001*[2 1; 1 2];
W = [zeros(6) zeros(6,2); zeros(2,6) Wbar]; % disturbance covariance

Th = 50;  % steering time horizon

% initialize
x0 = [0.5, 0. 0 0 0 0 0 0]; % tree root (initial state mean)
S0 = [0.001*eye(2) zeros(2,6); zeros(6,2) zeros(6)];