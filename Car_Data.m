% Double integrator data

% Euler discretized discrete time double integrator
dt = 0.01; % time discretization period
Th = 10;   % steering time horizon
% dynamics matrix

v_0     = 10;
len     = 0.05;
phi_0   = 90*pi/180;
theta_0 = pi/2;

Acts = [0 0 -v_0*sin(phi_0+theta_0) -v_0*sin(phi_0+theta_0) 
        0 0 v_0*cos(phi_0+theta_0)  v_0*cos(phi_0+theta_0)        
        0 0 0               v_0*cos(phi_0)/len
        0 0 0               0];
Bcts = [cos(phi_0+theta_0)      0
        sin(phi_0+theta_0)      0 
        sin(phi_0)/len  0
        0               1];
n = size(Acts,1);
m = size(Bcts,2);
A = expm(dt*Acts);
B = dt*Bcts + 0.5*dt*Acts*Bcts + (1/6)*dt^2*Acts^2*Bcts;
eig(A)
G = B; % disturbance input matrix

% run dynamic programming to compute optimal controller
Q        = 0.4*eye(n) ; % state stage cost
QT       = 10*eye(n); % state terminal cost
R        = 20*eye(m); % input cost
diag_vec = n:-1:1;
W        = 0.0001*diag(diag_vec); % disturbance covariance

% initialize
x0 = [0.5 0 0 0]; % tree root (initial state mean)
S0 = [0.001*eye(2) zeros(2); zeros(2) zeros(2)];
