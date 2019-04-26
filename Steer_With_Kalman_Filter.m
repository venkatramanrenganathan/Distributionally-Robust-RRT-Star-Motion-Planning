function [z, Sigma] = Steer_With_Kalman_Filter(xnear, Snear, xrand, A, B, Q, QT, R, T, W)
% Steer system from xnear toward xrand
%
% Inputs:   xnear = intial state on current tree
%           xrand = random state in configuration space
%
% Outputs: x = collision free trajectory from xnear toward xrand 

    xnear = xnear';
    xrand = xrand';

    n = size(A,1);
    m = size(B,2);

    % run dynamic programming to compute optimal controller
    K = zeros(m,n,T);
    k = zeros(m,T);
    P = zeros(n,n,T+1);
    p = zeros(n,T+1);  

    P(:,:,end) = QT;
    p(:,end)   = -QT*xrand;

    for t=T:-1:1
        P(:,:,t) = Q + A'*P(:,:,t+1)*A - A'*P(:,:,t+1)*B*inv(R+B'*P(:,:,t+1)*B)*B'*P(:,:,t+1)*A;
        K(:,:,t) = -inv(R+B'*P(:,:,t+1)*B)*B'*P(:,:,t+1)*A;
        k(:,t)   = -inv(R+B'*P(:,:,t+1)*B)*B'*p(:,t+1);
        p(:,t)   = A'*p(:,t+1) - Q*xrand + K(:,:,t)'*B'*p(:,t+1) + A'*P(:,:,t+1)*B*k(:,t) + K(:,:,t)'*(R+B'*P(:,:,t+1)*B)*k(:,t);
    end

    % propagate state mean and covariance under optimal controller
    x          = zeros(n,T+1);
    u          = zeros(m,T);
    x(:,1)     = xnear;
    V          = eye(n);
    C          = eye(n);
    H          = eye(n);
    G          = eye(n);
    P_x0       = zeros(n);
    P_x_est_0  = zeros(n);
    pi_0       = blkdiag(P_x0, P_x_est_0);
    Sigma_V    = zeros(n,n,T);
    S          = zeros(n,n,T+1);
    x_est      = zeros(n,T+1);
    x_est(:,1) = xnear; % Estimated State
    KG         = zeros(n,n,T+1); % Kalman Gain
    S(:,:,1)   = Snear; 
    A_bar      = zeros(2*n,2*n,T);
    pi         = zeros(2*n,2*n,T+1);
    pi(:,:,1)  = pi_0;

    for t=1:T    
        dummy          = randn(n,n);
        Sigma_V(:,:,t) = 0.001*eye(n);
        u(:,t)         = K(:,:,t)*x_est(:,t) + k(:,t); % control uses estimated state
        x(:,t+1)       = A*x(:,t) + B*u(:,t);
        KG(:,:,t)      = S(:,:,t) * C' * inv(C*S(:,:,t)*C' + H*Sigma_V(:,:,t)*H');
        x_est(:,t+1)   = KG(:,:,t)*C*A*x(:,t) + (eye(n) - KG(:,:,t)*C)*A*x_est(:,t) + B*u(:,t); 
        A_bar(:,:,t)   = [A             B*K(:,:,t)
                          KG(:,:,t)*C*A (eye(n)-KG(:,:,t)*C)*A+B*K(:,:,t) ];
        B_bar(:,:,t)   = [B;B];
        G_bar(:,:,t)   = [G             zeros(n)
                          KG(:,:,t)*C*G KG(:,:,t)*H]; 
        pi(:,:,t+1)    = A_bar(:,:,t)*pi(:,:,t)*A_bar(:,:,t)' + G_bar(:,:,t)*blkdiag(W, Sigma_V(:,:,t))*G_bar(:,:,t)'; 
        % Extract the true state covariance alone    
        S(:,:,t+1)     = [eye(n) zeros(n)] * pi(:,:,t+1) * [eye(n) zeros(n)]';      
    end

    z = x;
    Sigma = S;

end
