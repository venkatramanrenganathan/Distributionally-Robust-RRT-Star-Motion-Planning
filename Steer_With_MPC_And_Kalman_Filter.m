function [z, Sigma] = Steer_With_MPC_And_Kalman_Filter(xnear, Snear, xrand, A, B, C, Q, QT, R, Time, W)
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
    K = zeros(m,n,Time);
    k = zeros(m,Time);
    P = zeros(n,n,Time+1);
    p = zeros(n,Time+1); 
    
    P(:,:,end) = QT;
    p(:,end)   = -QT*xrand;
    
    for t=Time:-1:1
        P(:,:,t) = Q + A'*P(:,:,t+1)*A - A'*P(:,:,t+1)*B*inv(R+B'*P(:,:,t+1)*B)*B'*P(:,:,t+1)*A;
        K(:,:,t) = -inv(R+B'*P(:,:,t+1)*B)*B'*P(:,:,t+1)*A;
        k(:,t)   = -inv(R+B'*P(:,:,t+1)*B)*B'*p(:,t+1);
        p(:,t)   = A'*p(:,t+1) - Q*xrand + K(:,:,t)'*B'*p(:,t+1) + A'*P(:,:,t+1)*B*k(:,t) + K(:,:,t)'*(R+B'*P(:,:,t+1)*B)*k(:,t);
    end
    
    % propagate state mean and covariance under optimal controller
    x            = zeros(n,Time+1);
    u            = zeros(m,Time);
    x(:,1)       = xnear;
    V            = eye(n);
    C            = eye(n);
    H            = eye(n);
    Q            = eye(n);      
    R            = eye(m);
    G            = eye(n);    
    Sigma_V      = 0.001*eye(n);
    S            = zeros(n,n,Time+1);
    Sigma        = zeros(n,n,Time+1);
    x_est        = zeros(n,Time+1);
    x_est(:,1)   = xnear; % Estimated State
    Kal_Gain     = zeros(n,n,Time+1); % Kalman Gain
    S(:,:,1)     = Snear;    
    P_x0         = zeros(n);
    P_x_est_0    = zeros(n);    
    pi           = zeros(2*n,2*n,Time+1);
    pi(:,:,1)    = blkdiag(P_x0, P_x_est_0);
    Sigma(:,:,1) = zeros(n);

    % dynamics block matrices    
    BB    = kron(eye(Time),B);
    psi   = zeros(n*Time,n);
    theta = eye(n*Time);
    QQ    = kron(eye(Time),Q);
    RR    = kron(eye(Time),R);
    K     = zeros(n,n,Time);    

    for i=1:Time        
        psi((i-1)*n+1:n*i,:) = C*A^i;
    end

    for i=1:Time
        for j=1:Time
            if i > j                
                theta((i-1)*n+1:n*i,(j-1)*n+1:n*j) = C*A^(i-j);
            end
        end
    end
    
    theta = theta*BB;
    
    for t=1:Time+1           
        Kal_Gain(:,:,t) = S(:,:,t) * C' * inv(C*S(:,:,t)*C' + H*Sigma_V*H');        
        x_est(:,t+1)    = Kal_Gain(:,:,t)*C*x(:,t) + (eye(n) - Kal_Gain(:,:,t)*C)*x_est(:,t); 
        control_gain    = inv(theta'*QQ*theta + RR)*theta'*QQ;
        u(:,t)          = [eye(m) zeros(m,m*Time-m)]*control_gain* (repmat(xrand',1,Time)' - psi*x_est(:,t+1)); % control uses estimated state
        x(:,t+1)        = A*x(:,t) + B*u(:,t);
        K(:,:,t)        = A*Kal_Gain(:,:,t);        
        x_est(:,t+1)    = (A - K(:,:,t)*C)*x_est(:,t) + K(:,:,t)*C*x(:,t) + B*u(:,t);
        S(:,:,t+1)      = A*S(:,:,t)*A' - K(:,:,t)*(C*S(:,:,t)*C' + H*Sigma_V*H')*K(:,:,t)' + G*W*G';  
    end

    z = x;
    Sigma = S;
end
