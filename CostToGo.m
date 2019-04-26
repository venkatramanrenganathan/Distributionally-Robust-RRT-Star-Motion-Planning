function P0 = CostToGo(A, B, Q, QT, R, T)
% Computes optimal cost to go function (to origin)
%
% Inputs:  
%
n = size(A, 1);

P = zeros(n,n,T+1);

P(:,:,end) = QT;

for t=T:-1:1
    P(:,:,t) = Q + A'*P(:,:,t+1)*A - A'*P(:,:,t+1)*B*inv(R+B'*P(:,:,t+1)*B)*B'*P(:,:,t+1)*A;
end

P0 = P(:,:,1);
