function [V, Q] = NearestStateDistributions(x, T, P, r)
% Finds set of vertices within a distance r of sample x in terms of
% optimal cost-to-go distance
%
% Inputs: x = random sample from state space (point in plane with zero velocity)
%         T = current tree (struct with lists of node means, covariances, and edges)
%         P = optimal cost to go matrix (to origin)
%         r = search radius
% Outputs: V = set of nearest vertices
% Outputs: xnear = state of nearest vertex
%          Snear = covariance matrix of nearest vertex

q = T.NodeMeans;
n = size(q, 1);
m = size(q, 2);
V = [];
Q = [];
for i = 1:n
    distances(i) = (q(i,:) - x)*P*(q(i,:) - x)'; % distances = sqrt(sum((q - kron(ones(n,1), x)).^2, 2));
    if distances(i) < r
        V = [V; q(i,:)];         
        Q = [Q; T.NodeCovariances{i}]; % Q = [Q; T.NodeCovariances(1+m*(i-1):m*i,:)]; - Old implementation
    end
end
    
end
