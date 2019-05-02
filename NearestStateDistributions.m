function [V, Q] = NearestStateDistributions(x, T, P, r)
% Finds set of vertices within a distance r of sample x in terms of
% optimal cost-to-go distance
%
% Inputs: x = random sample from state space (point in plane with zero velocity)
%         T = current tree (struct with lists of node means, covariances, and edges)
%         P = optimal cost to go matrix (to origin)
%         r = search radius
% Outputs: V = Cell of nearest vertices
%          Q = Cell of covariances of nearest vertices

q = T.NodeMeans;
n = length(q);
V = {};
Q = {};
for i = 1:n
    distances(i) = (q{i} - x)*P*(q{i} - x)'; % distances = sqrt(sum((q - kron(ones(n,1), x)).^2, 2));
    if distances(i) < r
        V{end+1} = q{i};         
        Q{end+1} = T.NodeCovariances{i}; % Q = [Q; T.NodeCovariances(1+n*(i-1):n*i,:)]; - Old implementation
    end
end
    
end
