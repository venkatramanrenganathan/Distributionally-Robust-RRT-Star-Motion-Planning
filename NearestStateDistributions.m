function [V, Q] = NearestStateDistributions(x, T, P, r)
% Finds set of vertices within a distance r of sample x in terms of
% optimal cost-to-go distance
%
% Inputs: x = random sample from state space (point in plane with zero velocity)
%         T = current tree (struct with lists of node means, covariances, and edges)
%         P = optimal cost to go matrix (to origin)
%         r = radius
% Outputs: V = set of nearest vertices
% Outputs: xnear = state of nearest vertex
%          Snear = covariance matrix of nearest vertex

    n         = size(T.NodeMeans, 1);
    q         = T.NodeMeans;
%     distances = sqrt(sum((q - kron(ones(n,1), x)).^2, 2));
    distances = (q - kron(ones(n,1), x))*P*(q - kron(ones(n,1), x))';
    Q         = T.NodeCovariances;
    V         = find(distances < r);

end
