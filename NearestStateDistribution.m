function [xnear, Snear] = NearestStateDistribution(x, T, P)
% Finds nearest vertex between sample and current tree in terms of
% optimal cost-to-go distance
%
% Inputs: x = random sample from state space (point in plane with zero velocity)
%         T = current tree (struct with lists of node means, covariances, and edges)
%         P = optimal cost to go matrix (to origin)
% Outputs: xnear = state of nearest vertex
%          Snear = covariance matrix of nearest vertex
    
    n = length(x);
    d = inf;
    
    for i=1:size(T.NodeMeans, 1)
        if size(T.NodeMeans, 1) == 1 
            q = T.NodeMeans;
            d = (q - x)*P*(q - x)';
            Q = T.NodeCovariances;
        else
            if (T.NodeMeans(i,:) - x)*P*(T.NodeMeans(i,:) - x)' < d
                q = T.NodeMeans(i,:);
                d = (q - x)*P*(q - x)';
                Q = T.NodeCovariances(n*(i-1)+1:n*i,:);
            end
        end
    end

    xnear = q;
    Snear = Q; 

end