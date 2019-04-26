function v = nearest_vertex(x, T)
% Finds nearest vertex between sample and current tree in terms of
% Euclidean distance
%
% Inputs: x = random sample from state space (point in plane)
%         T = current tree (struct with lists of nodes and edges)
% Outputs: v = coordinates of nearest vertex

d = inf;
for i=1:size(T.nodes, 1)
    if size(T.nodes, 1) == 1 
        q = T.nodes;
        d = norm(T.nodes - x);
    else
        if norm(T.nodes(i,:) - x) < d
            q = T.nodes(i,:);
            d = norm(q - x);
        end
    end
end
v = q;