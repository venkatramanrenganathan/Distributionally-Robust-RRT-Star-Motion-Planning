function V = nearest_vertices(x, T, r)
% Finds set of vertices within a distance r of sample x in terms of
% Euclidean distance
%
% Inputs: x = random sample from state space (point in plane)
%         T = current tree (struct with lists of nodes and edges)
%         r = radius
% Outputs: V = set of nearest vertices

n = size(T.nodes, 1);
distances = sqrt(sum((T.nodes - kron(ones(n,1), x)).^2, 2));
V = find(distances < r);