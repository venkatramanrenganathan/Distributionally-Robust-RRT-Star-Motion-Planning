% Rapidly Exploring Random Tree (RRT*)
% Builds and visulaizes an RRT* in the plane with no obstacles

% parameters
dx = 0.1; % step size toward random sample
r = dx; % radius for neighbor search

% state space: [0,1]^2
x0 = [0.5, 0.5]; % tree root
K = 700; % number of nodes
T.nodes = x0;
T.costs = 0;
T.edges = [];

% visualize
figure;
plot(x0(1), x0(2), 'x', 'Color', [0 .5 .5]);
axis([0 1 0 1]);
hold on;

% build tree
for k=1:K
    xrand = [rand, rand]; % random sample from state space
    xnear = nearest_vertex(xrand, T); % find nearest vertex in current tree
    xnew = xnear + dx*sqrt(log(k)/k)*(xrand - xnear)/norm(xrand - xnear); % steer from nearest vertext toward sample to new vertex
    Xnear = nearest_vertices(xnew, T, min(5*log(k)/k,r)); % find set of vertices near new vertex
    T.nodes = [T.nodes; xnew]; % add new vertex to tree (TODO: collision check)
    T.costs = [T.costs; T.costs(find(T.nodes == kron(ones(size(T.nodes,1),1), xnear),1))  + norm(xnear - xnew)]; % compute cost for new vertex to nearest one
    plot(xnew(1), xnew(2), '.', 'Color', [0 .5 .5]);
    
    % add edge to one of set of near vertices if cost is improved
    xmin = xnear;
    cmin = T.costs(end);
    for j=1:length(Xnear)
        if T.costs(Xnear(j)) + norm(Xnear(j) - xnew) < cmin
            xmin = T.nodes(Xnear(j),:);
            cmin = T.costs(Xnear(j)) + norm(T.nodes(Xnear(j),:) - xnew);
        end
    end
    T.edges = [T.edges; xmin xnew];
    line([xmin(1) xnew(1)], [xmin(2) xnew(2)], 'Color', [0 .5 .5], 'LineWidth', 1);
    T.costs(end) = cmin;
    
    % rewire vertices in Xnear if improved cost and delete edges to maintain tree structure TODO
    for j=1:length(Xnear)
        if cmin + norm(T.nodes(Xnear(j),:) - xnew) < T.costs(Xnear(j))
            vparent = find(T.edges(:,3:4) == kron(ones(size(T.edges,1),1), T.nodes(Xnear(j),:)), 1); % find old parent of Xnear(j)
            xparent = T.nodes(vparent);
            T.edges = [T.edges; xnew T.nodes(Xnear(j),:)]; % add better edge
            line([xnew(1) T.nodes(Xnear(j),1)], [xnew(2) T.nodes(Xnear(j),2)], 'Color', [0 .5 .5], 'LineWidth', 1);
            line([T.edges(vparent,1) T.edges(vparent,3)], [T.edges(vparent,2) T.edges(vparent,4)], 'Color', [1 1 1], 'LineWidth', 1); % HACK: add a white line!
            T.edges(vparent,:) = []; % delete edge to maintain tree
        end
    end
    
    pause(0.0002);
    k
end
   