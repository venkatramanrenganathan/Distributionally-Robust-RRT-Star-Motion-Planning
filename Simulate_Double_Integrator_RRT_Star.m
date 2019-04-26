% Rapidly Exploring Random Tree (RRT) - Star
% Builds and visulaizes an RRT-Star in the plane with rectangular obstacles and
% nth-order integrator dynamics

clear all; close all; clc;

% Load the Double Integrator Data
Data_DoubleIntegrator;

%% initialize parameters
K                 = 600; %number of nodes
T.NodeMeans       = x0;
T.NodeCovariances = S0; 
T.edges           = [];
P0                = CostToGo(A, B, Q, QT, R, Th); % optimal unconstrained cost to go matrix (to origin)

%% visualize
figure;
plot(x0(1), x0(2), 'o', 'Color', [0 .5 .5]);
axis([0 1 0 1]);
set(gca, 'fontsize', 20)
hold on;

%% obstacles (randomly placed and sized rectangles)
Nobs        = 5;
Obstacles   = [0.95*rand(Nobs,2) 0.05 + 0.05*rand(Nobs,2)];
alpha       = 0.01 + (0.05-0.01) .* rand(Nobs,1);
relax_param = 0.05;

for i=1:size(Obstacles,1)
    rectangle('Position',Obstacles(i,:),'FaceColor',[0 0 0 0.9]);
end


%% RRT-Star parameters - State Space: [0,1]^2
dx      = 0.1; % step size toward random sample
radius  = dx;  % radius for neighbor search
T.nodes = x0;
T.costs = 0;
T.edges = [];

% build tree
for k=1:K    
    k     
    
    path_check_param.relax_param = 0.05;
    path_check_param.Obstacles   = Obstacles;
    random_sample_point          = rand_free(path_check_param); % sample position from obstacle-free space (with zero velocity)    
    xrand                        = [random_sample_point zeros(1,length(x0) - 2)]; 
    
    plot(xrand(1), xrand(2), 'x', 'Color', [0 .5 .5]);
    
    [xnearest, Snearest] = NearestStateDistribution(xrand, T, P0); % find nearest vertex in the current tree    
    [xnew, Snew]         = Steer_With_Kalman_Filter(xnearest, Snearest, xrand, A, B, Q, QT, R, Th, W); % steer system to xrand
    
    for i=2:size(xnew,2) 
        
        x_new = xnew(:,i)';
    
        % Perform Distributionally Robust Collision Check
        path_check_param.alpha      = alpha;
        path_check_param.cov_matrix = Snew(:,:,i-1);
        path_check_param.Obstacles  = Obstacles;
        path_check_param.x          = [x_new(1) x_new(2)];
        
        if DRCheckCollision(path_check_param) % no collisions              
            
            
            [Xnear, Snear]    = NearestStateDistributions(xrand, T, P0,radius); % find nearest vertex in the current tree    
            T.NodeMeans       = [T.NodeMeans; x_new]; % add vertex to tree
            T.NodeCovariances = [T.NodeCovariances; Snew(:,:,i)]; 
            
            x_min   = xnearest;
%             T.nodes = [T.nodes; xnew]; % add new vertex to tree
            T.costs = [T.costs; T.costs(find(T.NodeMeans == kron(ones(size(T.NodeMeans,1),1), xnearest),1))  + norm(xnearest - x_new)]; % compute cost for new vertex to nearest one
            c_min   = T.costs(end);
            
            % Connect along the minimum cost path
            for j=1:length(Xnear)
                if T.costs(Xnear(j)) + norm(Xnear(j) - x_new) < c_min
                    x_min = T.NodeMeans(Xnear(j),:);
                    c_min = T.costs(Xnear(j)) + norm(T.NodeMeans(Xnear(j),:) - x_new);
                end
            end
            T.edges = [T.edges; x_min x_new];            
            line([x_min(1) x_new(1)], [x_min(2) x_new(2)], 'Color', [0 .5 .5], 'LineWidth', 1);
            T.costs(end) = c_min;
            
            % rewire vertices in Xnear if improved cost and delete edges to maintain tree structure TODO
            for j=1:length(Xnear)
                if c_min + norm(T.NodeMeans(Xnear(j),:) - x_new) < T.costs(Xnear(j))
                    vparent = find(T.edges(:,3:4) == kron(ones(size(T.edges,1),1), T.NodeMeans(Xnear(j),:)), 1); % find old parent of Xnear(j)
                    xparent = T.NodeMeans(vparent);
                    T.edges = [T.edges; x_new T.nodes(Xnear(j),:)]; % add better edge
                    line([x_new(1) T.NodeMeans(Xnear(j),1)], [x_new(2) T.NodeMeans(Xnear(j),2)], 'Color', [0 .5 .5], 'LineWidth', 1);
                    line([T.edges(vparent,1) T.edges(vparent,3)], [T.edges(vparent,2) T.edges(vparent,4)], 'Color', [1 1 1], 'LineWidth', 1); % HACK: add a white line!
                    T.edges(vparent,:) = []; % delete edge to maintain tree
                end
            end
                        
            % Plot the covariance ellipse
            if i==size(xnew,2)
                [V,E] = eig(Snew(1:2,1:2,i-1));
                h_1 = ellipse(sqrt(E(1,1)), sqrt(E(2,2)), atan2(V(1,2), V(1,1)), xnew(1,i-1), xnew(2,i-1));  
                pause(0.0005);
            end
            
        else
            
            % If there is collision, plot intersecting ellipse & break
            [V,E] = eig(Snew(1:2,1:2,i-1));
            h_1 = ellipse(sqrt(E(1,1)), sqrt(E(2,2)), atan2(V(1,2), V(1,1)), xnew(1,i-1), xnew(2,i-1));            
            pause(0.0005);
            break;
            
        end
            
    end
    
end


