% Rapidly Exploring Random Tree (RRT) - Star
% Builds and visulaizes an RRT-Star in the plane with rectangular obstacles and
% nth-order integrator dynamics

clear all; close all; clc;

% Load the Double Integrator Data
Data_DoubleIntegrator;

%% RRT-Star parameters - State Space: [0,1]^2

q_start.coord  = [x0(1) x0(2)];
q_start.cost   = 0;
q_start.parent = 0;
q_goal.coord   = [999 999];
q_goal.cost    = 0;
nodes(1)       = q_start;
dx             = 0.1; % step size toward random sample
radius         = dx;  % radius for neighbor search
T.nodes        = x0;
T.costs        = 0;
T.edges        = [];

%% Initialize Parameters of Tree
num_nodes         = 600; %number of nodes
T.NodeMeans       = x0;
T.NodeCovariances = S0; 
P0                = CostToGo(A, B, Q, QT, R, Th); % optimal unconstrained cost to go matrix (to origin)

%% visualize
figure;
plot(x0(1), x0(2), 'O', 'Color', 'r', 'MarkerSize',15, 'MarkerFaceColor', 'r');
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

path_check_param.x_0         = x0;
path_check_param.relax_param = 0.05; % relaxing distance around the obstacles
path_check_param.Obstacles   = Obstacles; 

%% Build Tree

for k=1:num_nodes    
    k     
    
    if(k == 1)
        x_goal = [rand_free(path_check_param) zeros(1,length(x0) - 2)];     
        plot(x_goal(1), x_goal(2), 'O', 'Color', 'b', 'MarkerSize',15, 'MarkerFaceColor', 'b');  
    end
    
    % radius for searching is a function of number of nodes visited
    radius = dx*sqrt(log(k)/k); 
    
    % Get a random sampling point in free space with 0 velocity & plot it             
    xrand = [rand_free(path_check_param) zeros(1,length(x0) - 2)];     
    plot(xrand(1), xrand(2), 'x', 'Color', [0 .5 .5], 'MarkerSize',15);
    
    % Break if goal node is already reached
    for j = 1:1:length(nodes)
        if nodes(j).coord == q_goal.coord
            break
        end
    end
    
    % Pick the closest node from existing list to branch out from    
    [xnearest, Snearest] = NearestStateDistribution(xrand, T, P0); % find nearest vertex in the current tree   
    
    % Steer from xnearest to xrand or else reach the best xnew node
    [xnew, Snew] = Steer_With_Kalman_Filter(xnearest, Snearest, xrand, A, B, Q, QT, R, Th, W); 
    
    for i=2:size(xnew,2) 
    
        x_new = xnew(:,i)';
        S_new  = Snew(:,:,i-1); 

        % Perform Distributionally Robust Collision Check
        path_check_param.alpha      = alpha;
        path_check_param.cov_matrix = S_new;
        path_check_param.Obstacles  = Obstacles;
        path_check_param.x          = x_new'; %[x_new(1) x_new(2)];

        if DRCheckCollision(path_check_param) % no collisions  
            line([xnearest(1), x_new(1)], [xnearest(2), x_new(2)], 'Color', 'k', 'LineWidth', 2);
            drawnow
            hold on

            % compute cost for new vertex to nearest one
            %T.costs = [T.costs; T.costs(find(T.NodeMeans == kron(ones(size(T.NodeMeans,1),1), xnearest),1)) + norm(xnearest - x_new)]; 
            T.costs = [T.costs; T.costs(find(T.NodeMeans == kron(ones(size(T.NodeMeans,1),1), xnearest),1)) + (xnearest - x_new)*P0*(xnearest - x_new)']; 

            % Initialize best cost to currently known value
            x_min   = xnearest;        
            c_min   = T.costs(end);

            T.NodeMeans       = [T.NodeMeans; x_new]; % add vertex to tree
            T.NodeCovariances = [T.NodeCovariances; S_new]; 

            % Within a radius of r, find all nearest vertices in current tree
            [Xnear, Snear] = NearestStateDistributions(x_new, T, P0, radius);        

            % Iterate through all nearest neighbors to find alternate lower
            % cost paths and connect along the minimum cost path           

            for j=1:length(Xnear)   
                XNear_Cost = T.costs(find(T.NodeMeans == kron(ones(size(T.NodeMeans,1),1), Xnear(j)),1));
                if XNear_Cost + (Xnear(j) - x_new)*P0*(Xnear(j) - x_new) < c_min
                    x_min = Xnear(j);                
                    c_min = T.costs(Xnear(j)) + (Xnear(j) - x_new)*P0*(Xnear(j) - x_new);                    
                end
            end        

            T.edges = [T.edges; x_min x_new];
            line([x_min(1), x_new(1)], [x_min(2), x_new(2)], 'Color', 'g'); 
            T.costs(end) = c_min;
            

            % Update parent to least cost-from node
            for j = 1:1:length(T.NodeMeans)
                if T.NodeMeans(j)== x_min
                    x_new.parent = j;
                end
            end        

            % rewire vertices in Xnear if improved cost and delete edges to maintain tree structure 
            for j=1:length(Xnear)
                x_path             = Xnear(j);
                path_check_param.x = x_path'; %[x_path(1) x_path(2)];            
                if DRCheckCollision(path_check_param) && T.costs(Xnear(j)) > c_min + (Xnear(j) - x_new)*P0*(Xnear(j) - x_new)
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
                [V,E] = eig(S_new(1:2,1:2));
                h_1 = ellipse(sqrt(E(1,1)), sqrt(E(2,2)), atan2(V(1,2), V(1,1)), xnew(1), xnew(2));  
                pause(0.0005);
            end

        else

            % If there is collision, plot intersecting ellipse & break
            [V,E] = eig(S_new(1:2,1:2));
            h_1 = ellipse(sqrt(E(1,1)), sqrt(E(2,2)), atan2(V(1,2), V(1,1)), xnew(1), xnew(2));            
            pause(0.0005);
            break;

        end   
    end
    
end