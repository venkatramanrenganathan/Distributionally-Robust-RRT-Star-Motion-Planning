% Rapidly Exploring Random Tree (RRT) - Star
% Builds and visulaizes an RRT-Star in the plane with rectangular obstacles and
% nth-order integrator dynamics
% Implements Distributionally Robust Collison Check

clear all; close all; clc;
dbstop if error;

% Load the Double Integrator Data
Data_DoubleIntegrator;

%% RRT-Star parameters - State Space: [0,1]^2

q_start.coord  = [x0(1) x0(2)];
q_start.cost   = 0;
q_start.parent = 0;
nodes(1)       = q_start;
dx             = 0.1; % step size toward random sample
radius         = dx;  % radius for neighbor search
T.nodes        = x0;
T.costs        = 0;

%% Initialize Parameters of Tree
num_nodes            = 600; %number of nodes
T.NodeMeans{1}       = x0;
T.NodeCovariances{1} = S0; 
P0                   = CostToGo(A, B, Q, QT, R, Th); % optimal unconstrained cost to go matrix (to origin)

%% visualize
figure;
plot(x0(1), x0(2), 'O', 'Color', 'r', 'MarkerSize',15, 'MarkerFaceColor', 'r');
axis([0 1 0 1]);
set(gca, 'fontsize', 20)
hold on;

%% obstacles (randomly placed and sized rectangles)
Nobs        = 5;
Obstacles   = [0.95*rand(Nobs,2) 0.05 + 0.05*rand(Nobs,2)];
alpha       = 0.05*ones(Nobs,1); % 0.01 + (0.05-0.01) .* rand(Nobs,1);
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
    
    % radius for searching is a function of number of nodes visited
    radius = dx*sqrt(log(k)/k); 
    
    % Get a random sampling point in free space with 0 velocity & plot it             
    xrand = [rand_free(path_check_param) zeros(1,length(x0) - 2)];        
    
    % Pick the closest node from existing list to branch out from    
    [xnearest, Snearest] = NearestStateDistribution(xrand, T, P0); % find nearest vertex in the current tree   
    
    % Steer from xnearest to xrand or else reach the best xnew node
    [xnew, Snew] = Steer_With_Kalman_Filter(xnearest, Snearest, xrand, A, B, Q, QT, R, Th, W); 
    
    % Perform Distributionally Robust Collision Check
    collision_check_flag = 1;
    for i=2:size(xnew,2)        
        path_check_param.alpha      = alpha;
        path_check_param.cov_matrix = Snew(:,:,i);
        path_check_param.Obstacles  = Obstacles;
        path_check_param.x          = xnew(:,i); 
        collision_check_flag        = collision_check_flag*(~DRCheckCollision(path_check_param));
    end
    
    x_new = xnew(:,size(xnew,2))';
    S_new = Snew(:,:,size(xnew,2)-1); 

    if collision_check_flag % no collisions  
        line([xnearest(1), x_new(1)], [xnearest(2), x_new(2)], 'Color', 'k', 'LineWidth', 2);
        drawnow
        hold on                

        % Within a radius of r, find all nearest vertices in current tree
        % Xnear is cell containing all nearby nodes as its elements
        % Snear is cell containing the covariances of nearby nodes as its elements
        search_radius  = min(5*sqrt(log(length(nodes))/length(nodes)),radius);  
        [Xnear, Snear] = NearestStateDistributions(x_new, T, P0, search_radius);        
        
        % Initialize best cost to currently known value
        x_min   = xnearest;        
        % compute cost for new vertex to nearest one            
        T.costs = [T.costs; T.costs(find(T.NodeMeans == kron(ones(size(T.NodeMeans,1),1), xnearest),1)) + (xnearest - x_new)*P0*(xnearest - x_new)']; 
        c_min   = T.costs(end);

        T.NodeMeans{end+1}       = x_new; % add vertex to tree
        T.NodeCovariances{end+1} = S_new; 

        % Iterate through all nearest neighbors to find alternate lower
        % cost paths and connect along the minimum cost path           
        collision_check_flag = 1;
        for j=1:size(Xnear,2)  
            % Perform Distributionally Robust Collision Check
            path_check_param.alpha      = alpha;
            path_check_param.cov_matrix = Snear{j};
            path_check_param.Obstacles  = Obstacles;
            path_check_param.x          = Xnear{j}'; 
            collision_check_flag        = collision_check_flag*(~DRCheckCollision(path_check_param));
            XNear_Cost = T.costs(find(T.NodeMeans == kron(ones(size(T.NodeMeans,2),1), Xnear{j}),1));
            c_new      = XNear_Cost + (Xnear{j} - x_new)*P0*(Xnear{j} - x_new)';
            if c_min > c_new && collision_check_flag
                x_min = Xnear{j};                
                c_min = c_new;                    
            end
        end        

        T.edges{end+1} = [x_min' x_new'];
        line([x_min(1), x_new(1)], [x_min(2), x_new(2)], 'Color', 'g'); 
        drawnow; hold on;
        T.costs(end) = c_min;           

        % rewire vertices in Xnear if improved cost and delete edges to maintain tree structure 
        collision_check_flag = 1;
        for j=1:length(Xnear)            
            path_check_param.alpha      = alpha;
            path_check_param.Obstacles  = Obstacles;
            path_check_param.cov_matrix = Snear{j};
            path_check_param.x          = Xnear{j}'; 
            collision_check_flag        = collision_check_flag*(~DRCheckCollision(path_check_param));
            
            if collision_check_flag && (T.costs(Xnear(j,:)) > c_min + (Xnear(j,:) - x_new)*P0*(Xnear(j,:) - x_new))
                vparent = find(T.edges(:,3:4) == kron(ones(size(T.edges,1),1), T.NodeMeans(Xnear(j,:),:)), 1); % find old parent of Xnear(j)
                xparent = T.NodeMeans(vparent);
                T.edges = [T.edges; x_new T.nodes(Xnear(j),:)]; % add better edge
                line([x_new(1) T.NodeMeans(Xnear(j,:),1)], [x_new(2) T.NodeMeans(Xnear(j,:),2)], 'Color', [0 .5 .5], 'LineWidth', 1);
                line([T.edges(vparent,1) T.edges(vparent,3)], [T.edges(vparent,2) T.edges(vparent,4)], 'Color', [1 1 1], 'LineWidth', 1); % HACK: add a white line!
                T.edges(vparent,:) = []; % delete edge to maintain tree
            end
            
        end
    end 
    % Plot the covariance ellipse if no collision.
    % If there is collision, plot intersecting ellipse
    [V,E] = eig(S_new(1:2,1:2));
    h_1 = ellipse(sqrt(E(1,1)), sqrt(E(2,2)), atan2(V(1,2), V(1,1)), xnew(1), xnew(2));  
    pause(0.0005);
    
end