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
dx             = 0.5; % step size toward random sample
T.nodes        = x0;
T.edges        = {};   
T.costs{1}     = 0;

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
    [xnearest, Snearest] = NearestStateDistribution(xrand, T, P0); 
    
    % Steer from xnearest to xrand or else reach the best xnew node
    [xnew, Snew] = Steer_With_Kalman_Filter(xnearest, Snearest, xrand, A, B, Q, QT, R, Th, W); 
    
    collision_check_flag = 1;
    for i=2:size(xnew,2)
        x_new                       = xnew(:,i)'; % xnew(:,size(xnew,2))';
        S_new                       = Snew(:,:,i-1); % Snew(:,:,size(xnew,2)-1);    
        % Perform Distributionally Robust Collision Check
        path_check_param.alpha      = alpha;
        path_check_param.cov_matrix = S_new;
        path_check_param.Obstacles  = Obstacles;
        path_check_param.x          = x_new';
        collision_check_flag        = collision_check_flag*(~DRCheckCollision(path_check_param));
    end
    if collision_check_flag % no collisions 
        
        % add vertex to tree
        T.NodeMeans{end+1}       = x_new; 
        T.NodeCovariances{end+1} = S_new; 
%         line([xnearest(1), x_new(1)], [xnearest(2), x_new(2)], 'Color', 'k', 'LineWidth', 2);        
%         drawnow
%         hold on 
        
        % Within a search radius, find all nearest vertices in current tree
        % Xnear is a cell containing all nearby nodes as its elements
        % Snear is a cell containing the covariances of nearby nodes as its elements
        search_radius  = min(5*sqrt(log(length(T.NodeMeans))/length(T.NodeMeans)),radius);  
        [Xnear, Snear] = NearestStateDistributions(x_new, T, P0, search_radius);

        % Initialize best cost to currently known value
        % compute cost for new vertex to nearest one
        x_min     = xnearest;                        
        near_cost = compute_cost(xnearest, T);
        min_cost  = near_cost + (xnearest - x_new)*P0*(xnearest - x_new)';                
        % Iterate through all nearest neighbors to find alternate lower
        % cost paths and connect along the minimum cost path                  
        for j=1:length(Xnear)
            if ~isempty(Xnear)
                % Steer from x_new tp Xnear{j}
                [xnow, Snow] = Steer_With_Kalman_Filter(x_new, S_new, Xnear{j}, A, B, Q, QT, R, Th, W);
                path_check_flag = 1;
                for l=2:size(xnow,2)                        
                    x_now = xnow(:,l)';
                    S_now = Snow(:,:,l);                         
                    % Perform Distributionally Robust Collision Check
                    path_check_param.alpha      = alpha;
                    path_check_param.cov_matrix = S_now;
                    path_check_param.Obstacles  = Obstacles;
                    path_check_param.x          = x_now';
                    path_check_flag             = path_check_flag*(~DRCheckCollision(path_check_param));
                end
                if path_check_flag % no collisions                                                 
                    near_cost    = compute_cost(Xnear{j}, T); 
                    now_cost     = near_cost + (Xnear{j} - x_now)*P0*(Xnear{j} - x_now)';  
                    if (min_cost > now_cost)
                        x_min    = Xnear{j};                
                        min_cost = now_cost;                    
                    end
                end 
                
                x_end                    = xnow(:,end);
                T.edges{end+1}           = [x_min' x_end];
                line([x_min(1), x_end(1)], [x_min(2), x_end(2)], 'Color', 'k'); 
                drawnow; hold on;
                T.costs{end+1} = min_cost; 
            end
        end        

        % rewire vertices in Xnear if cost is improved 
        % Also delete the correspondng edges to maintain tree structure                 
        for j=1:length(Xnear)
            % Steer from x_new tp Xnear{j}
            [xnow, Snow] = Steer_With_Kalman_Filter(x_new, S_new, Xnear{j}, A, B, Q, QT, R, Th, W); 
            path_check_flag = 1;
            for l=2:size(xnow,2)                        
                x_now = xnow(:,l)';
                S_now = Snow(:,:,l);
                % Perform Distributionally Robust Collision Check
                path_check_param.alpha      = alpha;
                path_check_param.Obstacles  = Obstacles;
                path_check_param.cov_matrix = S_now;
                path_check_param.x          = x_now';  
                path_check_flag             = path_check_flag*(~DRCheckCollision(path_check_param));
            end
            if path_check_flag % no collisions
                now_cost     = compute_cost(x_now, T);  
                new_cost     = now_cost + (Xnear{j} - x_now)*P0*(Xnear{j} - x_now)';
                near_cost    = compute_cost(Xnear{j}, T); 
                if near_cost > now_cost + new_cost
                    j_node      = Xnear{j};
                    parent_node = get_parent(Xnear{j}, T);                             
                    for p = 1:length(T.edges) 
                        pth_edge  = T.edges{p};
                        edge_from = pth_edge(:,1);                               
                        if isequal(parent_node,edge_from) 
                            T.edges{p} = [];
                            line([parent_node(1), j_node(1)], [parent_node(2), j_node(2)], 'Color', 'w'); 
                            break;
                        end
                    end  
                    T.edges{end+1} = [x_now' Xnear{j}'];
                    line([x_now(1), j_node(1)], [x_now(2), j_node(2)], 'Color', 'k'); 
                end
            end            
        end
        % Plot the covariance ellipse if no collision. If there is collision, plot intersecting ellipse
        [V,E] = eig(S_new(1:2,1:2));
        h_1   = ellipse(sqrt(E(1,1)), sqrt(E(2,2)), atan2(V(1,2), V(1,1)), xnew(1), xnew(2));  
        pause(0.0005);    
    end    
end