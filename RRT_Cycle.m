% Rapidly Exploring Random Tree (RRT)
% Builds and visulaizes an RRT in the plane with rectangular obstacles and
% nth-order integrator dynamics

clear all; close all; clc;

Data_Cycle;

%% initialize parameters
K                 = 624; % number of nodes
T.NodeMeans       = x0;
T.NodeCovariances = S0; 
T.edges           = [];
P0                = CostToGo(A, B, Q, QT, R, Th); % optimal unconstrained cost to go matrix (to origin)

% visualize
figure;
plot(x0(1), x0(2), 'x', 'Color', [0 .5 .5]);
axis([0 1 0 1]);
set(gca, 'fontsize', 20)
hold on;

% % obstacles (randomly placed and sized rectangles)
Nobs = 5;
Obstacles = [0.95*rand(Nobs,2) 0.05 + 0.05*rand(Nobs,2)];

for i=1:size(Obstacles,1)
    rectangle('Position',Obstacles(i,:),'FaceColor',[0 .5 .5 0.1]);
end

%% build tree
for k=1:K
    xrand = [rand_free(Obstacles) zeros(1,length(x0) - 2)]; % sample position from obstacle-free space (with zero velocity)
    plot(xrand(1), xrand(2), 'x', 'Color', [0 .5 .5]);
    [xnear, Snear] = NearestStateDistribution(xrand, T, P0); % find nearest vertex in the current tree
%     [xtraj, S] = steer(xnear, Snear, xnear + dx*(sqrt(log(k)/k))*(xrand - xnear)/norm(xrand - xnear)); % steer system toward xrand
%     [xtraj, S] = Steer(xnear, Snear, xrand, A, B, Q, QT, R, Th, W); % steer system to xrand
    [xtraj, S] = Steer_With_MPC_And_Kalman_Filter(xnear, Snear, xrand, A, B, Q, QT, R, Th, W); % steer system to xrand
    
    ellipse_flag = 0;
    
    for i=2:size(xtraj,2)
        if CheckCollision([xtraj(1,i) xtraj(2,i)], Obstacles) % no collisions            
            no_collision_flag = 1;
            T.NodeMeans = [T.NodeMeans; xtraj(:,i)']; % add vertex to tree
            T.NodeCovariances = [T.NodeCovariances; S(:,:,i)]; 
            T.edges = [T.edges; xnear xtraj(:,i)']; % add edge to tree
            % Plot the trajectory from current to next
            line([xtraj(1,i-1) xtraj(1,i)], [xtraj(2,i-1) xtraj(2,i)], 'Color', [0 .5 .5], 'LineWidth', 2);
            % Plot the covariance ellipse
            if i==size(xtraj,2)
                [V,E] = eig(S(1:2,1:2,i-1));
                h_1 = ellipse(sqrt(E(1,1)), sqrt(E(2,2)), atan2(V(1,2), V(1,1)), xtraj(1,i-1), xtraj(2,i-1));
                ellipse_flag = 1;
%                 pause(0.0005);
            end
        else
            % If there is collision, plot intersecting ellipse & break
            [V,E] = eig(S(1:2,1:2,i-1));
            h_1 = ellipse(sqrt(E(1,1)), sqrt(E(2,2)), atan2(V(1,2), V(1,1)), xtraj(1,i-1), xtraj(2,i-1));
            ellipse_flag = 1;
%             pause(0.0005);
            break;
        end
%         if(ellipse_flag)
%             set(h_1,'Visible','off')
%         end
    end
end
   