function hits = DRCheckCollision(path_check_param)
% Checks for obstacle collisions in a distributionally robust way
%
% Inputs:   Obstacle matrix (N_obs x 4)
%           where each row specifies a rectangular obstacle: [px, py, sx, sy]
%           where (px, py) = bottom left corner position and (sx, sy) are
%           side lengths
% Outputs:  returns True or False if collision or not
% Function: Constructs a voronoid rectangle over the existing obstacle and
%           checks if the robot position is within the voronoid space.

alpha      = path_check_param.alpha;
cov_matrix = path_check_param.cov_matrix;
Obstacles  = path_check_param.Obstacles;
x          = path_check_param.x;

hits = 0;
a_x  = [1 0 0 0]';    
a_y  = [0 1 0 0]'; 
x_v  = [0 0 1 0]';
y_v  = [0 0 0 1]';

add_p = 0.05; % Should be 0

for i = 1:size(Obstacles,1) % check if sample hits an obstacle
    
    x_tight   = sqrt((1-alpha(i))/alpha(i))*norm(cov_matrix*Obstacles(i,:)') + add_p;
    y_tight   = sqrt((1-alpha(i))/alpha(i))*norm(cov_matrix*Obstacles(i,:)') + add_p;
    x_v_tight = sqrt((1-alpha(i))/alpha(i))*norm(cov_matrix*Obstacles(i,:)') + add_p;
    y_v_tight = sqrt((1-alpha(i))/alpha(i))*norm(cov_matrix*Obstacles(i,:)') + add_p;
    
    if Obstacles(i,:)*x >= Obstacles(i,1) - x_tight && ...
       Obstacles(i,:)*x <= Obstacles(i,1) + Obstacles(i,3) + x_v_tight && ...
       Obstacles(i,:)*x >= Obstacles(i,2) - y_tight && ...
       Obstacles(i,:)*x <= Obstacles(i,2) + Obstacles(i,4) + y_v_tight
        hits = hits + 1;
        return
    end 
end
