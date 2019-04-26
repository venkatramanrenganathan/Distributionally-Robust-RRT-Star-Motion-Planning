function status = CheckCollision(path_check_param)
% Checks for obstacle colisions
%
% Inputs:   Obstacle matrix (N_obs x 4)
%           where each row specifies a rectangular obstacle: [px, py, sx, sy]
%           where (px, py) = bottom left corner position and (sx, sy) are
%           side lengths
% Outputs:  returns True or False if collision or not
% Function: Constructs a voronoid rectangle over the existing obstacle and
%           checks if the robot position is within the voronoid space.



hits        = 0;
relax_param = path_check_param.relax_param;
Obstacles   = path_check_param.Obstacles;
x           = path_check_param.x;

for i=1:size(Obstacles,1) % check if sample hits an obstacles
    if x(1) >= Obstacles(i,1) - relax_param && ...
       x(1) <= Obstacles(i,1) + Obstacles(i,3) + relax_param && ...
       x(2) >= Obstacles(i,2) - relax_param && ...
       x(2) <= Obstacles(i,2) + Obstacles(i,4) + relax_param     
        hits = hits + 1;
    end
end

status = (hits == 0); % sample does not hit any obstacle
