function s = rand_free(path_check_param)
% Generates random point not in obstacles
%
% Inputs: Obstacle matrix (N_obs x 4)
%         where each row specifies a rectangular obstacle: [px, py, sx, sy]
%         where (px, py) = bottom left corner position and (sx, sy) are
%         side lengths
% Outputs: s = sample from obstacle free space

xrand              = [rand rand];
path_check_param.x = xrand;

while ~CheckCollision(path_check_param)
    xrand              = [rand rand];
    path_check_param.x = xrand;
end

s = xrand;