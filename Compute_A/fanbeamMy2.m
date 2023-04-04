function [projData,sensAngl, rotAngl, rotCenIm] = fanbeamMy2(I, D, senSpacing, rotIncr, accNum)

if (nargin < 5)
    accNum = 1;
    if (nargin < 4)
        rotIncr = 1;
    end
end

rotCenIm = floor((size(I)+1)/2); % pixel coordinates of the rotation center 
numSenHalf = ceil(asin(norm(size(I)-rotCenIm+1)/D)*(180/pi)/senSpacing);
sensAngl = (-numSenHalf:numSenHalf)'*senSpacing;
sensAnglRad = sensAngl*pi/180;
numTheta = 360/rotIncr;
projData = zeros(length(sensAngl),numTheta);
rotAngl = (0:numTheta-1)*rotIncr;
subStep = 1/accNum;

% t0 = cputime;
source0 = [0, D]; % the inital coordinates (x0, y0) of source (the origin is at the rotation center) when theta = 0 (directly below the image)
raySteps = D-norm(size(I)-rotCenIm+1):subStep:D+norm(size(I)-rotCenIm+1);
rayXGrid = source0(1)+sin(sensAnglRad)*raySteps;
rayYGrid = source0(2)-cos(sensAnglRad)*raySteps;

% parfor (view = 1:numTheta, 4)
for view = 1:numTheta
    theta_rad = rotAngl(view)*pi/180;
    rayXGridTheta = rayXGrid*cos(theta_rad)+rayYGrid*sin(theta_rad)+rotCenIm(2);
    rayYGridTheta = rayYGrid*cos(theta_rad)-rayXGrid*sin(theta_rad)+rotCenIm(1);
    rayMatrix = interp2(I,rayXGridTheta,rayYGridTheta,'linear',0); 
    projData(:,view) = sum(rayMatrix,2);
end
projData=projData/accNum;
% t1 = cputime-t0

   