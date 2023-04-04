function [im, alphaGridAllViews,d2GridAllViews,projDataWF] = ifanbeanMy(projData, D, senSpacing, rotIncr, Imsz)
    im = 0;
    [M, N]= size(projData); % must be odds
    if nargout>1
        alphaGridAllViews = zeros(Imsz(1), Imsz(2), N);
        d2GridAllViews = zeros(Imsz(1), Imsz(2), N);
        projDataWF = zeros(size(projData));
    end
    if mod(M,2)
        senSpacingRad = senSpacing*pi/180;
        senAnglesRad = ((1:M)-(M+1)/2)*senSpacingRad;
        rotAnglesRad = (0:N-1)*rotIncr*pi/180;
        fltRamp = (D/2)*rampfilter(M, senSpacing);
        save('w_b.mat','fltRamp');
        projDataW = projData.*cos(senAnglesRad');
        rotCenIm = floor((Imsz+1)/2);
        
        [xGrid, yGrid] = meshgrid((1:Imsz(2))-rotCenIm(2), (1:Imsz(1))-rotCenIm(1));
        
%         srcXY = [D*sin(rotAnglesRad); D*cos(rotAnglesRad)]; % xy coordinates of the ray source in the image centered coordinate system
%         xGridCenSrc = single(xGrid)-single(reshape(srcXY(1,:),1,1,[])); % xGrid for all views centered at ray source
%         yGridCenSrc = single(yGrid)-single(reshape(srcXY(2,:),1,1,[])); % yGrid for all views centered at ray source
%         d2Grid = xGridCenSrc.^2 + yGridCenSrc.^2; % the square of the distances between every pixel and ray source 
%         xGridSrc = xGridCenSrc.*reshape(single(cos(rotAnglesRad)),1,1,[]) - yGridCenSrc.*reshape(single(sin(rotAnglesRad)),1,1,[]); % xGrid in the ray-source coordinate system (O_source)
%         alphaGridAllViews = asin(xGridSrc./sqrt(d2Grid))/single(senSpacingRad) + single((M+1)/2); % alpha index for each pixel
%         
        for view = 1:N
            theta = rotAnglesRad(view);
            srcXY = [D*sin(theta); D*cos(theta)]; % xy coordinates of the ray source in the image centered coordinate system
            xGridCenSrc = xGrid-srcXY(1); % xGrid centered at ray source
            yGridCenSrc = yGrid-srcXY(2); % yGrid centered at ray source
            d2Grid = xGridCenSrc.^2 + yGridCenSrc.^2; % the square of the distances between every pixel and ray source 

            xGridSrc = xGridCenSrc*cos(theta) - yGridCenSrc*sin(theta); % xGrid in the ray-source coordinate system (O_source)
            alphaGridId = asin(xGridSrc./sqrt(d2Grid))/senSpacingRad + (M+1)/2; % alpha index for each pixel
            alphaGridIdFlr = floor(alphaGridId);
%             alphaGridId = alphaGridAllViews(:,:,view);
%             alphaGridIdFlr = floor(alphaGridId);
            
            projLine = conv(projDataW(:,view),fltRamp','same');
            im = im + (projLine(alphaGridIdFlr).*(1-alphaGridId+alphaGridIdFlr) + ...
            projLine(alphaGridIdFlr+1).*(alphaGridId-alphaGridIdFlr))./d2Grid;
        
            if nargout>1
                d2GridAllViews(:,:,view) = d2Grid;
                alphaGridAllViews(:,:,view) = alphaGridId;
                projDataWF(:,view) = projLine; 
            end
        end
    end
end


function filt = rampfilter(M, senSpacing)

    % compute the standard ramp filter
    x = ((-M+1):(M-1));
    msk = (x~=0);
    filt = ones(size(x))/4;
    filt(msk)=((cos(pi*x(msk))-1)./((pi*x(msk)).^2) + sin(pi*x(msk))./(pi*x(msk)))/2;
    
    % compute the weighted ramp filter with (alpha/sin(alpha))^2
    if nargin==2
        senSpacingRad = senSpacing*pi/180;
        senAnglesRad = ((1:M)-(M+1)/2)*senSpacingRad;
        alphaRampFilter = [senAnglesRad(1)+senAnglesRad(1:(M-1)/2),senAnglesRad,senAnglesRad(end)+senAnglesRad((M+1)/2+1:end)];
        msk = (alphaRampFilter~=0);
        weights = ones(size(alphaRampFilter));
        weights(msk) = alphaRampFilter(msk)./sin(alphaRampFilter(msk));
        filt = (weights.^2).*rampfilter(M);
    end 
end