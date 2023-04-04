function normIm = normimage(I, maxV, minV)
    if  (nargin < 2)
        maxV = max(max(I));
        minV = min(min(I));
    end
    normIm = uint8((I-minV)*255/(maxV-minV));
end