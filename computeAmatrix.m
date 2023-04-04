function A = computeAmatrix(alpha3D, d2Grid3D, senNum)
    alpha3Dflr = floor(alpha3D);
    val2 = reshape(alpha3D-alpha3Dflr, [],1);
    d2 = reshape(d2Grid3D,[],1);
    M = senNum;
    N = size(alpha3D,3);
    Q = size(alpha3D,1)*size(alpha3D,2);
    rowId1 = repmat(1:Q, 1, N)';   
    colId1 = reshape(alpha3Dflr+reshape((0:N-1)*M,1,1,[]),[],1);
%     A = sparse([rowId1; rowId1], [colId1; colId1+1], [(1-val2)./d2; (val2)./d2], Q, M*N);
    A = [];
    I = [rowId1; rowId1];
    J = [colId1; colId1+1];
    index(1,:) = J;
    index(2,:) = I;
    val = [(1-val2)./d2; (val2)./d2];
    size_Proj = M*N;
    size_CT = Q;
    save('AT.mat','index','val','size_Proj','size_CT');
    
end