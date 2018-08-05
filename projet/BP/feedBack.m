function [ V,W ] = feedBack( P,Za,k,limite,lamda )
%FEEDBACK Summary of this function goes here
%   Detailed explanation goes here
    P = [P;ones(1,size(P,2))];
    [lp,cp] = size(P);
    [lZa,cZa] = size(Za);
    V=rand( lp,5)-1;
    W=rand( 6,lZa)-1;
    [Z,~,~,~]=feedForward(P,V,W);
    e=(sum(sum((Za-Z).^2)))^(0.5);
    while (e > limite)
        for i = 1:1:cp
            [Zs,Ys,Zc,Yc]=feedForward(P(:,i),V,W);
            deltaS = (Za(:,i)-Zs).*sigmoidDerive(Ys);
            deltaC = W([1:end-1],:)*deltaS;
            W=W+lamda*[Zc;1]*deltaS';
            V=V+lamda*P(:,i)*(deltaC.*sigmoidDerive(Yc))';
        end
        [Z,~,~,~]=feedForward(P,V,W)
        e=(sum(sum((Za-Z).^2)))^(0.5)
    end

end

