function [ Z,Ys,Zint,Yint ] = feedForward( P,V,W )
%FEEDFORWARD Summary of this function goes here
%   Detailed explanation goes here
    %[~,c]=size(P);
    %P = [P; ones(1,c) ];
    Yint = V'*P;
    Zint=sigmoid(Yint);
    Xs=[Zint ; ones(1,size(Zint,2))];
    Ys=W'*Xs;
    Z=sigmoid(Ys);

end

