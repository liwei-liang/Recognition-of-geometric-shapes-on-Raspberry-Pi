function [ Y ] = sigmoidDerive( X )
%SIGMOIDDERIVE Summary of this function goes here
%   Detailed explanation goes here
    Y=2*(1./(1+exp(-X))).*(1-(1./(1+exp(-X))));

end

