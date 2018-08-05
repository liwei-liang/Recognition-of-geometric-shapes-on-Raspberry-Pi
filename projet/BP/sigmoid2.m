function [ Y ] = sigmoid( X )
%SIGMOID Summary of this function goes here
%   Detailed explanation goes here
    Y = (2./(1+exp(-X)))-1;

end

