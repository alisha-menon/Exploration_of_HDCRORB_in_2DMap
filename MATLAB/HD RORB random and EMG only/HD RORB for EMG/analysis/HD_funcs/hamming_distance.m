% Function Name: Hamming distance
%
% Description: Calculates the cosine similarity between two binary hypervectors
%
% Arguments:
%   u - first hypervector
%   v - second hypervector
% 
% Returns:
%   sim - the hamming between u and v
%

function sim = hamming_distance(u, v)
    sim = sum(xor(u,v));
end
