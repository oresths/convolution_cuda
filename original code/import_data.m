function [F, T, pF, Y] = ...
    import_data(mF,nF,mT,nT)
% function [dataset, index, centroid, sizes] = ...
%     import_data(numObjects,numAttributes,numClusters)
% Import data from binary files
% 'centroids.bin' 'dataset.bin' 'Index.bin' 'ClusterSize.bin'
% where
%   numObjects: elements of the initial dataset
%   numAttributes: number of attributes of the elements
%   numClusters: number of Clusters
%

fid = fopen('F.bin');
F = fread(fid, [nF mF], 'float')';
fclose(fid);

fid = fopen('T.bin');
T = fread(fid, [nT mT], 'float')';
fclose(fid);

fid = fopen('pF.bin');
pF = fread(fid, [(nT+nT+nF-2) (mT+mT+mF-2)  ], 'float');
pF=pF';
fclose(fid);

fid = fopen('Y.bin');
Y = fread(fid, [nF+nT-1 mF+mT-1 ], 'float');
Y=Y';
fclose(fid);


