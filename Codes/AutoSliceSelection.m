%% Automated Slice Selection based on centroid extraction from Chloe.
% Use the proportion to calculate corresponding coordinates in MIMICS.
% Find the closet number/slice in MIMICS(target slices).
n = 0:0.625:842.5;
origin = [516.0454056
491.0857329
467.6119061
442.8455301
417.3625898
392.4702781
365.7826506
337.7761153
307.5905997
276.2531141
244.3796698
205.677387
166.8930997
130.8676825
93.64050695
674.5896524
661.3878824
638.5905939
619.0041773
601.3067961
585.8303737
570.1635834
552.5989716
533.6812833
];
%% 

for i = 1:length(origin)
    [~,index(i)] = min(abs(n-origin(i)));
end

n_final = n(index);
n_final = n_final';
SlideNumber = index'-1;
