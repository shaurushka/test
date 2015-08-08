function [ resultTrain, extraInfo ] = train(parameters, data, labels)
if size(parameters, 2) > 1
    nBlocks = parameters(end, end);
else
    nBlocks = 1;
end;
blockSize = size(parameters,2)/nBlocks; % number of different K for each binarization

Kmin = parameters(1, 1);
Kmax = parameters(1, blockSize);

doBinarization = parameters(2, 1);
binarizationBounds = parameters(3, 1:blockSize:end);
S = parameters(4, 1);
G = parameters(5, 1);

if doBinarization == 0
    n = mean(sum(data,2)); % average length of codogram
    data = data./repmat(sum(data, 2), 1, size(data, 2)); %
end;

healthClassData = data(labels == 0, :);
sickClassData = data(labels == 1, :);

if S ==2
    Kmax = Kmax * 2;
end;

resultTrain = zeros(2, Kmax, nBlocks);
if doBinarization == 0
    [resultTrain(1, :, 1), resultTrain(2,:, 1)] = chooseEtalons(healthClassData ,...
        sickClassData, Kmax, S, G, doBinarization,  n);
else
    for idx = 1:nBlocks
        [resultTrain(1, :, idx), resultTrain(2,:, idx)] = chooseEtalons(healthClassData ,...
            sickClassData, Kmax, S, G, doBinarization, binarizationBounds(idx));
    end;
end;

extraInfo = cell(1, 1);
end

function [ etalons, values ] = chooseEtalons( healthClassData,...
    sickClassData, K, S, G, doBinarization, param)
% Returns ranked lists of etalons and their weights
% param = theta, if doBinarization = 1

healthClassFrequences = findFrequences(healthClassData, doBinarization, param);
sickClassFrequences = findFrequences(sickClassData, doBinarization, param);
healthClassRegFrequences = findRegFrequences(healthClassData, doBinarization, param);
sickClassRegFrequences = findRegFrequences(sickClassData, doBinarization, param);

M = 300; % number of label permutations (if S = 8 or S = 9 or G = 6)

switch S
    case 1
        [~, ind1] = sort(sickClassFrequences, 'DESCEND');
        etalons = ind1(1:K);
    case 2
        [~, ind1] = sort(sickClassFrequences, 'DESCEND');
        [~, ind0] = sort(healthClassFrequences, 'DESCEND');
        etalons = [ind1(1:K/2) ind0(1:K/2)];
    case 3
        [~, ind1] = sort(sickClassFrequences - healthClassFrequences, 'DESCEND');
        etalons = ind1(1:K);
    case 4
        [~, ind1] = sort(log(sickClassRegFrequences./healthClassRegFrequences), 'DESCEND');
        etalons = ind1(1:K);
    case 5
        [~, ind1] = sort(abs(log(sickClassRegFrequences./healthClassRegFrequences)), 'DESCEND');
        etalons = ind1(1:K);
    case 6
        [~, ind1] = sort(log((sickClassRegFrequences.*(1-healthClassRegFrequences))./...
            (healthClassRegFrequences.*(1-sickClassRegFrequences))), 'DESCEND');
        etalons = ind1(1:K);
    case 7
        [~, ind1] = sort(abs(log((sickClassRegFrequences.*(1-healthClassRegFrequences))./...
            (healthClassRegFrequences.*(1-sickClassRegFrequences)))), 'DESCEND');
        etalons = ind1(1:K);
    case {8, 9}
        baseValue = sickClassFrequences;
        P = [healthClassData; sickClassData];
        labels = [zeros(size(healthClassData,1),1); ones(size(sickClassData, 1),1)];
        mixValue = zeros(M, size(P,2));
        for idx = 1:M
            permutation = randperm(length(labels));
            indMix = find(labels(permutation)==1);
            mixp = P(indMix,:); % new "sick" class
            mixValue(idx,:) = findFrequences( mixp, doBinarization, param );
        end;
        Dw = (2*baseValue - min(mixValue) - max(mixValue))./(max(mixValue) - min(mixValue));
        Dw(isnan(Dw)) = 0;
        
        if S == 8
            [~, ind1] = sort(Dw, 'DESCEND');
        else
            [~, ind1] = sort(abs(Dw), 'DESCEND');
        end;
        etalons = ind1(1:K);       
    case 10
        etalons = newEtalons( healthClassData, sickClassData, K);
    otherwise
        disp('ERROR: etalonChoice is not valid')
end

nonzeroEtalons = etalons(etalons~=0);
switch G
    case 1
        values = ones(1,K);
    case 2
        values = [sickClassFrequences(nonzeroEtalons) zeros(1, K - length(nonzeroEtalons))];
    case 3
        values = [sickClassFrequences(nonzeroEtalons) - healthClassFrequences(nonzeroEtalons)...
            zeros(1, K - length(nonzeroEtalons))];
    case 4
        values = [log(sickClassRegFrequences(nonzeroEtalons)./healthClassRegFrequences(nonzeroEtalons))...
            zeros(1, K - length(nonzeroEtalons))];
    case 5
        values = [log((sickClassRegFrequences(nonzeroEtalons).*(1-healthClassRegFrequences(nonzeroEtalons)))./...
            (healthClassRegFrequences(nonzeroEtalons).*(1-sickClassRegFrequences(nonzeroEtalons))))...
            zeros(1, K - length(nonzeroEtalons))];
    case 6
        if S ~= 8 && S~=9
            baseValue = sickClassFrequences;
            P = [healthClassData; sickClassData];
            labels = [zeros(size(healthClassData,1),1); ones(size(sickClassData, 1),1)];
            mixValue = zeros(M, size(P,2));
            for idx = 1:M
                permutation = randperm(length(labels));
                indMix = find(labels(permutation)==1);
                mixp = P(indMix,:); %new "sick" class
                mixValue(idx,:) = findFrequences( mixp, doBinarization, param);
            end;
            Dw = (2*baseValue - min(mixValue) - max(mixValue))./(max(mixValue) - min(mixValue));
            Dw(isnan(Dw)) = 0;
        end;
        values = [Dw(nonzeroEtalons) zeros(1, K - length(nonzeroEtalons))];
    otherwise
        disp('ERROR: etalonWeight is not valid')
end

end

function [ frequences ] = findFrequences( p, doBinarization, param )
l = size(p,1);
if doBinarization == 1
    % calculate B_w
    frequences = (sum(p > param, 1))/l;
else
    % calculate F_w
    frequences = mean(p, 1);
end;
end

function [ frequences ] = findRegFrequences( p, doBinarization, param )
l = size(p,1);
if doBinarization == 1
    % calculate regularized B_w
    frequences = (sum(p > param, 1) + 1)/(l + 2);
else
    frequences = (sum(p,1)+2/param)/(size(p,1)+1);
end;
end

function [etalons] = newEtalons(healthClassData, sickClassData, K)
etalons = 1:K; %???
end
