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
else
    n = -1; % fake
end;

healthClassData = data(labels == 0, :);
sickClassData = data(labels == 1, :);

if S ==2
    Kmax = Kmax * 2;
end;

resultTrain = zeros(2, Kmax, nBlocks);
for idx = 1:nBlocks
    [resultTrain(1, :, idx), resultTrain(2,:, idx)] = chooseEtalons(healthClassData ,...
        sickClassData, Kmax, S, G, doBinarization, binarizationBounds(idx), n);
end;

extraInfo = cell(1, 1);
end

function [ etalons, values ] = chooseEtalons( healthClassData,...
    sickClassData, K, S, G, doBinarization, theta, n)
% Returns ranked lists of etalons and their weights

if doBinarization == 0
    healthClassFrequences = findFrequences(healthClassData);
    sickClassFrequences = findFrequences(sickClassData);
    healthClassRegFrequences = findRegFrequences(healthClassData, doBinarization, n);
    sickClassRegFrequences = findRegFrequences(sickClassData, doBinarization, n);
else
    healthClassFrequences = findFrequences(healthClassData, theta);
    sickClassFrequences = findFrequences(sickClassData, theta);
    healthClassRegFrequences = findRegFrequences(healthClassData, doBinarization, theta);
    sickClassRegFrequences = findRegFrequences(sickClassData, doBinarization, theta);
end;

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
            mixValue(idx,:) = findFrequences( mixp, theta );
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
        %greedyEtalons
        etalons = 1:K;
    case 11
        %greedyInternalEtalons
        etalons = 1:K;
    otherwise
        disp('ERROR: etalonChoice is not valid')
end
if S < 10
  values = getWeights(etalons, S, G, K, M, ...
                      sickClassFrequences, healthClassFrequences, ...
                      sickClassRegFrequences, healthClassRegFrequences, ...
                      healthClassData, sickClassData, theta);
else
  [etalons, values] = greedyEtalons(S, G, K, M, ...
                                    sickClassFrequences, ...
                                    healthClassFrequences, ...
                                    sickClassRegFrequences, ...
                                    healthClassRegFrequences, ...
                                    healthClassData, sickClassData, theta);
end                  
end

function [values] = getWeights(etalons, S, G, K, M, ...
                               sickClassFrequences, healthClassFrequences, ...
                               sickClassRegFrequences, healthClassRegFrequences, ...
                               healthClassData, sickClassData, theta)
nonzeroEtalons = etalons(etalons~=0);
switch G
    case 1
        values = [ones(1, length(nonzeroEtalons)) zeros(1, K - length(nonzeroEtalons))];
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
                mixValue(idx,:) = findFrequences( mixp, theta );
            end;
            Dw = (2*baseValue - min(mixValue) - max(mixValue))./(max(mixValue) - min(mixValue));
            Dw(isnan(Dw)) = 0;
        end;
        values = [Dw(nonzeroEtalons) zeros(1, K - length(nonzeroEtalons))];
    otherwise
        disp('ERROR: etalonWeight is not valid')
end
end

function [ frequences ] = findFrequences( p, theta )
l = size(p,1);
if nargin > 1
    % calculate B_w
    frequences = (sum(p > theta, 1))/l;
else
    % calculate F_w
    frequences = mean(p, 1);
end;
end

function [ frequences ] = findRegFrequences( p, doBinarization, parameter )
l = size(p,1);
if doBinarization == 1
    % calculate regularized B_w
    frequences = (sum(p > parameter, 1) + 1)/(l + 2);
else
    frequences = (sum(p,1)+2/parameter)/(size(p,1)+1);
end;
end


function [etalons] = naiveEtalons(healthClassData, sickClassData, K)
%etalons = 1:K; %???
rng(12345);  
eps = 0.05;

[data, classLabels] = getSubsetOfData(healthClassData, sickClassData);
[L, featuresCount] = size(data);
trainSize = floor(0.7 * L);

featuresQeps = zeros(1, featuresCount);
for featureIndex = 1:featuresCount
  chain = makeChainWithGaps(data(:, featureIndex), classLabels);
  
  featuresQeps(featureIndex) = getQepsMC(chain, trainSize, eps);
end
[~, ind] = sort(featuresQeps);
etalons = ind(1:K);

end

function [etalons, weights] = greedyEtalons(S, G, K, M, ...
                                            sickClassFrequences, ...
                                            healthClassFrequences, ...
                                            sickClassRegFrequences, ...
                                            healthClassRegFrequences, ...
                                            healthClassData, sickClassData, ...
                                            theta)
rng(12345);  

[data, classLabels] = getSubsetOfData(healthClassData, sickClassData);
[L, featuresCount] = size(data);
trainSize = floor(0.7 * L);

weights = getWeights(1:featuresCount, S, G, K, M, ...
                     sickClassFrequences, healthClassFrequences, ...
                     sickClassRegFrequences, healthClassRegFrequences, ...
                     healthClassData, sickClassData, theta);
etalons = zeros(1, K);
classifier = zeros(L, 1);
notEtalons = 1:featuresCount;
topFeatures = 1:featuresCount;
topFeaturesSize = 10;
maxBlockSize = 5;
it = 1;
numMCIt = 2000;
trainIndices = generateTrainIndices(numMCIt, L, trainSize);
while it <= length(etalons) 
  %disp(it);
  if mod(it, 20) == 0
    featuresSet = notEtalons;
  else
    featuresSet = topFeatures;
  end
  %выбираем фичу
  featuresOverfit = zeros(1, length(featuresSet));
  for index = 1:length(featuresSet)
    %disp(index);
    featureIndex = featuresSet(index);
    chain = makeChainWithGaps(classifier + weights(featureIndex) * data(:, featureIndex), ...
                              classLabels);
    if S == 10
      featuresOverfit(index) = getOverfitValueMC(chain, trainIndices);
    else
      featuresOverfit(index) = min(sum(chain));
    end
  end
  [~, ind] = sort(featuresOverfit);
  if it > 10
    blockSize = min([length(etalons) - it + 1 length(ind) maxBlockSize]);
  else
    blockSize = 1;
  end
  currentEtalons = featuresSet(ind(1:blockSize));
  classifier = classifier + data(:, currentEtalons) * weights(currentEtalons)';
  %обновляем множество топовых фичей
  if mod(it, 20) == 0 && blockSize + 1 <= length(featuresSet)
    topFeatures = featuresSet(ind(blockSize + 1 : ...
                                  min(blockSize + topFeaturesSize + 1, ...
                              length(featuresSet))));
  end 
  %убираем из рассмотрения добавленную фичу
  if it + blockSize - 1 <= length(etalons)
    etalons(it : it + blockSize - 1) = currentEtalons;
    notEtalons = setxor(notEtalons, currentEtalons);
    topFeatures = setxor(topFeatures, currentEtalons);
  end
  it = it + blockSize;
end
weights = weights(etalons);
end



