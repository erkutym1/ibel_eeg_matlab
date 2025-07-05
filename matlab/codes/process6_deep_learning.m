% 📁 Dosya yolları
featureRoot = 'process4_features_psd_baseline_normalized';
modelRoot   = 'process6_models_dl_stratified';
labelFile   = '../movie_emotions.csv';
if ~exist(modelRoot, 'dir'); mkdir(modelRoot); end

labels = readtable(labelFile, "VariableNamingRule","preserve");
metrics = {'Valence', 'Arousal', 'Dominance'};
sourceCols = {'Valence Mean', 'Arousal Mean', 'Dominance Mean'};
threshold = 3.0;

% Kanal başlıkları
chans = {'AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4'};
bands = {'theta','alpha','beta'};
headers = {};
for b = 1:length(bands)
    for c = 1:length(chans)
        headers{end+1} = sprintf('%s_%s', bands{b}, chans{c});
    end
end

reportLines = ["Subject            Metric        TrainAcc   TrainF1    TestAcc   TestF1"];
X_all = {}; Y_all = {};

% Tüm subject verisi toplanır
subjects = dir(fullfile(featureRoot, 'subject_*'));
for subj = subjects'
    subjectID = subj.name;
    subjPath = fullfile(featureRoot, subjectID);
    videoDirs = dir(fullfile(subjPath, 'video_*'));

    X = []; Y = [];
    for vid = videoDirs'
        vidIdx = str2double(erase(vid.name, 'video_'));
        if isnan(vidIdx) || vidIdx > height(labels), continue; end
        file = fullfile(subjPath, vid.name, 'stimuli_psd_normalized.csv');
        if ~isfile(file), continue; end
        vec = table2array(readtable(file));
        if size(vec,2) ~= 42, continue; end
        vec = log10(max(vec, eps));  % Kompleks engelleme
        label = zeros(1,3);
        for k = 1:3
            label(k) = double(labels.(sourceCols{k})(vidIdx) > threshold);
        end
        X = [X; vec]; Y = [Y; label];
    end
    if isempty(X), continue; end
    X_all{end+1} = X;
    Y_all{end+1} = Y;
end

% 📦 Genel model oluşturma
X = vertcat(X_all{:});
Y = vertcat(Y_all{:});
model = struct();

for k = 1:3
    y = Y(:,k);
    if numel(unique(y)) < 2
        reportLines(end+1) = sprintf('%-18s %-13s NA         NA         NA       NA', 'Overall', metrics{k});
        continue;
    end

    % Stratified 5-Fold partition
    c = cvpartition(y, 'KFold', 5, 'Stratify', true);
    accTrain = []; accTest = []; f1Train = []; f1Test = [];

    for fold = 1:5
        trIdx = training(c, fold);
        teIdx = test(c, fold);

        Xtr = fillmissing(X(trIdx,:), 'linear', 2, 'EndValues','nearest');
        Xte = fillmissing(X(teIdx,:), 'linear', 2, 'EndValues','nearest');
        ytr = y(trIdx); yte = y(teIdx);

        if numel(unique(ytr)) < 2 || numel(unique(yte)) < 2, continue; end

        layers = [
            featureInputLayer(42)
            fullyConnectedLayer(64)
            reluLayer
            dropoutLayer(0.4)
            fullyConnectedLayer(32)
            reluLayer
            fullyConnectedLayer(2)
            softmaxLayer
            classificationLayer];

        options = trainingOptions('adam', ...
            'MaxEpochs', 50, ...
            'MiniBatchSize', 8, ...
            'Shuffle','every-epoch', ...
            'Verbose', false);

        net = trainNetwork(Xtr, categorical(ytr), layers, options);
        ytr_pred = double(classify(net, Xtr)) - 1;
        yte_pred = double(classify(net, Xte)) - 1;

        accTrain(end+1) = mean(ytr_pred == ytr);
        accTest(end+1)  = mean(yte_pred == yte);
        f1Train(end+1)  = f1score(ytr, ytr_pred);
        f1Test(end+1)   = f1score(yte, yte_pred);
    end

    % 📊 Ortalama skorları yaz
    model.(sprintf('model_%s', lower(metrics{k}))) = net;
    reportLines(end+1) = sprintf('%-18s %-13s %.4f     %.4f     %.4f   %.4f', ...
        'Overall', metrics{k}, mean(accTrain), mean(f1Train), mean(accTest), mean(f1Test));
end

model.featureNames = headers;
save(fullfile(modelRoot, 'overall_dl_model.mat'), '-struct', 'model');
writelines(reportLines, fullfile(modelRoot, 'results.txt'));

% 🎯 F1 hesaplama
function f1 = f1score(ytrue, ypred)
    tp = sum((ytrue==1) & (ypred==1));
    fp = sum((ytrue==0) & (ypred==1));
    fn = sum((ytrue==1) & (ypred==0));
    prec = tp / (tp + fp + eps);
    rec  = tp / (tp + fn + eps);
    f1 = 2 * prec * rec / (prec + rec + eps);
end
