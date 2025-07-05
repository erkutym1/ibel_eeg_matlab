% 📁 Dosya ve parametre ayarları
featureRoot = 'process4_features_psd_baseline_normalized';
modelRoot   = 'process5_models';
labelFile   = '../movie_emotions.csv';

if ~exist(modelRoot, 'dir'); mkdir(modelRoot); end

labels = readtable(labelFile, "VariableNamingRule","preserve");
metrics = {'Valence', 'Arousal', 'Dominance'};
sourceCols = {'Valence Mean', 'Arousal Mean', 'Dominance Mean'};

chans = {'AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4'};
bands = {'theta','alpha','beta'};
headers = {};
for b = 1:length(bands)
    for c = 1:length(chans)
        headers{end+1} = sprintf('%s_%s', bands{b}, chans{c});
    end
end

threshold = 3.0;
reportLines = ["Subject            Metric        TrainAcc   TrainF1    TestAcc   TestF1"];
X_all = {}; Y_all = {};

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

        vec_log = log10(vec + eps);  % sadece valence için kullanılacak
        targetVec = zeros(1,3);
        for k = 1:3
            score = labels.(sourceCols{k})(vidIdx);
            targetVec(k) = double(score > threshold);
        end
        X = [X; vec];
        X_log{size(X,1)} = vec_log;  % valence için log-dönüşümlü hali
        Y = [Y; targetVec];
    end

    if size(X,1) < 5
        for k = 1:3
            reportLines(end+1) = sprintf('%-18s %-13s NA         NA         NA       NA', subjectID, metrics{k});
        end
        continue;
    end

    model = struct();
    for k = 1:3
        metric = metrics{k};
        y = Y(:,k);
        if numel(unique(y)) < 2
            reportLines(end+1) = sprintf('%-18s %-13s NA         NA         NA       NA', subjectID, metric);
            continue;
        end

        % Özellik seçim: Valence için log-dönüşümlü
        if strcmp(metric, 'Valence')
            X_metric = vertcat(X_log{:});
        else
            X_metric = X;
        end

        X_metric = fillmissing(X_metric, 'linear', 2, 'EndValues','nearest');
        c = cvpartition(y, 'KFold', 5);
        accTrain = []; accTest = []; f1Train = []; f1Test = [];

        for fold = 1:5
            trIdx = training(c, fold); teIdx = test(c, fold);
            Xtr = X_metric(trIdx,:); ytr = y(trIdx);
            Xte = X_metric(teIdx,:); yte = y(teIdx);
            if numel(unique(ytr)) < 2 || numel(unique(yte)) < 2, continue; end
            M = fitcsvm(Xtr, ytr, 'KernelFunction','linear', 'Standardize',true, ...
                        'BoxConstraint',0.1, 'ClassNames',[0 1], 'Prior','uniform');
            ytr_pred = predict(M, Xtr);
            yte_pred = predict(M, Xte);
            accTrain(end+1) = mean(ytr_pred == ytr);
            accTest(end+1)  = mean(yte_pred == yte);
            f1Train(end+1)  = f1score(ytr, ytr_pred);
            f1Test(end+1)   = f1score(yte, yte_pred);
        end

        model.(sprintf('model_%s', lower(metric))) = M;
        reportLines(end+1) = sprintf('%-18s %-13s %.4f     %.4f     %.4f   %.4f', ...
            subjectID, metric, mean(accTrain), mean(f1Train), mean(accTest), mean(f1Test));
    end

    model.featureNames = headers;
    save(fullfile(modelRoot, sprintf('%s_model.mat', subjectID)), '-struct', 'model');

    % Genel modelde log/düzgün ayrımı yapılacağı için X'i kaydediyoruz
    X_all{end+1} = X;
    Y_all{end+1} = Y;
end

% 🌍 Genel model (Valence log, diğerleri düz)
X = vertcat(X_all{:});
Y = vertcat(Y_all{:});
model = struct();

for k = 1:3
    metric = metrics{k};
    y = Y(:,k);
    if numel(unique(y)) < 2
        reportLines(end+1) = sprintf('%-18s %-13s NA         NA         NA       NA', 'Overall', metric);
        continue;
    end

    if strcmp(metric, 'Valence')
        X_log = log10(X + eps);
        X_filled = fillmissing(X_log, 'linear', 2, 'EndValues','nearest');
    else
        X_filled = fillmissing(X, 'linear', 2, 'EndValues','nearest');
    end

    c = cvpartition(y, 'KFold', 5);
    accTrain = []; accTest = []; f1Train = []; f1Test = [];

    for fold = 1:5
        trIdx = training(c, fold); teIdx = test(c, fold);
        Xtr = X_filled(trIdx,:); ytr = y(trIdx);
        Xte = X_filled(teIdx,:); yte = y(teIdx);
        if numel(unique(ytr)) < 2 || numel(unique(yte)) < 2, continue; end
        M = fitcsvm(Xtr, ytr, 'KernelFunction','linear', 'Standardize',true, ...
                    'BoxConstraint',0.1, 'ClassNames',[0 1], 'Prior','uniform');
        ytr_pred = predict(M, Xtr);
        yte_pred = predict(M, Xte);
        accTrain(end+1) = mean(ytr_pred == ytr);
        accTest(end+1)  = mean(yte_pred == yte);
        f1Train(end+1)  = f1score(ytr, ytr_pred);
        f1Test(end+1)   = f1score(yte, yte_pred);
    end

    model.(sprintf('model_%s', lower(metric))) = M;
    reportLines(end+1) = sprintf('%-18s %-13s %.4f     %.4f     %.4f   %.4f', ...
        'Overall', metric, mean(accTrain), mean(f1Train), mean(accTest), mean(f1Test));
end

model.featureNames = headers;
save(fullfile(modelRoot, 'overall_model.mat'), '-struct', 'model');
writelines(reportLines, fullfile(modelRoot, 'results.txt'));

% 🎯 F1 hesaplayıcı
function f1 = f1score(ytrue, ypred)
    tp = sum((ytrue==1)&(ypred==1));
    fp = sum((ytrue==0)&(ypred==1));
    fn = sum((ytrue==1)&(ypred==0));
    prec = tp / (tp + fp + eps);
    rec  = tp / (tp + fn + eps);
    f1   = 2 * prec * rec / (prec + rec + eps);
end
