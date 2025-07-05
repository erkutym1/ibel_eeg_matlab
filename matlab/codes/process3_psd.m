% Giriş ve çıkış klasörleri
inputRoot = 'process2_cleaned';
outputRoot = 'process3_features_psd';
fs = 128;  % DREAMER EEG örnekleme frekansı
window = 2 * fs;  % 2 saniye pencere
noverlap = window / 2;

% Frekans bantları
bands = {
    'theta', [4 8];
    'alpha', [8 13];
    'beta',  [13 20];
};

% Sabit kanal listesi
channelLabels = {'AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4'};
numChannels = numel(channelLabels);

% Kanal + bant başlıkları
headers = {};
for b = 1:size(bands,1)
    for ch = 1:numChannels
        headers{end+1} = sprintf('%s_%s', bands{b,1}, channelLabels{ch}); %#ok<SAGROW>
    end
end

% Katılımcı klasörleri
subjects = dir(fullfile(inputRoot, 'subject_*'));
for subj = subjects'
    subjectID = subj.name;
    subjectPath = fullfile(inputRoot, subjectID);
    videoDirs = dir(fullfile(subjectPath, 'video_*'));

    for vid = videoDirs'
        movieID = vid.name;
        moviePath = fullfile(subjectPath, movieID);
        savePath = fullfile(outputRoot, subjectID, movieID);
        if ~exist(savePath, 'dir'); mkdir(savePath); end

        for cond = ["baseline", "stimuli"]
            inFile = fullfile(moviePath, cond + "_cleaned.csv");
            if ~isfile(inFile)
                fprintf('⛔ Dosya yok: %s\n', inFile);
                continue;
            end

            try
                T = readtable(inFile);
                data = T{:,1:numChannels};  % ICA flag dışındaki ilk 14 kanal
            catch
                warning('❌ Okuma hatası: %s', inFile);
                continue;
            end

            data = data';  % kanal x zaman
            [~, F] = pwelch(zeros(256,1), window, noverlap, [], fs);  % frekans vektörü çıkar (tek sefer)

            PSD_features = nan(1, numChannels * size(bands,1));  % 1x42

            for ch = 1:numChannels
                x = data(ch,:);

                % Tüm veri NaN ise o kanalı atla
                if all(isnan(x))
                    continue;
                end

                try
                    [Pxx, ~] = pwelch(x, window, noverlap, [], fs);

                    for b = 1:size(bands,1)
                        fRange = bands{b,2};
                        idx = F >= fRange(1) & F <= fRange(2);
                        bandPower = mean(Pxx(idx), 'omitnan');
                        PSD_features((b-1)*numChannels + ch) = bandPower;
                    end
                catch
                    warning('⚠️ pwelch hatası: %s / %s / %s → %s', subjectID, movieID, cond, channelLabels{ch});
                end
            end

            % Tablo olarak kaydet
            outTable = array2table(PSD_features, 'VariableNames', headers);
            outFile = fullfile(savePath, cond + "_psd_features.csv");
            writetable(outTable, outFile);
        end
    end
end
