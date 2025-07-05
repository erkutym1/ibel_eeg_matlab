% Giriş ve çıkış klasörleri
inputRoot = 'process3_features_psd';
outputRoot = 'process4_features_psd_baseline_normalized';

% Kanal ve bant etiketleri (aynı sırayla)
channelLabels = {'AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4'};
bands = {'theta','alpha','beta'};

expectedHeaders = {};
for b = 1:length(bands)
    for ch = 1:length(channelLabels)
        expectedHeaders{end+1} = sprintf('%s_%s', bands{b}, channelLabels{ch}); %#ok<SAGROW>
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

        try
            % Dosyaları oku
            baseFile = fullfile(moviePath, 'baseline_psd_features.csv');
            stimFile = fullfile(moviePath, 'stimuli_psd_features.csv');

            if ~isfile(baseFile) || ~isfile(stimFile)
                warning('⛔ Eksik PSD dosyası: %s', movieID);
                continue;
            end

            baseT = readtable(baseFile);
            stimT = readtable(stimFile);

            % Başlık ve yapı kontrolü
            if ~isequal(baseT.Properties.VariableNames, stimT.Properties.VariableNames)
                warning('⚠️ Başlıklar uyuşmuyor: %s / %s', subjectID, movieID);
                continue;
            end

            % Beklenen başlık kontrolü (42 özellik)
            if ~isequal(baseT.Properties.VariableNames, expectedHeaders)
                warning('⚠️ Başlık yapısı farklı (değişmiş olabilir): %s / %s', subjectID, movieID);
                continue;
            end

            % Normalize et: stimul / baseline
            baseVec = table2array(baseT);
            stimVec = table2array(stimT);

            normVec = stimVec ./ baseVec;  % NaN uyumlu (NaN/NaN → NaN)

            % Kaydet
            outTable = array2table(normVec, 'VariableNames', expectedHeaders);
            outFile = fullfile(savePath, 'stimuli_psd_normalized.csv');
            writetable(outTable, outFile);

        catch ME
            warning('❌ Normalizasyon hatası: %s / %s → %s', subjectID, movieID, ME.message);
        end
    end
end
