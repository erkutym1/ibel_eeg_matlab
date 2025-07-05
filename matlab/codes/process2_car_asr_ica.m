% Giriş ve çıkış klasörleri
inputRoot = 'process1_filtered';
outputRoot = 'process2_cleaned';
fs = 128;

% EEGLAB başlat
[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;

% Sabit DREAMER kanal etiketleri
channelLabels = {'AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4'};
numChannels = numel(channelLabels);

% Log dosyası
logFile = fullfile(outputRoot, 'clean_trials.txt');
fid = fopen(logFile, 'w');

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
        if ~exist(savePath, 'dir')
            mkdir(savePath);
        end

        for cond = ["baseline", "stimuli"]
            csvFile = fullfile(moviePath, cond + ".csv");
            if ~isfile(csvFile)
                fprintf('⛔ Dosya yok: %s\n', csvFile);
                continue;
            end

            rawData = readmatrix(csvFile);  % zaman x kanal

            if size(rawData,2) ~= numChannels
                warning('⚠️ %s / %s / %s: Beklenen 14 kanal yerine %d kanal var, atlandı.\n', ...
                    subjectID, movieID, cond, size(rawData,2));
                continue;
            end

            % EEGLAB import
            EEG = pop_importdata('data', rawData', 'srate', fs, 'dataformat', 'array');
            for k = 1:numChannels
                EEG.chanlocs(k).labels = channelLabels{k};
            end
            EEG.nbchan = numChannels;
            EEG.trials = 1;
            EEG.pnts = size(rawData, 1);
            EEG.xmin = 0;
            EEG.xmax = (EEG.pnts - 1) / fs;
            EEG.setname = cond;

            % 1️⃣ CAR
            EEG = pop_reref(EEG, []);

            % 2️⃣ ASR (kanal silme kapalı)
            EEG = clean_artifacts(EEG, ...
                'FlatlineCriterion', 5, ...
                'BurstCriterion', 5, ...
                'WindowCriterion', 'off', ...
                'ChannelCriterion', 'off');

            % 3️⃣ ICA
            fprintf('🔁 CPU ile ICA: %s / %s / %s\n', subjectID, movieID, cond);
            EEG = pop_runica(EEG, 'extended', 1, 'interrupt', 'off');

            % 4️⃣ Kanal kaybı kontrol ve kayıt
            cleanedData = EEG.data';  % zaman x kanal
            actualChannels = size(cleanedData, 2);
            isFlagged = actualChannels < numChannels;

            paddedData = nan(size(cleanedData,1), numChannels);
            paddedData(:,1:actualChannels) = cleanedData;

            flagColumn = double(isFlagged) * ones(size(paddedData,1), 1);
            finalData = [paddedData, flagColumn];

            % Başlıklar
            varNames = [channelLabels, {'ICA_ChannelLossFlag'}];
            T = array2table(finalData, 'VariableNames', varNames);

            if isFlagged
                warning('⚠️ %s / %s / %s: ICA sonrası %d kanal kaldı → FLAG = 1', ...
                    subjectID, movieID, cond, actualChannels);
            else
                logFile = fullfile(outputRoot, 'clean_trials.txt');
                [~, logRootDir] = fileparts(outputRoot);
                if ~exist(outputRoot, 'dir')
                    mkdir(outputRoot);
                end
                
                fid = fopen(logFile, 'w');
                if fid == -1
                    error('❌ Log dosyası açılamadı: %s', logFile);
                end
            end

            % CSV yaz
            outFile = fullfile(savePath, cond + "_cleaned.csv");
            writetable(T, outFile);
        end
    end
end

% Log dosyasını kapat
fclose(fid);
