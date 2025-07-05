% DREAMER dataset ve çıktı klasörü
dataPath = '../DREAMER.mat';
outputRoot = 'process1_filtered';

% EEGLAB başlat
[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;

% Veriyi yükle
load(dataPath);  % DREAMER struct'ı içinde
subjects = DREAMER.Data;
fs = DREAMER.EEG_SamplingRate;
noSubjects = length(subjects);
noVideos = 18;

% Filtre parametreleri
lowCutoff = 4;
highCutoff = 30;

% Kanal sırası (sabit 14 kanal, DREAMER)
channelLabels = {'AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4'};

for subjIdx = 1:noSubjects
    subjectID = sprintf('subject_%02d', subjIdx);
    subj = subjects{subjIdx};

    for vidIdx = 1:noVideos
        movieID = sprintf('video_%02d', vidIdx);
        saveDir = fullfile(outputRoot, subjectID, movieID);
        if ~exist(saveDir, 'dir')
            mkdir(saveDir);
        end

        for cond = ["baseline", "stimuli"]
            rawData = subj.EEG.(cond){vidIdx};  % (zaman x kanal)

            if size(rawData,2) ~= 14
                warning('⚠️ %s / %s: Kanal sayısı beklenenden farklı (%d)', subjectID, movieID, size(rawData,2));
                continue;
            end

            EEG = pop_importdata('data', rawData', ...
                                 'srate', fs, ...
                                 'dataformat', 'array');

            % Kanal isimlerini veriye ekle (14 sabit kanal)
            for k = 1:14
                EEG.chanlocs(k).labels = channelLabels{k};
            end

            % High–Low bandpass filtre (4–30 Hz)
            EEG = pop_eegfiltnew(EEG, lowCutoff, highCutoff);

            % Çıktıyı kaydet
            filteredData = EEG.data';  % zaman x kanal
            filename = fullfile(saveDir, cond + ".csv");
            writematrix(filteredData, filename);
        end
    end
end
