function [shat, lambdaS_, vVoiced, vPitch,MIN_GAIN] = ProcessDereverbSpectralSubtract(noisy, stConfig)
%
% this file performs single channel noise reduction based on
% [1] Timo Gerkmann, Richard C. Hendriks, "Unbiased MMSE-based Noise
%   Power Estimation with Low Complexity and Low Tracking Delay",
%   IEEE Trans. Audio, Speech and Language Processing, 2012

% [2] Colin Breithaupt, Timo Gerkmann, Rainer Martin, "A Novel A Priori
%   SNR Estimation Approach Based on Selective Cepstro-Temporal Smoothing",
%   IEEE Int. Conf. Acoustics, Speech, Signal Processing, Las Vegas, NV, USA, Apr. 2008.
% [3] Timo Gerkmann, Rainer Martin, "On the Statistics of Spectral
%   Amplitudes After Variance Reduction by Temporal Cepstrum Smoothing and Cepstral
%   Nulling", IEEE Trans. Signal Processing, Vol. 57, No. 11, pp. 4165-4174, Nov. 2009
%
% Author Timo Gerkmann, Universitaet Oldenburg, Germany 2012
% All rights reserved.
%
% modified B.Cauchi, Fraunhofer IDMT, 2014
% modified I.Kodrasi, Uni Oldenburg, 2015

% Single channel scheme described in:
%
% B. Cauchi, I. Kodrasi, R. Rehr, S. Gerlach, A. Jukić, T. Gerkmann,
% S. Doclo, and S. Goetze. Combination of MVDR beamforming and
% single-channel spectral processing for enhancing noisy and reverberant
% speech. EURASIP Journal on Advances in Signal Processing,
% 2015:1–12, July 2015.
%
%
%
%



%% configuration
% set sampling frequency
dFs = stConfig.dFs;

% minimum gain
MIN_GAIN = stConfig.dMinimumGain;

% lower limit on a priori SNR
SNR_LOW_LIM    = db2pow(-Inf);


% smoothing constant for decision directed approach
alphaDD = stConfig.dAlphaDD;

% estimate reverb time if necessary
if isnan(stConfig.dT60)
        stConfig.dT60 = T60EstimationEaton(noisy, dFs);
end

% apply estimation factor
stConfig.dT60 = stConfig.dT60 .* stConfig.dT60EstFac;

% parameters for reverb suppression
% see Lebart paper for relation between rho and T60
rho = 6.908 ./ stConfig.dT60;


%% constants for block processing and pre allocations
% frame size
frLen   = stConfig.iFrameLength;

% frame shift
frShift  = round(stConfig.iFrameLength / stConfig.iOverlapFactor);

% compute number of frames
nFrames = floor((length(noisy) - frLen) / frShift) + 1;

% create analysis and synthesis window
anWin  = sqrt(hanning(frLen, 'periodic'));
synWin = sqrt(hanning(frLen, 'periodic'));

% pre allocate output signal
shat = zeros(size(noisy));

% pre allocate result vector for first speech estimation
lambdaS_ = zeros(frLen/2+1,nFrames);

%% initialize SPP based noise estimator
% This function computes the initial noise PSD estimate. It is assumed
% that the first 5 time-frames are noise-only.
noisePow = init_noise_tracker_ideal_vad(noisy,frLen,frLen,frShift, anWin);

% constants for a posteriori SPP
% a priori probability of speech presence
q               = stConfig.stSPP.APrioriSPP;
priorFact       = q./(1-q);

% optimal fixed a priori SNR for SPP estimation
xiOptDb         = stConfig.stSPP.dOptimalAPrioriSNR;
xiOpt           = 10.^(xiOptDb./10);
logGLRFact      = log(1./(1+xiOpt));
GLRexp          = xiOpt./(1+xiOpt);

% initialization for mean speech presence probability
PH1mean         = 0.5;

% smoothing constant for computing mean speech presence probability
alphaPH1mean    = stConfig.stSPP.dAlphaMeanPH1;
alphaPSD        = stConfig.stSPP.dAlphaPSD;

% compute number of frames to be skipped due to a higher overlap factor
iOverlapSkip = stConfig.iOverlapFactor / 2;

% pre allocate vector for voiced frames and pitch
vVoiced = false(nFrames, 2);
vPitch = zeros(nFrames, 2);

%% generate input spectra
% generate matrix containing block indeces
matIdx = bsxfun(@plus, (1:frLen).', (0:nFrames - 1) .* frShift);

% split signal into frames, apply window function and perform fourier
% transformation
matSpectra = fft(bsxfun(@times, noisy(matIdx), anWin));

% throw away half of the spectrum
matSpectra = matSpectra(1:floor(frLen / 2 + 1), :);

% compute sub sampled spectra
matSpectraSub = matSpectra(:, 1:iOverlapSkip:end);
matSubIdx = matIdx(:, 1:iOverlapSkip:end);

% pre allocate noise power estimation
matNoisePSD = zeros(size(matSpectraSub));

if stConfig.bPEFAC
    % initialize PEFAC
    stPEFAC = InitModGonzalez(stConfig.dFs);
    
    % get vda decision
    [~, vTime, vVDA] = ProcessModGonzalez(stPEFAC, noisy);
    
    vVDA = InterpVUDecision(vTime, vVDA, ...
        (1:nFrames) .* frShift ./ stConfig.dFs);
else
    vVDA = NaN(nFrames, 1);
end

for indFr = 1:size(matNoisePSD, 2)
    % get noisy DFT frame
    noisyDftFrame = matSpectraSub(:, indFr);
    
    % compute magnitude squared spectrum
    noisyPer = noisyDftFrame .* conj(noisyDftFrame);
    
    switch stConfig.szNoiseEstimator
        case 'spp'
            if indFr > 1
                % a posteriori SNR based on old noise power estimate
                snrPost1 =  noisyPer./(noisePow);
                
                %% noise power estimation
                % compute generalized likelihood ratio
                GLR     = priorFact .* exp(min(logGLRFact + GLRexp.*snrPost1,200));
                
                % a posteriori speech presence probability
                PH1     = GLR ./ (1 + GLR);
                
                % smooth speech presence probability
                PH1mean  = alphaPH1mean * PH1mean + (1-alphaPH1mean) * PH1;
                
                % detect stuck frames and set speech presence probability to upper
                % floor
                stuckInd = PH1mean > 0.99;
                PH1(stuckInd) = min(PH1(stuckInd),0.99);
                
                % estimate noise power spectrum
                estimate =  PH1 .* noisePow + (1-PH1) .* noisyPer;
                
                % estimate noise variance
                noisePow = alphaPSD * noisePow + (1 - alphaPSD) * estimate;
            end
        case 'minimumstat'
            if indFr == 1
                % initialize minimum statistics estimator
                [noisePow, stMinStat] = ...
                    minimumStatistics2001(abs(noisyDftFrame), 10, [], ...
                    stConfig.stMinStat, true);
            else
                % process minimum statistics
                [noisePow, stMinStat] = ...
                    minimumStatistics2001(abs(noisyDftFrame), 10, ...
                    stMinStat, stConfig.stMinStat, false);
            end
    end
    
    % store estimated noise power in vector
    matNoisePSD(:, indFr) = noisePow;
    
    %% SNR estimation
    switch lower(stConfig.APrioriSNREstimationMethod)
        case 'cepstral'
            %
            if indFr == 1
                [snrPrio, cepSNRstate, vVoiced(indFr, 1), vPitch(indFr, 1)] = ...
                    cepSNR(noisyDftFrame, noisePow, dFs, true, [], ...
                    stConfig.iSmoothingConstantSetting, ...
                    vVDA(indFr));
            else
                [snrPrio, cepSNRstate, vVoiced(indFr, 1), vPitch(indFr, 1)] = ...
                    cepSNR(noisyDftFrame, noisePow, dFs, false, ...
                    cepSNRstate, stConfig.iSmoothingConstantSetting, ...
                    vVDA(indFr));
            end
            
            % ml estimator
            %             snrPrio = max(noisyPer ./ noisePow - 1, 0);
        case 'dd'
            % compute a posteriori SNR
            snrPost = noisyPer ./ noisePow;
            
            % compute a priori SNR
            if indFr == 1
                % initialize
                snrPrio = max(0, snrPost - 1);
            else
                snrPrio = alphaDD .* abs(shatDftFrame).^2 ./ noisePowLast  + ...
                    (1 - alphaDD) .* max(0, snrPost - 1);
            end
            
            noisePowLast = noisePow;
        otherwise
            error('unknown estimation method')
    end
    
    % threshold snr
    snrPrio = max(snrPrio, SNR_LOW_LIM);
    
    %% dereverb
    if stConfig.bDereverb
        if 0
            Wiener  = snrPrio./(1+snrPrio);
            snrPost =  noisyPer./(noisePow);%a posteriori SNR
            
            % mmse (?) estimator for |S|^2
            ESY     = Wiener.*(1./snrPost + Wiener) .* noisyPer;
            
            % smoothing constant to compute variance
            etaS    = exp(-frShift/(dFs*70e-3)); % = 0.82
            if indFr == 1
                lambdaS_(:,indFr) = ESY;
            else
                lambdaS_(:,indFr) = etaS .* lambdaS_(:,indFr-1) + (1-etaS) .* ESY;
            end
        else
            lambdaS_(:,indFr) = snrPrio .* noisePow;
        end
        
        % estimate reverb energy
        if indFr - stConfig.iLateFrames > 0
            lambdaSl = ...
                lambdaS_(:, indFr - stConfig.iLateFrames) .* ...
                exp(-2 .* rho .* stConfig.dLateReverb);
        else
            lambdaSl = 0;
        end
        
        % add reverb energy to noise power estimation
        noisePow = noisePow + lambdaSl;
        
        % add noise reverberant noise power to noise estimation
        matNoisePSD(:, indFr) = noisePow;
        
        %% SNR Re-estimation
        switch lower(stConfig.APrioriSNREstimationMethod)
            case 'cepstral'
                if indFr == 1
                    [snrPrio, cepSNRstateDereverb, vVoiced(indFr, 2), vPitch(indFr, 2)] = ...
                        cepSNR(noisyDftFrame, noisePow, dFs, true, [], ...
                        stConfig.iSmoothingConstantSetting, vVDA(indFr));
                else
                    [snrPrio, cepSNRstateDereverb, vVoiced(indFr, 2), vPitch(indFr, 2)] = ...
                        cepSNR(noisyDftFrame, noisePow, dFs, false, ...
                        cepSNRstateDereverb, stConfig.iSmoothingConstantSetting, ...
                        vVDA(indFr));
                end
            case 'dd'
                % compute a posteriori SNR
                snrPost = noisyPer ./ noisePow;
                
                % compute a priori SNR
                if indFr == 1
                    snrPrio = max(0, snrPost - 1);
                else
                    snrPrio = alphaDD .* abs(shatDftFrame).^2 ./ noisePowLast + ...
                        (1 - alphaDD) .* max(0, snrPost - 1);
                end
                
                noisePowLast = noisePow;
            otherwise
                error('unknown estimation method')
        end
        
        % threshold snr
        snrPrio = max(snrPrio, SNR_LOW_LIM);
        
    end
    
    switch lower(stConfig.szGainFunction)
        case 'betaorder'
            %% beta order gain function
            vTempEnhanced = betaOrder(noisyDftFrame, noisePow, ...
                snrPrio, stConfig.dBetaOrderMue, stConfig.dBetaOrderBeta);
        case 'wiener'
            %% Weener Filter
            vTempEnhanced = snrPrio ./ (1 + snrPrio) .* noisyDftFrame;
    end
    
    %% apply spectral floor
    if 0
        gain = max(abs(vTempEnhanced) ./ abs(noisyDftFrame), ...
            MIN_GAIN .* noisePow ./ (noisePow + lambdaSl));
    else
        gain = max(abs(vTempEnhanced) ./ abs(noisyDftFrame), MIN_GAIN);
    end
    
    %% store matrices
    shatDftFrame = gain .* noisyDftFrame(1:frLen/2+1);
    
    shat_ = real(ifft([shatDftFrame; conj(shatDftFrame(end-1:-1:2))], 'symmetric'));
    shat(matSubIdx(:, indFr)) = shat(matSubIdx(:, indFr)) + shat_.*synWin;
end

%% here phase sensitive estimator from Martin
if stConfig.bPhaseSensitive
    % get amplitude of DFT frames
    absX = abs(matSpectra);
    
    % get phase from DFT frames
    phaseX = angle(matSpectra);
    
    % compute number of bins
    nBins = frLen / 2 + 1;
    
    % Distance between center-frequencies of two adjacent STFT-bands in [Hz]
    fftBA       = dFs / frLen;
    
    magEnh         = zeros(nBins,nFrames);  % enhanced amplitude
    phaseEnh       = zeros(nBins,nFrames);  % reconstructed phase
    fftpadder      = zeros(nBins-2,1);      % just needed for ifft
    normalizer     = frLen / frShift /2;      % normalizing term for overlap add (higher overlap than 50%)
    
    
    % Estimate fundamental frequency here: (or within the for loop if you like)
    if dFs == 16000
        gmmmodel = 'white_babble_volvo_timit_16khz_allsnr_20130605_original.model';
    elseif dFs == 8000
        gmmmodel = 'white_babble_volvo_timit_8khz_allsnr_20130619_original.model';
    else
        error('Sampling frequency not supported! Aborting!');
    end
    
    % initialise fundamental frequency estimator
    stGonza = InitModGonzalez(stConfig.dFs, ...
        'freq_smoothing_window', 0.6, ... % st�rkere Gl�ttung
        'allow_complete_smoothing', false, ... % Komplettgl�ttung abstellen
        'realtime_smooth', true, ... % echtzeitgl�ttung aktivieren
        'vud_smoother', 'dynamicprogrammingoriginal', ...
        'smoother', 'dynamicprogrammingoriginal', ...
        'backtrace', false, ... % R�ckverfolgung abschalten
        'gmm_model_file', gmmmodel, ...
        'hop_size', frShift / stConfig.dFs);
    
    centerf0YN = 1;
    winlen     = 90.5e-3;
    
    if centerf0YN == 1
        [f0,~,voicedYN,~,voicedProb] = ProcessModGonzalez(stGonza, [zeros(round((winlen - frLen ./ dFs)/2 * dFs),1); noisy; zeros(round((winlen - frLen ./ dFs)/2 * dFs),1)]);
    else
        [f0,~,voicedYN,~,voicedProb] = ProcessModGonzalez(stGonza, [zeros(round((winlen - frLen ./ dFs) * dFs),1); noisy]);
    end
    
    % pre allocate new output signal
    sHat           = zeros(size(noisy));        % time-domain reconstruction
    sHatappendedPhase = zeros(size(noisy));     % time-domain reconstruction (using noisy phase for reconstruction)
    
    % Frame-wise processing:
    for frameIdx = 1:nFrames
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Phase Estimation:
        
        % Set enhanced phase to noisy phase as a starting point for enhancement:
        phaseEnh(:,frameIdx) = phaseX(:,frameIdx);
        
        % Check if current frame is voiced-speech onset:
        % isOnsetYN = voicedYN(frameIdx) > voicedYN(max(1,frameIdx-1));
        
        % Fundamental-Frequency-Dependent Bandwidth (neighboring frequency bands also dominated by a specific harmonic):
        deltaFreq = f0(frameIdx)/2;
        deltaBin  = floor(deltaFreq/fftBA) + 1;
        
        if frameIdx>1 % && voicedYN(frameIdx)
            % Compute harmonic frequencies and corresponding frequency bins:
            freqMax     = dFs/2-fftBA;                      % Upper frequency limit for phase enhancement
            nHarmonics  = floor(freqMax  / f0(frameIdx));  % Number of harmonics
            frequencies = [1:nHarmonics].' * f0(frameIdx); % Frequencies of the harmonics
            freqBin     = 1 + round(frequencies/fftBA);    % STFT-bands containing the harmonic frequencies
            
            % a) Estimation of pitch-phase along TIME:
            % if ~isOnsetYN
            phaseEnh(freqBin,frameIdx) = phaseEnh(freqBin,frameIdx-1) + 2.*pi.*frequencies.* (frShift ./ dFs);
            % end
            
            % b) Phase estimation along FREQUENCY:
            if f0(frameIdx)>fftBA % only do this if harmonics can be resolved (not too short STFT / too low fundamental frequency)
                % exchange phase in neighboring bands:
                for binCntr = -deltaBin:+deltaBin
                    binIdx = max(2,min(freqBin+binCntr,nBins-1)); % Omit index out of bounds errors
                    if ~any(binIdx==freqBin)
                        % Approximating the phase response of the analysis window by a linear phase -> jumps of pi from one band to the next
                        phaseEnh(binIdx,frameIdx) = phaseEnh(freqBin,frameIdx) + binCntr*pi;
                    end
                end
                % Take care of boundaries (lowest & highest frequencies):
                if (freqBin(1)-deltaBin) >2
                    toFixIdx = freqBin(1)-deltaBin-1:-1:2;
                    phaseEnh(toFixIdx,frameIdx) = phaseEnh(freqBin(1),frameIdx) + (-deltaBin-[1:length(toFixIdx)])*pi;
                end
                if (freqBin(end)+deltaBin) < nBins-1
                    toFixIdx = freqBin(end)+deltaBin+1:nBins-1;
                    phaseEnh(toFixIdx,frameIdx) = phaseEnh(freqBin(end),frameIdx) + (deltaBin+[1:length(toFixIdx)])*pi;
                end
            end
        end
        
        % Wrap it!
        phaseEnh(:,frameIdx) = angle(exp(1i*phaseEnh(:,frameIdx)));
        
        % phaseEnh(:,frameIdx) = angle(S(:,frameIdx)); % ORACLE (use abs phase difference and dont humanize! )
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Amplitude Enhancement:
        
        % Estimate noise power, a priori SNR, and a posteriori SNR @ 50% overlap (subsampling):
        noisePow     = ...
            matNoisePSD(:, floor((frameIdx - 1)./ iOverlapSkip) + 1);
        
        %% SNR Re-estimation (use overlap of 50 %)
        if mod(frameIdx - 1, iOverlapSkip) == 0
            switch lower(stConfig.APrioriSNREstimationMethod)
                case 'cepstral'
                    if frameIdx == 1
                        [snrPrio, cepSNRstateDereverb] = ...
                            cepSNR(matSpectra(:, frameIdx), noisePow, dFs, true);
                    elseif mod(frameIdx - 1, iOverlapSkip) == 0
                        [snrPrio, cepSNRstateDereverb] = ...
                            cepSNR(matSpectra(:, frameIdx), noisePow, dFs, false, cepSNRstateDereverb);
                    end
                case 'dd'
                    % compute a posteriori SNR
                    snrPost = (absX(:, frameIdx).^2) ./ noisePow;
                    
                    % compute a priori snr
                    if frameIdx == 1
                        snrPrio = max(SNR_LOW_LIM, snrPost - 1);
                    elseif mod(frameIdx - 1, iOverlapSkip) == 0
                        snrPrio = max(alphaDD .* abs(magEnh(:,max(1,frameIdx-iOverlapSkip))).^2 ./ noisePowLast  + ...
                            (1 - alphaDD) .* (snrPost - 1), SNR_LOW_LIM);
                    end
                    
                    noisePowLast = noisePow;
                otherwise
                    error('unknown estimation method')
            end
        end
        
        % Phase sensitive amplitude enhancement:
        theta     = phaseX(:,frameIdx) - phaseX(:,max(1,frameIdx-1)); % noisy phase
        phi       = angle(exp(1i*(phaseEnh(:,frameIdx)))) - angle(exp(1i*(phaseEnh(:,max(1,frameIdx-1))))); % Estimated Phase
        phaseDiff = angle(exp(1i*(theta-phi)));
        magEnhTMP = betaOrderGivenPhase(matSpectra(:,frameIdx),noisePow,snrPrio,phaseDiff,mue,beta);
        gainGP    = max(magEnhTMP./absX(:,frameIdx),MIN_GAIN);
        % gainGP    = max(magEnhTMP./abs(Y(:,frameIdx)),setup.MIN_GAIN) .* exp(1i*(phaseEnh(:,frameIdx)-angle(Y(:,frameIdx))));
        
        % Phase-blind (classical) amplitude enhancement:
        SEnhTMP   = betaOrder(matSpectra(:,frameIdx),noisePow,snrPrio,mue,beta);
        gain      = max(abs(SEnhTMP)./absX(:,frameIdx),MIN_GAIN);
        
        % Mix gains based on voiced/unvoiced probability:
        mixFactor = voicedProb(frameIdx);%.*SPPGlobal;
        gain      = mixFactor.*gainGP + (1-mixFactor).*gain;
        
        % Enhanced spectral amplitude:
        magEnh(:,frameIdx) = gain.*absX(:,frameIdx);
        
        % Store enhanced complex coefficients (appended phase):
        % SHat(:,frameIdx)   = magEnh(:,frameIdx).*exp(1i*phaseEnh(:,frameIdx));
        
        % Humanize/Mix the phaseEstimate with the noisy phase (used only for appended phase)
        % -> Should influence the amplitude enhancement in case absolute phase
        % differences are employed instead of differences of phase-shifts
        phaseEnh(:,frameIdx) = humanize(phaseEnh(:,frameIdx),phaseX(:,frameIdx),1-mixFactor.');
        % ALTERNATIVES:
        % a) phase not mixed & used only for amplitude enhancement (-> sHat)
        % b) phase and amplitude mixed separately; mixed phase only used for appending (-> sHatappendedPhase)
        % c) mix complex estimates of S (mixing amplitude and phase at the same time)
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Overlap-add synthesis:
        window_area        = matIdx(:, frameIdx);
        
        % Enhanced amplitude with noisy phase: (Thats what is presented in the paper)
        stftFrame          = magEnh(:,frameIdx).*exp(1i*phaseX(:,frameIdx));
        time_frame         = ifft([stftFrame;fftpadder],frLen,'symmetric');
        sHat(window_area)  = sHat(window_area) + time_frame(1:frLen).*synWin./normalizer;
        
        % Enhanced amplitude AND phase:
        stftFrame          = magEnh(:,frameIdx).*exp(1i*phaseEnh(:,frameIdx));
        time_frame         = ifft([stftFrame;fftpadder],frLen,'symmetric');
        sHatappendedPhase(window_area)  = sHatappendedPhase(window_area) + time_frame(1:frLen).*synWin./normalizer;
        
    end % end of frameIdx
    
    shat = sHatappendedPhase;
end

return
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

function   noise_psd_init =init_noise_tracker_ideal_vad(noisy,fr_size,fft_size,hop,sq_hann_window)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%This m-file computes an initial noise PSD estimate by means of a
%%%%Bartlett estimate.
%%%%Input parameters:   noisy:          noisy signal
%%%%                    fr_size:        frame size
%%%%                    fft_size:       fft size
%%%%                    hop:            hop size of frame
%%%%                    sq_hann_window: analysis window
%%%%Output parameters:  noise_psd_init: initial noise PSD estimate
%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%Author: Richard C. Hendriks, 15/4/2010
%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%

for I=1:5
    noisy_frame=sq_hann_window.*noisy((I-1)*hop+1:(I-1)*hop+fr_size);
    noisy_dft_frame_matrix(:,I)=fft(noisy_frame,fft_size);
end
noise_psd_init=mean(abs(noisy_dft_frame_matrix(1:fr_size/2+1,1:end)).^2,2);%%%compute the initialisation of the noise tracking algorithms.
end

function [mixed] = humanize(est,noisy,mixParam)
% mixParam: 0: no humanizing (only est phase)
%           1: noisy phase only
%           in between: mixing (0.5 -> middle)

%OLD DEFINITION (outdated):
% +1: noisy phase only % -1: estimated phase only % 0: 50% of each phase (Middle)
% Method 1: non-linear mapping along angles. Problem: Mapping is different for different phase differences
% mixed = angle((1+mixParam)*exp(1i*noisy) + (1-mixParam)*exp(1i*est));

% Method 2: linear mapping between noisy and est (BUT WITHOUT mixed1!!!):
% mixParam2 = 0.5*(mixParam + 1);%mapping from -1...1 to 0...1
phaseDiff = angle(exp(1i*(noisy-est)));%./exp(1i*est)); % exp(1i) needed to avoid problems with wrapping!
mixed     = angle(exp(1i*(est + mixParam.'.*phaseDiff))); % here, exp(1i) just to stay on the save side
end
% EOF