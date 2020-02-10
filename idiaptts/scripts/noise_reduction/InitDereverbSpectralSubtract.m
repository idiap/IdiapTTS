% Unfortunately, 'InitDereverbSpectralSubtract' has no description yet.
% By calling InitDereverbSpectralSubtract you will have a lot of fun.
function stConfig = InitDereverbSpectralSubtract(dFs, varargin)

% File   :  InitDereverbSpectralSubtract.m
% Author :  Robert Rehr <r.rehr AT uni-oldenburg.de>
% Date   :  09.09.2013
%
% Updates:

%% parse input arguments
% initialize input parser object
oParser = inputParser();

% add parameters to parser
oParser.addRequired('dFs', @(x) x > 0);

% frame length and shift
oParser.addParamValue('frame_length', 32e-3, @(x) x > 0);

% smoothing constant decision directed a priori SNR estimation
oParser.addParamValue('alpha_dd', 0.98, @(x) x >= 0 && x <= 1);

% switch for noise estimator
oParser.addParamValue('noise_estimator', 'spp', ...
    @(x) strcmpi(x, 'spp') || strcmpi(x, 'minimumstat'));

% Minimum statistics based noise estimator
oParser.addParamValue('ms_buffer_length', 1.536, @(x) x > 0);

% SPP based noise estimator
oParser.addParamValue('spp_apriori_spp', 0.5, @(x) x >= 0 && x <= 1);
oParser.addParamValue('spp_xi_opt', 15);
oParser.addParamValue('spp_alpha_mean', 0.9, @(x) x >= 0 && x <= 1);
oParser.addParamValue('spp_alpha_psd', 0.8, @(x) x >= 0 && x <= 1);

% reverberation time
oParser.addParamValue('t60', 0.8, @(x) x >= 0 || isnan(x));

% setting for reverb underestimation
oParser.addParamValue('t60_est_fac', 1, @(x) x >= 0);
oParser.addParamValue('late_reverb_threshold', 80e-3, @(x) x >= 0);

% initial speech estimation
oParser.addParamValue('apriori_snr_estimation_method', 'cepstral', ...
    @(x) strcmpi(x, 'cepstral') || strcmpi(x, 'dd'));

% smoothing variant
oParser.addParamValue('smoothing_constant_variant', 1);

% switch for dereverberation
oParser.addParamValue('dereverb', true, @islogical);

% minimum gain for filter
oParser.addParamValue('minimum_gain', -17);

% set gain function
oParser.addParamValue('phase_sensitive', false, @islogical);

% set parameter for gain function
oParser.addParamValue('beta_order_mue', 0.5);
oParser.addParamValue('beta_order_beta', 0.5);

oParser.addParamValue('gain_function', 'betaorder', ...
    @(x) strcmpi(x, 'wiener') || strcmpi(x, 'betaorder'));

oParser.addParamValue('use_pefac_vda', false, @islogical);

% parse input parameters
oParser.parse(dFs, varargin{:});

%% setup struct
% save sampling frequency
stConfig.dFs = oParser.Results.dFs;

% get setting if phase sensitive enhancement should be processed
stConfig.bPhaseSensitive = oParser.Results.phase_sensitive;

% set parameters for beta order mod
stConfig.dBetaOrderMue = oParser.Results.beta_order_mue;
stConfig.dBetaOrderBeta = oParser.Results.beta_order_beta;

% compute frame length and frame shifht in samples
stConfig.iFrameLength = ...
    round(oParser.Results.frame_length .* stConfig.dFs);
% stConfig.iFrameShift = ...
%     round(oParser.Results.frame_length ./ 2 .* stConfig.dFs);

if stConfig.bPhaseSensitive
    stConfig.iOverlapFactor = 8;
else
    stConfig.iOverlapFactor = 2;
end

% smoothing constant setting
stConfig.iSmoothingConstantSetting = ...
    oParser.Results.smoothing_constant_variant;

% late reverberation time
stConfig.dT60 = oParser.Results.t60;
stConfig.dT60EstFac = oParser.Results.t60_est_fac;

% set parameter for decision directed apriori snr estimation
stConfig.dAlphaDD = oParser.Results.alpha_dd;

%% select noise estimator
stConfig.szNoiseEstimator = lower(oParser.Results.noise_estimator);

%% setup minimum statisticss based noise estimator
% frame length
stConfig.stMinStat.fL = stConfig.iFrameLength;

% frame shift
stConfig.stMinStat.fShift = round(stConfig.iFrameLength ./ 2);

% sampling frequency
stConfig.stMinStat.fs = stConfig.dFs;

% ring buffer length
stConfig.stMinStat.bufferLen = oParser.Results.ms_buffer_length;

%% setup for spp noised estimator
% apriori speech presence probababilty
stConfig.stSPP.APrioriSPP = oParser.Results.spp_apriori_spp;

% optimal a priori snr
stConfig.stSPP.dOptimalAPrioriSNR = oParser.Results.spp_xi_opt;

% smoothing constant for mean a posteriori probability
stConfig.stSPP.dAlphaMeanPH1 = oParser.Results.spp_alpha_mean;

% smoothing constant for psd
stConfig.stSPP.dAlphaPSD = oParser.Results.spp_alpha_psd;

%% Filter configuration
% minimum gain
stConfig.dMinimumGain = 10.^(oParser.Results.minimum_gain ./ 20);

%% Reverb suppression
% switch
stConfig.bDereverb = oParser.Results.dereverb;

% time after which 'late reverberation' kicks in (in frames)
stConfig.dLateReverb = oParser.Results.late_reverb_threshold;
stConfig.iLateFrames = stConfig.dLateReverb ./ ...
    (round(stConfig.iFrameLength ./ 2) ./ stConfig.dFs);

%% modes for a priori snr estimation
stConfig.APrioriSNREstimationMethod = ...
    lower(oParser.Results.apriori_snr_estimation_method);

% set gain function
stConfig.szGainFunction = oParser.Results.gain_function;

% use pefac vda
stConfig.bPEFAC = oParser.Results.use_pefac_vda;

% End of InitDereverbSpectralSubtract.m
