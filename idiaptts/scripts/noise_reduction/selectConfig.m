function stConfig = selectConfig(nConf)

stConfig.t60=NaN;
if nConf > 4 && nConf <= 8
    stConfig.phase_sensitive= true;
else
    stConfig.phase_sensitive= false;
end

stConfig.gain_function = 'betaorder';
stConfig.minimum_gain = -10;
    

switch nConf
    case {1,5}
        stConfig.noise_estimator= 'minimumstat';
        stConfig.ms_buffer_length = 3;
        stConfig.apriori_snr_estimation_method = 'dd';
    case {2,6}
        stConfig.noise_estimator= 'minimumstat';
        stConfig.ms_buffer_length = 3;
        stConfig.apriori_snr_estimation_method = 'cepstral';
    case {3,7}
        stConfig.noise_estimator= 'spp';
        stConfig.apriori_snr_estimation_method = 'dd';
    case {4,8}
        stConfig.noise_estimator= 'spp';
        stConfig.apriori_snr_estimation_method = 'cepstral';
    case 9
        stConfig.noise_estimator = 'minimumstat';
        stConfig.ms_buffer_length = 3;
        stConfig.apriori_snr_estimation_method = 'cepstral';
        stConfig.t60_est_fac = 0.8;
    case 10
        stConfig.noise_estimator = 'minimumstat';
        stConfig.ms_buffer_length = 3;
        stConfig.apriori_snr_estimation_method = 'cepstral';
        stConfig.smoothing_constant_variant = 7;
end

cConfig = [fieldnames(stConfig) struct2cell(stConfig)].';

dFs = 16000;
stConfig = InitDereverbSpectralSubtract(dFs, cConfig{:});

end
