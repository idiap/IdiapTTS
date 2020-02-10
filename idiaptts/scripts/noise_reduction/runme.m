function [shat] = runme(noisy,fs)

% Set config to 2 to replicate results from reverb paper and REVERB journal
stConfig=selectConfig(2);
% Set value for T60 estimate
stConfig.dT60=0;
[shat,~,~,~,g] = ProcessDereverbSpectralSubtract(noisy, stConfig);
