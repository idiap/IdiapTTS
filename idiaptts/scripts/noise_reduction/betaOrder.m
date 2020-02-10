function [SHat] = betaOrder(noisyDFT,noisePow,prioSNR,mue,beta)
% Estimates the STFT clean speech via the estimation formula given in
% equation (5) in 
%
% Colin Breithaupt, Martin Krawczyk, Rainer Martin, "Parameterized MMSE
% spectral magnitude estimation for the enhancement of noisy speech", IEEE
% Int. Conf. Acoustics, Speech, Signal Processing, Las Vegas, NV, USA, Apr.
% 2008.
%
% ATTENTION: Approximative formula (2.13)+(2.14) as presented in Colins diss 
% are used here! Fast and relatively acurate, but only valid for a limited
% set of combinations of mue and beta!
%
% Input:  noisyDFT  - Noisy DFT coefficients (complex) (might be a vector)
%         noisePow  - Noise power (same size as noisyDFT)
%         prioSNR   - A priori SNR (same size as noisyDFT)
%         mue       - Shape parameter (mue<1 -> supergaussian)
%         beta      - Exponent for compression function
%
% Output: SHat      - Estimated clean speech STFT (amplitude and phase) (same size as noisyDFT)
%
% Version 0.1
% July 2012

gammaFactor = (gamma(mue+beta/2)./gamma(mue));

postSNR = (abs(noisyDFT).^2)./noisePow;
nue     = prioSNR./(mue+prioSNR).*postSNR;
% AHat    = sqrt(prioSNR./(mue+prioSNR)) .* ( gammaFactor .* (hypergeom(1-mue-beta/2,1,-nue)./hypergeom(1-mue,1,-nue)) ).^(1/beta) .* sqrt(noisePow);
% SHat    = AHat .* exp(1i*angle(noisyDFT));

% Approximation: (p.25 of Colin Breithaupt's dissertation)
aHat0 = sqrt(prioSNR./(mue+prioSNR)) .* (gammaFactor).^(1/beta) .* sqrt(noisePow); %(2.13) - Output for zero-input
if (beta==1 || beta==2) && mue==1
    p0 = 0.2; pInf = 1;
elseif beta==0.5 && (mue==1 || mue==0.5)
    p0 = 0.5; pInf = 1;
elseif beta==0.001 && mue==1
    p0 = 0.3; pInf = 1.2;
elseif beta==0.001 && mue==0.5
    p0 = 0.5; pInf = 1.5;
elseif beta==0.001 && mue==0.3
    p0 = 0.1; pInf = 2.8;
else
    display(['No approximation found for \mu=' num2str(mue) ' and \beta=' num2str(beta) '! Aborting!']);
    return;
end
AHat = (1./(1+nue)).^p0 .* aHat0 + (nue./(1+nue)).^pInf .* prioSNR./(mue+prioSNR) .* abs(noisyDFT); %(2.14)
SHat    = AHat .* exp(1i*angle(noisyDFT));

% EOF