function magEnh = betaOrderGivenPhase(noisyDFT,noisePow,snrPrio,phaseDiff,mue,beta)
% Computes an extended beta-order magnitude estimator based on the one proposed in 
%
% Colin Breithaupt, Martin Krawczyk, Rainer Martin, "Parameterized MMSE
% spectral magnitude estimation for the enhancement of noisy speech", IEEE
% Int. Conf. Acoustics, Speech, Signal Processing, Las Vegas, NV, USA, Apr.
% 2008,
%
% where the noisy and clean phase are taken into account as well.
%
% Input:  noisyDFT  - Noisy DFT coefficients or magnitudes (might be a vector)
%         noisePow  - Noise power (same size as noisyDFT)
%         snrPrio   - A priori SNR (same size as noisyDFT)
%         phaseDiff - Difference between noisy and (estimated) clean phase
%         in [rad] (theta-phi) (same size as noisyDFT)
%         mue       - Shape parameter (mue<1 -> supergaussian)
%         beta      - Exponent for compression function
%
% Output: magEnh    - Estimated clean speech amplitude (same size as noisyDFT)
%
% Version 0.1
% June 2012
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Martin Krawczyk
%
% Signal Processing, Fakultät V
% Carl von Ossietzky Universität Oldenburg
% D - 26111 Oldenburg, Germany
%
% Phone:  +49 (0) 441 798-4943
% Email:  martin.krawczyk@uni-oldenburg.de
% WWW:    http://sigproc.uni-oldenburg.de
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

lookupYN=1;

persistent mueLast;
persistent betaLast;
persistent AHatBySqrtNoisePowMat;
persistent sqrtGammaCosPhiRange
persistent prioSNRRange
% persistent prioSNRMat
% persistent sqrtGammaCosPhiMat
persistent sqrtGammaCosPhiRangeDBPos
persistent prioSNRRangeDB

if lookupYN==0
    % Exact formula:
    gammaFact            = gamma(2*mue+beta)./gamma(2*mue);
    snrMueRatio          = snrPrio./(mue+snrPrio);

    cylFuncInput         = - sqrt(2*(noisyDFT.*conj(noisyDFT))./noisePow.*snrMueRatio) .* cos(phaseDiff);
    [parCylFunNum,~,~]   = ParCylFun(-(2*mue+beta),cylFuncInput);
    [parCylFunDenom,~,~] = ParCylFun(-2*mue,cylFuncInput);
    magEnh               = sqrt(0.5.*noisePow .* snrMueRatio) .* ( gammaFact.* parCylFunNum.'./parCylFunDenom.').^(1/beta);
else
    % Check if something changed:
    if isempty(mueLast)
        mueLast  = mue;
        betaLast = beta;
    end
    if (mueLast~=mue) || (betaLast~=beta) || (isempty(AHatBySqrtNoisePowMat))
        mueLast  = mue;
        betaLast = beta;
        [AHatBySqrtNoisePowMat,prioSNRRange,sqrtGammaCosPhiRange] = tabulateBetaOrderGP(mue,beta);
        
        % (V1):----
        %prioSNRRange = prioSNRRange.';
        %prioSNRMat   = prioSNRRange(ones(size(snrPrio)),:);
     
        %sqrtGammaCosPhiRange = sqrtGammaCosPhiRange.';
        %sqrtGammaCosPhiMat   = sqrtGammaCosPhiRange(ones(size(snrPrio)),:);
        
        % (V2):----
        prioSNRRangeDB = pow2db(prioSNRRange);
        sqrtGammaCosPhiRangeDBPos = pow2db(sqrtGammaCosPhiRange(ceil(end/2)+1:end));
        %sqrtGammaCosPhiRangeDBPos = pow2db(sqrtGammaCosPhiRange(ceil(end/2)+1:end));
        %stepDBy                   = sqrtGammaCosPhiRangeDBPos(2)-sqrtGammaCosPhiRangeDBPos(1);
    end
    
    sqrtGammaCosPhi = sqrt((noisyDFT.*conj(noisyDFT))./noisePow).*cos(phaseDiff);
    
    % Look up Indices: (V1)
%       
%     [~,prioIdx]            = min(abs( prioSNRMat-snrPrio(:,ones(1,length(prioSNRRange))) ),[],2);
%     [~,sqrtGammaCosPhiIdx] = min(abs( sqrtGammaCosPhiMat-sqrtGammaCosPhi(:,ones(1,length(sqrtGammaCosPhiRange))) ),[],2);
%     
%     linearIdx = sub2ind(size(AHatBySqrtNoisePowMat), prioIdx, sqrtGammaCosPhiIdx);
%     
%     magEnh = AHatBySqrtNoisePowMat(linearIdx);
   
    
    % Look up Indices: (V2)
    step = prioSNRRangeDB(2)-prioSNRRangeDB(1);

    a_prioridb=round(pow2db(snrPrio)/step)*step;
    [Ia_priori]=min(max(min(prioSNRRangeDB),a_prioridb), max(prioSNRRangeDB));
    Ia_priori=Ia_priori-min(prioSNRRangeDB)+step;
    Ia_priori=max(1,floor(Ia_priori/step));

    % sqrtGammaCosPhi(abs(sqrtGammaCosPhi)<db2pow(-80)) = sign(sqrtGammaCosPhi).*db2pow(-80);
    a_postdb=round(pow2db(abs(sqrtGammaCosPhi))/step)*step;
    [Ia_post]=min(max(min(sqrtGammaCosPhiRangeDBPos),a_postdb), max(sqrtGammaCosPhiRangeDBPos));
    Ia_post=Ia_post-min(sqrtGammaCosPhiRangeDBPos)+step;
    Ia_post=Ia_post/step;
    Ia_post(sign(sqrtGammaCosPhi)==1)  = round(Ia_post(sign(sqrtGammaCosPhi)==1) + ceil(length(sqrtGammaCosPhiRange)/2));
    Ia_post(sign(sqrtGammaCosPhi)==-1) = round(ceil(length(sqrtGammaCosPhiRange)/2) - Ia_post(sign(sqrtGammaCosPhi)==-1));


    magEnh=AHatBySqrtNoisePowMat(Ia_priori+(Ia_post-1)*length(AHatBySqrtNoisePowMat(:,1))); 
    
    
    
    % Get non-normalized amplitude:
    magEnh = magEnh.*sqrt(noisePow);    
end
