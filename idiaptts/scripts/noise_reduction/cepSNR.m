function [snr_apriori,cepSNRstate, bVoiced, dPitch] = cepSNR( noisy, noise_pow, fs, firstFrame, cepSNRstate, iVariant, bVoiced)
    % Timo Gerkmann, Colin Breithaupt, IKA, Ruhr-Universitaet Bochum, 2010
    % - Parameters according to the ICASSP 2008 paper
    % - Bias compensation according to 
    % 	Timo Gerkmann, Rainer Martin, "On the Statistics of Spectral
    % 	Amplitudes After Variance Reduction by Temporal Cepstrum
    % 	Smoothing and Cepstral Nulling", IEEE Trans. Signal
    % 	Processing, Vol. 57, No. 11, pp. 4165-4174, Nov. 2009. 
    %
    % cepSNRstate is the memory of the function
    %
    % to initialize, first call with an empty cepSNRstate, as
    % [snr_apriori,cepSNRstate] = cepSNR( noisy, noise_pow, fs, true)
    % then, use the following in the loop
    % [snr_apriori,cepSNRstate] = cepSNR( noisy, noise_pow, fs, false, cepSNRstate)
    %
    %
    %%%%%%%%%%%%%%%%%%%%%% Copyright (c) 2012, Timo Gerkmann
    %%%%%%%%%%%%%%%%%%%%%% Authors: Timo Gerkmann, Colin Breithaupt, Rainer Martin
    %%%%%%%%%%%%%%%%%%%%%% Universitaet Oldenburg, Germany
    %%%%%%%%%%%%%%%%%%%%%% Ruhr-Universitaet Bochum
    %%%%%%%%%%%%%%%%%%%%%% Contact: timo.gerkmann@uni-oldenburg.de
    % All rights reserved.
    % 
    % Redistribution and use in source and binary forms, with or without
    % modification, are permitted provided that the following conditions are
    % met:
    %
    %     * Redistributions of source code must retain the above copyright
    %       notice, this list of conditions and the following disclaimer.
    %     * Redistributions in binary form must reproduce the above copyright
    %       notice, this list of conditions and the following disclaimer in
    %       the documentation and/or other materials provided with the distribution
    %     * Neither the name of the Universitaet Oldenburg, or Ruhr-Universitat 
    %     Bochum nor the names of its contributors may be used to endorse or 
    %     promote products derived from this software without specific prior 
    %     written permission.
    %
    % THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    % AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    % IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    % ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
    % LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    % CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    % SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    % INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    % CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    % ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    % POSSIBILITY OF SUCH DAMAGE.

    
   
newBiasYN = true;
    
if firstFrame
    % define filename
    szFilePrefix = 'ZetaRiemannTable';
    szFileName = [szFilePrefix '.bin'];
    
    %   disp(' ')
    %    disp('(c) Timo Gerkmann, Universitaet Oldenburg, 2012')
    %    disp('For non-academic use, please contact the author.')
    %   disp(' ')
    fL = (length(noisy)-1)*2;
    if newBiasYN
        if ~exist(szFileName, 'file')
            % compute zeta riemann table
            zetaMatrix = zetaRiemannMakeTable;
            
            % save table to file
            fhZetaTable = fopen(szFileName, 'w', 'ieee-le');

            if fhZetaTable ~= -1
                % write dimension of table
                fwrite(fhZetaTable, size(zetaMatrix), 'uint32');
                
                % write data
                fwrite(fhZetaTable, zetaMatrix, 'double');
                
                % close file handle
                fclose(fhZetaTable);
            else
                warning('cepSNR:WriteFile', ...
                    'Could not open file for writing Zeta-Riemann table!');
            end
        else
            % open file
            fhZetaTable = fopen(szFileName, 'r');
            
            if fhZetaTable ~= -1
                % read dimension
                vZetaMatrixSize = fread(fhZetaTable, 2, 'uint32').';
                
                % read data
                zetaMatrix = fread(fhZetaTable, Inf, 'double');
                
                % reshape data
                zetaMatrix = reshape(zetaMatrix, vZetaMatrixSize);
                
                % close file handle
                fclose(fhZetaTable);
            else
                error('Could not open file containing Zeta-Riemann table!');
            end
        end
        
        mu1 = 1;
        nu = 2*ones(fL/2+1,1); nu(1) = 0.5; nu(end) = 0.5;
        kappa0  = zetaRiemann(2,mu1,zetaMatrix);
        kappa12 = computeKappa(mu1,[2/3 1/6]);
        varBefore = kappa0/fL;
        for indm  = 1:length(kappa12)
            varBefore = varBefore + 1/fL*2*kappa12(indm)*cos(indm*2*pi*(0:fL/2)'/fL);
        end
        varBefore(1) = varBefore(1)*2;varBefore(end) = varBefore(end)*2;
        cepSNRstate.zetaMatrix    = zetaMatrix;
        cepSNRstate.varBefore     = varBefore;
        cepSNRstate.nu            = nu;
        cepSNRstate.mu1           = mu1;
    end
    
    pitch_cutoff = 2000;
    nhamming     = fix(fs/pitch_cutoff);
    cepSNRstate.pitch_win    = hamming(nhamming)/sum(hamming(nhamming));

else
    cepst_speech_pow_ml_prev = cepSNRstate.cepst_speech_pow_ml_prev;
    smoother                 = cepSNRstate.smoother;
    if newBiasYN
        zetaMatrix             = cepSNRstate.zetaMatrix;
        varBefore              = cepSNRstate.varBefore ;
        nu                     = cepSNRstate.nu        ;
        mu1                    = cepSNRstate.mu1       ;
    end
    fL = (length(noisy)-1)*2;
end

snr_ml_min = 10^(-30/10);

% constants
pitch_width     = floor(2 * fs/16e3);

pitch_low       = fix(fs/300)+1;
pitch_high      = min( fL/2+1-pitch_width,  fix(fs/70)+1);
pitch_threshold = .2 * 16e3/fs;

specsize = length(noisy);

smoother_const = zeros(specsize,1);

% introduce smoothing constant factor
% switch iVariant
%     case 2
%         smoother_const(:)               = .92;
%         smoother_const(1:round(fs/8e3)+1)      = .2;
%         smoother_const(round(fs/8e3)+2:round(fs/800)) = .4;
%         pitch_smoother = .15;
%     case 3
%         smoother_const(:)                             = .97;
%         smoother_const(1:round(fs/8e3)+1)             = .3;
%         smoother_const(round(fs/8e3)+2:round(fs/800)) = .5;
%         pitch_smoother                                = .2;
%     case 4
%         smoother_const(:)                             = .9;
%         smoother_const(1:round(fs/8e3)+1)             = .0;
%         smoother_const(round(fs/8e3)+2:round(fs/800)) = .3;
%         pitch_smoother                                = .1;
%     case 5
%         smoother_const(:)                             = .5;
%         smoother_const(1:round(fs/8e3)+1)             = .0;
%         smoother_const(round(fs/8e3)+2:round(fs/800)) = .0;
%         pitch_smoother                                = .0;
%     case 6
%         smoother_const(:)                             = 0;
%         smoother_const(1:round(fs/8e3)+1)             = 0;
%         smoother_const(round(fs/8e3)+2:round(fs/800)) = 0;
%         pitch_smoother                                = 0;
%     % kolossa
%     case 7
%         smoother_const(:) = 0.9;
%         smoother_const(1:round(fs/2e3)) = 0;
%         smoother_const(round(fs/2e3) + 1:round(fs/1e3)) = 0.5;
%         pitch_smoother = 0.9;
%         % no pitch protect
%     case 8
%         smoother_const(:)               = .97;
%         smoother_const(1:round(fs/8e3)+1)      = .5;
%         smoother_const(round(fs/8e3)+2:round(fs/800)) = .7;
%         pitch_smoother = 0.97;
%     % original stuff
%     otherwise
        smoother_const(:)               = .97;
        smoother_const(1:round(fs/8e3)+1)      = .5;
        smoother_const(round(fs/8e3)+2:round(fs/800)) = .7;
        pitch_smoother = .2;
%end

% smoother_const(:)               = .95;
% smoother_const(1:round(fs/8e3)+1)      = .3;
% smoother_const(round(fs/8e3)+2:round(fs/800)) = .6;

if firstFrame
    smoother = smoother_const;
end

alpha_smoother = .96;

%--------------------------

snr_aposteriori = abs(noisy).^2 ./ noise_pow;
snr_apriori_ml = max(snr_ml_min, ...
                snr_aposteriori - 1);

speech_pow_ml = snr_apriori_ml.*noise_pow;

% cepstral transform
log_speech_pow_ml = reallog(max(eps,speech_pow_ml));

cepst_speech_pow_ml_fullspec = ...
    real(ifft([log_speech_pow_ml;...
    conj(log_speech_pow_ml(end-1:-1:2))]));

cepst_speech_pow_ml = ...
    cepst_speech_pow_ml_fullspec(1:specsize);

% PITCH DETECTION
smooth_pitch_array = true;
if smooth_pitch_array,
    pitch_array = filtfilt(cepSNRstate.pitch_win,1,cepst_speech_pow_ml);
else
    pitch_array = cepst_speech_pow_ml;
end
[max_val max_idx] = max( pitch_array(pitch_low:pitch_high) );
max_idx = pitch_low-1 + max_idx;
dPitch = fs ./ max_idx;

% UN-VOICED/VOICED DETECTION
if nargin < 7 || isempty(bVoiced) || isnan(bVoiced)
    if( pitch_array(max_idx) > pitch_threshold ),
        if cepst_speech_pow_ml(2)>0,
            % ... that's c(1) !
        else
            max_idx = 0;
        end
    else
        max_idx = 0;
        % ... deactivates pitch detection
    end
    
    bVoiced = max_idx > 0;
end

% set smoothing constants
smoother = alpha_smoother * smoother ...
    + (1-alpha_smoother) * smoother_const;

if bVoiced,
    smoother(max_idx-pitch_width:max_idx+pitch_width) = pitch_smoother;
end

% recursive cepstral smoothing
if ~firstFrame
    cepst_speech_pow_ml = smoother.*cepst_speech_pow_ml_prev ...
        + (1-smoother).* cepst_speech_pow_ml;
end
cepst_speech_pow_ml_prev = cepst_speech_pow_ml;


% inverse cepstral transform
speech_pow_ml_fullspec = real(fft([cepst_speech_pow_ml;...
    conj(cepst_speech_pow_ml(end-1:-1:2))]));
log_speech_pow = speech_pow_ml_fullspec(1:specsize);


if newBiasYN
  varAfterAv = 1./fL.*sum( nu .* varBefore .* (1-smoother)./(1+smoother));
  mutilde = findMu(fL*varAfterAv,zetaMatrix);
  r2 = mu1/mutilde .* exp( psi(mutilde)-psi(mu1) );
  speech_pow = exp(log_speech_pow).*r2;
else
  speech_pow = exp(log_speech_pow+ 0.5 * 0.5772); %bias
end



% back to SNR
snr_apriori = speech_pow./noise_pow;



%-----------------------
cepSNRstate.cepst_speech_pow_ml_prev = cepst_speech_pow_ml_prev;
cepSNRstate.smoother                 = smoother;


%==================================================================


function mu1 = findMu(zeta,zetaMatrix)
% function mu1 = findMu(zeta,zetaMatrix)
%
% finds mu1 for given value of RIemann's zeta-function
% Timo Gerkmann

%  mu1 = max(zetaMatrix(1,(zetaMatrix(2,:)>varTheo)));

% linear interpolation
if zeta < zetaMatrix(2,end)
  x2 = zetaMatrix(2,end);
  x1 = zetaMatrix(2,end-1);
  y2 = zetaMatrix(1,end);
  y1 = zetaMatrix(1,end-1);
else
  ind = find(zetaMatrix(2,:)<zeta,1,'first');
  x2 = zetaMatrix(2,ind);
  x1 = zetaMatrix(2,ind-1);
  y2 = zetaMatrix(1,ind);
  y1 = zetaMatrix(1,ind-1);
end
slope  = (y2-y1)/(x2-x1);
offset = y1 - slope*x1;
mu1 = offset + slope * zeta;
return



%------------------------------------------------------------------

function zeta = zetaRiemann(z,q,zetaMatrix)
% zeta = zetaRiemann(z,q,zetaMatrix)
%
%Riemann's zeta function Gradshteyn (9.521.1)


if z~=2
  error('only evaluated for z=2')
end
  %   zeta = max(zetaMatrix(2,(zetaMatrix(1,:)>q)));
  
  % linear interpolation
if q > zetaMatrix(1,end)
  x2 = zetaMatrix(1,end);
  x1 = zetaMatrix(1,end-1);
  y2 = zetaMatrix(2,end);
  y1 = zetaMatrix(2,end-1);
else
  [~, ind] = max(zetaMatrix(1,:)>q);
  x2 = zetaMatrix(1,ind);
  x1 = zetaMatrix(1,ind-1);
  y2 = zetaMatrix(2,ind);
  y1 = zetaMatrix(2,ind-1);
end
slope  = (y2-y1)/(x2-x1);
offset = y1 - slope*x1;
zeta = offset + slope * q;
return

%------------------------------------------------------------------

function zetaMatrix = zetaRiemannMakeTable()
%tabulate Riemann's zeta-function according to Gradshteyn (9.521.1)

z=2;

lenZ = 256;

qind = logspace(-2,1.5,lenZ);
%q is mu1

zetaMatrix = zeros(2,lenZ);
ind = 0;
for q = qind
  ind = ind+1;
  n = 0;
  zeta = 0;
  zetaold = zeta;
  zeta = zetaold + 1./(q+n).^z;
  while abs(zeta-zetaold) / zeta >= 1e-8 % precision
    n = n+1;
    zetaold = zeta;
    zeta = zetaold + 1./(q+n).^z;
  end
  zetaMatrix(:,ind) = [q;zeta];
end

%------------------------------------------------------

function kappa = computeKappa(mu1,rho_)
%function kappa = computeKappa(mu1,rho_)
  %computes the variance of correlated log periodograms
  % Timo Gerkmann
  
  if nargin<2
    rho_ = [2/3 1/6];
  end
  
  kappa = zeros(length(rho_),1);
  
  count = 0;
  for rho = rho_
    count = count+1;
  rho2 = rho.^2;
  sum1 = 0; sum2 = 0;
  %  sum3 = 0;
  for k = 0:120
    A = 1/2*(1-rho2)^mu1 ./(sqrt(pi).*gamma(mu1)) .* (1+(-1)^k) .* 2^k .* rho^k .* gamma((k+1)/2) .* gamma(k/2+mu1) ./ factorial(k);
  B = psi(mu1+k/2) + log( 2*(1-rho2) );
  sum1 = sum1 + A .*  B.^2;
  sum2 = sum2 + A .*  B ;
  % integral = 1:
  %  sum3 = sum3 + A; % um zu zeigen, dass pdf gueltig, da flaeche drueber =1
  end
  kappa(count) = sum1 - sum2.^2;
  end
  
%------------------------------------------------------