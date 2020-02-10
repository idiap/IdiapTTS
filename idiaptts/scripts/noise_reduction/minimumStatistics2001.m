function [sigmaSquareN,minstatState] = minimumStatistics2001(Yabs,ltSNR,minstatState,conf,firstFrameYN)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This file implements the Minimum Statistic algorithm taken
% from IEEE Trans. On Speech and Audio Processing, Vol. 9, No. 5,
% July 2001.
% input:    Y               : spectrum (0..pi) of the noisy speech signal
%           ltSNR           : long-term SNR (either estimate this online, or put 10dB)
%  	    firstFrameYN    : put 'true' for initialization, 'false' otherwise
%  	    conf.fShift     : frame shift
%  	    conf.fs         : sampling rate
%  	    conf.fL         : frame length
%  	    minstatState    : state variable (memory of function)
% output:   sigmaSquareN    : noise power spectral density estimate
%           minstatState    : updated State variable (memory of function)
%  Dirk Mauler, changes by Timo Gerkmann

% Edit B. Cauchi 2014

%  example call:
% % INITIALIZATION:
%  conf.fL     = 32e-3;
%  conf.fshift = conf.fL/2;
%  conf.fs     = 16e3;;
%  [noisePow,minstatState] = minimumStatistics2001(abs(noisyDftFrame),10,[],conf,true);
% %
% % FRAME PROCESSING:
% for frameInd = 1:nFrames
%  [noisePow,minstatState] = minimumStatistics2001(abs(noisyDftFrame),10,minstatState,conf,false);
% ...process
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 nBins=size(Yabs,1);%fL/2 + 1;
updateRate  = 1;%conf.updateRate; %change this, if you only update every updateRate's  frame
fShift      = conf.fShift;
fs          = conf.fs;
fL          = conf.fL;
dBufferLen  = conf.bufferLen;

if firstFrameYN  %INITIALISIERUNG:nur einmal ausf�hren
    %%%%%%%%%%%%%%%%%%  constants:  %%%%%%%%%%%%%%%%%%
    V           = 192e-3/(fShift*updateRate/fs); %12;                               % length of subwindow
    U           = ceil(dBufferLen*fs/fShift/V/updateRate );        % number of subwindows; total length of 1.536sec
    D           = V*U;                              % entire number of frames for min. search
    ALPHA_MAX   = 0.96;
    ALPHA_MIN   = 0.3;
    ALPHA_Csmooth = 0.7;

    % 512-long/-short window pair, halfwinlen 64, shift 32, prepzeros 64, 32*3-overlap:
    
    Dall   =[1  2    3    4    5    6     7     8     9    10    15    20    30    40    50    60    70    80    90    100    120    140   160   312  468];
    dselall=[1  2              5                8          10    11    12    13    14          16          18                 21     22    23    24    25];
    mdall  =[0  0.26           0.48             0.58       0.61  0.668 0.705 0.762 0.8         0.841       0.865              0.89   0.9   0.91  0.96  1.00000000];
    Hdall  =[0  0.15           0.48             0.78       0.98  1.55  2.0   2.3   2.52        2.9         3.25               4.0    4.1   4.1   4.1   4.10000000];


    % Spline interpolation of parameter tables
    mdspline = spline(Dall(dselall),mdall,1:Dall(end));
    minstatState.MD       = mdspline(D);
    minstatState.MV       = mdspline(V);
    

    %% pass variables
    minstatState.ALPHA_MAX           = ALPHA_MAX;
    minstatState.ALPHA_MIN           = ALPHA_MIN;
    minstatState.ALPHA_Csmooth       = ALPHA_Csmooth;
    minstatState.D                   = D;
    minstatState.V                   = V;
    minstatState.U                   = U;

    %%%%%%%%%%%%%%  memory allocation:  %%%%%%%%%%%%%%
   % nBins=size(Yabs,1);%fL/2 + 1;
    minstatState.Pmin_u          = zeros(nBins,1);        % current minimum of the circular buffer
    minstatState.alpha_hat       = ones(nBins,1);
    minstatState.B_c             = ones(nBins,1);
    minstatState.B_min           = ones(nBins,1);
    minstatState.B_min_sub       = ones(nBins,1);
    minstatState.var_P           = zeros(nBins,1);
    minstatState.P               = zeros(nBins,1);
    minstatState.sigmaSquareN    = zeros(nBins,1);
    minstatState.minBuffer       = zeros(U,nBins) ;       % circular buffer
    minstatState.actmin          = zeros(nBins,1);
    minstatState.actmin_sub      = zeros(nBins,1);
    minstatState.P_bar           = zeros(nBins,1);
    minstatState.P_sqr_bar       = zeros(nBins,1);
end % of inconstants and memory allocation  of indy ==1
%%%%%%%%%%%%%%%%%%%%%%  1st frame only:  %%%%%%%%%%%%%%%%%%%%%%%%%%%%
if firstFrameYN
    P            = Yabs.^2;
    sigmaSquareN = 0.7*Yabs.^2 +eps;
    Pmin_u       = Yabs.^2;
    actmin       = realmax*ones(size(Yabs));%Yabs.^2; %% beseitigt klimpern in den ersten 1.5 sek
    actmin_sub   = realmax*ones(size(Yabs));%Yabs.^2;
    minBuffer(1:U,:)=repmat(realmax,U,nBins);
    %     end
    alpha_c         = 0.9;
    ltSNR            = 10;                       % long-term SNR
    subwc           = 1;                        % subwindow counter
    minBufferi      = 1;                        % index to circular buffer
    lmin_flag       = zeros(size(actmin));


    % pass variables
    minstatState.P            = P                     ;
    minstatState.sigmaSquareN = sigmaSquareN          ;
    minstatState.Pmin_u       = Pmin_u                ;
    minstatState.actmin       = actmin                ;
    minstatState.actmin_sub   = actmin_sub            ;
    minstatState.actmin       = actmin                ;
    minstatState.actmin_sub   = actmin_sub            ;
    minstatState.minBuffer    = minBuffer             ;
    minstatState.alpha_c      = alpha_c               ;
    minstatState.subwc        = subwc                 ;
    minstatState.minBufferi   = minBufferi            ;
    minstatState.lmin_flag    = lmin_flag             ;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%               loop over all frames (> first frame):
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% input for Minimum Statistics:
% 1)  Y             : noisy speech spectrum, Y(lambda,k)
% 2)  ltSNR           : long-term SNR

% additionally you have to provide the values of the Min.Statistics state
% variables which have been obtained in the preceding frame.

if ~firstFrameYN   %loop

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%   read variables           %%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ALPHA_MAX   = minstatState.ALPHA_MAX   ;
    ALPHA_MIN   = minstatState.ALPHA_MIN   ;
    ALPHA_Csmooth= minstatState.ALPHA_Csmooth;
    D           = minstatState.D           ;
    V           = minstatState.V           ;
    U           = minstatState.U           ;

    MD       = minstatState.MD ;
    MV       = minstatState.MV ;
    
    Pmin_u       = minstatState.Pmin_u       ;          % current minimum of the circular buffer
    alpha_hat    = minstatState.alpha_hat    ;
    B_c          = minstatState.B_c          ;
    B_min        = minstatState.B_min        ;
    B_min_sub    = minstatState.B_min_sub    ;
    var_P        = minstatState.var_P        ;
    P            = minstatState.P            ;
    sigmaSquareN = minstatState.sigmaSquareN ;
    minBuffer    = minstatState.minBuffer    ;         % circular buffer
    actmin       = minstatState.actmin       ;
    actmin_sub   = minstatState.actmin_sub   ;
    P_bar        = minstatState.P_bar        ;
    P_sqr_bar    = minstatState.P_sqr_bar    ;
    alpha_c       = minstatState.alpha_c     ;
    subwc         = minstatState.subwc       ;
    minBufferi    = minstatState.minBufferi  ;
    lmin_flag     = minstatState.lmin_flag    ;
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    alpha_c_tilde = 1./(  1+( sum(P)./sum(Yabs.^2)-1 ).^2  );

    alpha_c   = ALPHA_Csmooth*alpha_c + (1-ALPHA_Csmooth)*max(alpha_c_tilde,0.7); % paper:  ALPHA_Csmooth=0.7

    alpha_hat     = (ALPHA_MAX*alpha_c)./(1+(P./sigmaSquareN-1).^2);

    alpha_min     = max(min(ALPHA_MIN,ltSNR.^(-fShift*updateRate/(0.064*fs))),0.05);  % ARequire that P can decay from its peak value to the noise level in about 64ms (4 frames at fL = 256, fs = 8kHz)
    alpha         = max(alpha_hat,alpha_min);
    %     alpha_min_(round(indy/updateRate)) = alpha_min;
    %     alpha_(:,round(indy/updateRate)) = alpha;

    P = alpha.* P + (1-alpha).*(Yabs.^2);
    beta = min(alpha.^2,0.8);  % Smoothing constant for a second smoothed Periodogram required to estimate the variance of P and finally the dregrees of Freedom Qeq

    P_bar           = beta .* P_bar     + (1-beta).*P;
    P_sqr_bar       = beta .* P_sqr_bar + (1-beta).*P.^2;
    
    var_P           = P_sqr_bar - P_bar.^2;
    
    oneOverQeq      = min(var_P./(2*sigmaSquareN.^2),0.5);

    oneOverQeq_bar  = mean(oneOverQeq);
    B_c             = 1 + 2.12*sqrt(oneOverQeq_bar);


    
    Qeq_tilde       = (1./oneOverQeq - 2*MD)./(1-MD);
    Qeq_tilde_sub   = (1./oneOverQeq - 2*MV)./(1-MV);
    B_min           = 1 + (D-1).*2./Qeq_tilde;      % (approximation)
    B_min_sub       = 1 + (V-1).*2./Qeq_tilde_sub;  % (approximation)


    PBB             = P.* B_c.* B_min;              % temporary actmin
    PBB_sub         = P.* B_c.* B_min_sub;          % temporary actmin_sub

    k_mod = find(PBB < actmin);                     % only if PBB is smaller than old actmin
    actmin(k_mod)     = PBB(k_mod);
    actmin_sub(k_mod) = PBB_sub(k_mod);

    if subwc == V,                                  % last window of the subwindow

        lmin_flag(k_mod) = zeros(size(k_mod));

        minBuffer(minBufferi,:) = actmin ;          % circular buffer update
        minBufferi = mod(minBufferi,U)+1 ;          % pointer update
        [Pmin_u(:) Pmin_u_index] = min(minBuffer);
        alter_index = mod(minBufferi + U - Pmin_u_index,U);

        if (oneOverQeq_bar < 0.03)
            noise_slope_max =  8;
        elseif (oneOverQeq_bar < 0.05)
            noise_slope_max = 4;
        elseif (oneOverQeq_bar < 0.06)
            noise_slope_max = 2;
        else
            noise_slope_max =  1.2;
        end
        occurrence = find( lmin_flag...
            & ( actmin_sub < noise_slope_max*Pmin_u )...
            & ( actmin_sub > Pmin_u ) );
        if length(occurrence) > 0
            Pmin_u (occurrence) = max(Pmin_u(occurrence),actmin_sub(occurrence));
            minBuffer(:,occurrence) = repmat(Pmin_u(occurrence).',U,1);
        end
        lmin_flag = zeros(size(actmin));
        subwc     = 1;
        actmin    = realmax*ones(nBins,1);

    else
        if subwc > 1
            lmin_flag(k_mod) = ones(size(k_mod));
        end
        subwc=subwc+1;
        alter_index = zeros(1,nBins);
    end

    sigmaSquareN = real(min(actmin_sub,Pmin_u));
    Pmin_u       = sigmaSquareN;


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% return variables
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    minstatState.Pmin_u        = Pmin_u      ;          % current minimum of the circular buffer
    minstatState.alpha_hat     = alpha_hat   ;
    minstatState.B_c           = B_c         ;
    minstatState.B_min         = B_min       ;
    minstatState.B_min_sub     = B_min_sub   ;
    minstatState.var_P         = var_P       ;
    minstatState.P             = P           ;
    minstatState.sigmaSquareN  = sigmaSquareN +eps;
    minstatState.minBuffer     = minBuffer   ;         % circular buffer
    minstatState.actmin        = actmin      ;
    minstatState.actmin_sub    = actmin_sub  ;
    minstatState.P_bar         = P_bar       ;
    minstatState.P_sqr_bar     = P_sqr_bar   ;
    minstatState.alpha_c      = alpha_c       ;
    minstatState.ltSNR        = ltSNR         ;
    minstatState.subwc        = subwc         ;
    minstatState.minBufferi   = minBufferi    ;
    minstatState.lmin_flag    = lmin_flag     ;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                   end of loop over all frames.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    minstatState.oneOverQeq = oneOverQeq;
end
return


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% EOF