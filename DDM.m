function [r] = DDM(nTrial, p, g, d, doShow)
if nargin < 1;  nTrial = 1000;  end;
if nargin < 2
    p.Smax = 4;      % Grid range of states space (now we assume: S = [(zhat1+zhat2)/2, (zhat1-zhat2)/2]); zhat(t) = (varz*X(t)+varX)/(t*varz+varX) )
    p.resS  =  101;  % Grid resolution of state space for route finding
    p.resSH = 4001;  % Higher grid resolution of state space
    p.tmax = 5;      % Time limit
    p.dt   = .005;    % Time step
    p.c    =  0;     % Cost of evidene accumulation
    p.tNull = .25;    % Non-decision time + inter trial interval
    p.a = [10 0];     % Binary reward for [correct, incorrect] 
    p.t = 0:p.dt:p.tmax;
end
if nargin < 3
    g{1}.meanU =  -40; % Prior mean of state (dimension 1)
    g{1}.varU  =  8; % Prior variance of state
    g{1}.varX  =  4; % Observation noise variance
    g{2}.meanU =  0; % Prior mean of state (dimension 2)
    g{2}.varU  =  8; % Prior variance of state
    g{2}.varX  =  4; % Observation noise variance
end
if nargin < 4
    th = .5 * ones(1,length(p.t))' * [1 -1];            %  bDX(iTime, iDec)
    d.type = 'direct';
else
    th = d.bDX;
end
if nargin < 5;  doShow = 1;  end

r.u(:,1) = g{1}.meanU + sqrt(g{1}.varU) * randn(nTrial,1);        % Hidden states
r.u(:,2) = g{2}.meanU + sqrt(g{2}.varU) * randn(nTrial,1);
r.X = zeros(nTrial,length(p.t)); r.nResp = zeros(2,length(p.t)); r.confAvg = zeros(2,length(p.t));
r.RsumD = zeros(2,length(p.t));  r.RsumB = zeros(2,length(p.t)); r.nCorrect = zeros(2,length(p.t));
r.RT = NaN(1,nTrial); r.isTarget = NaN(1,nTrial);
for iT = 1:length(p.t)
    if iT == 1;
        r.X(:,iT) = g{2}.meanU + 0.00001*randn(nTrial,1);        % Initial estimate of hidden state (small noise to avoid waiting just on the boundary)
    else
        dX = p.dt * r.u(:,2) + sqrt(p.dt * g{2}.varX) * randn(nTrial,1);
        r.X(:,iT) = r.X(:,iT-1) + dX;
    end
    for iDec = 1:2
        iTrial = find((-1)^(iDec-1) * r.X(:,iT) > abs(th(iT,iDec)));    % iDec=1: +; iDec=2: -.
        if numel(iTrial)~=0
            r.nResp(iDec,iT) = numel(iTrial);
            r.nCorrect(iDec,iT) = sum((r.u(iTrial,2)>0).*(iDec==1) + (r.u(iTrial,2)<0).*(iDec==2));
            r.confAvg(iDec,iT) = mean(r.X(iTrial,iT));
            r.RsumD(iDec,iT) =  sum(rewardValue(r.u(iTrial,:), iDec, p, 'direct'));
            r.RsumB(iDec,iT) =  sum(rewardValue(r.u(iTrial,:), iDec, p, 'binary'));
            r.X(iTrial,iT) = NaN;
            r.RT(iTrial)   = p.t(iT);
            r.isTarget(iTrial) = (iDec==1)==(r.u(iTrial,2)>0);
        end
    end
end
r.nRespAll = sum(r.nResp,1);
r.correctRate = sum(r.nCorrect,1)./sum(r.nResp,1);
r.correctRateAll = sum(r.nCorrect(:))/nTrial;
r.RTAvg = r.nResp * p.t' ./ sum(r.nResp,2);
r.RTCum = cumul(r.nResp')';
r.RTAvgAll = r.nRespAll * p.t' ./ sum(r.nRespAll,2);
r.RTCumAll = cumul(r.nRespAll')';
r.RsumAllD = sum(r.RsumD,1);
r.RsumAllB = sum(r.RsumB,1);
r.RrateD = (sum(r.RsumAllD,2)/nTrial + p.c*r.RTAvgAll) / (r.RTAvgAll + p.tNull);
r.RrateB = (sum(r.RsumAllB,2)/nTrial + p.c*r.RTAvgAll) / (r.RTAvgAll + p.tNull);
% r.RrateD = sum((r.RsumAllD - p.c*r.nRespAll)) ./ (r.nRespAll * p.t' + p.tNull*nTrial);
% r.RrateB = sum((r.RsumAllB - p.c*r.RTAvgAll)) ./ (r.nRespAll * p.t' + p.tNull*nTrial);

r.z = [r.u(:,1)+r.u(:,2) r.u(:,1)-r.u(:,2)];

if doShow
    figure; colormap bone;
    subplotXY(4,2,1,1); hold on;
        plot(p.t, r.X');
        plot(p.t, th(:,1), 'm.-');
        plot(p.t, th(:,2), 'g.-');
    subplotXY(4,4,1,4); hold on; xlabel('z_1'); ylabel('z_2'); axis square; %axis(20*[-1 1 -1 1]);
        plot(r.z(:,1), r.z(:,2), 'b.','MarkerSize',1);
        h = errorEllipse(cov(r.z), mean(r.z)); set(h,'Color','c');
    subplotXY(4,2,2,1); hold on; ylabel('# trials');
        plot(p.t, r.nResp(1,:), 'm');
        plot(p.t, r.nResp(2,:), 'g');
        plot(r.RTAvg(1), 0, 'm^')
        plot(r.RTAvg(2), 0, 'g^')
    subplotXY(4,2,2,2); hold on; ylabel('Fraction of trials');
        plot(p.t, r.RTCum(1,:), 'm');
        plot(p.t, r.RTCum(2,:), 'g');
    subplotXY(4,2,3,1); hold on; ylabel('Correct rate');
        plot(p.t, sum(r.nCorrect,1)./sum(r.nResp,1),     'r');
    subplotXY(4,2,4,1); hold on; ylabel('R_{sum}'); xlabel('Time');
        plot(p.t, r.RsumAllD, 'k-');
        plot(p.t, r.RsumAllB, 'k:');
        legend({'Direct','Binary'});
    subplotXY(4,4,4,3); hold on; ylabel('R_{sum}'); xlabel('Actual reaward');
        bar([1 2], sum([r.RsumAllD; r.RsumAllB],2));
        set(gca,'XTick',1:2,'XTickLabel',{'Direct','Binary'}); rotateXLabels(gca,45);
    subplotXY(4,4,4,4); hold on; ylabel('Reward rate');
        bar([1 2], [r.RrateD; r.RrateB]);
        set(gca,'XTick',1:2,'XTickLabel',{});

end

function [R] = rewardValue(u, decision, p, type)
switch lower(type)
    case {'direct'}     % Direct (R=z):
        R_{1} = u(:,1)+u(:,2);
        R_{2} = u(:,1)-u(:,2);
        R = R_{decision};
    case {'binary'}     % Binary (R=aP(correct)+bP(incorrect)):
        R = p.a(1) * (u(:,2)>0 & decision==1) + p.a(1) * (u(:,2)<0 & decision==2) + ...
            p.a(2) * (u(:,2)<0 & decision==1) + p.a(2) * (u(:,2)>0 & decision==2);
end