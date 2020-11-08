function valueDecisionBoundaryRR()
% This code is based on the code of the paper by Tajima, Drugowitsch & Pouget (2016) [1].
% It extends it by including numerical simulation of a Drift Diffusion
% Model with and without value-sensitive noise
% 
% CITATION:
% [1] Satohiro Tajima*, Jan Drugowitsch*, and Alexandre Pouget.
% Optimal policy for value-based decision-making. 
% Nature Communications, 7:12400, (2016). 
% *Equally contributed.

tic;
utility='linear';
Smax = 8;      % Grid range of states space (now we assume: S = [(Rhat1+Rhat2)/2, (Rhat1-Rhat2)/2]); Rhat(t) = (varR*X(t)+varX)/(t*varR+varX) )
resS = 401;      % Grid resolution of state space
tmax = 3;       % Time limit
dt   = .01;       % Time step
c    = 0.1;       % Cost of evidene accumulation
tNull = 1;     % Non-decision time + inter trial interval
priorMeanValue = 0; % fix prior of mean to a fixed value
g{1}.meanR = priorMeanValue; % Prior mean of state (dimension 1)
g{1}.varR  = 5; % Prior variance of stte
g{1}.varX  = 2; % Observation noise variance
g{2}.meanR = priorMeanValue; % Prior mean of state (dimension 2)
g{2}.varR  = 5; % Prior variance of state
g{2}.varX  = 2; % Observation noise variance
t = 0:dt:tmax;
Sscale = linspace(-Smax,Smax,resS);
[S{1},S{2}] = meshgrid(Sscale, Sscale);
iS0 = [findnearest(g{1}.meanR, Sscale) findnearest(g{2}.meanR, Sscale)];
Slabel = {'$$\hat{r}_1$$', '$$\hat{r}_2$$'};

%% Utility functions:
if contains(utility , 'linear')
    utilityFunction = @(X) X;          % Linear utility function 
elseif contains(utility , 'tan')
    utilityFunction = @(X) tanh(X);    % Saturating utility function 
elseif contains(utility , 'logisticL')
    logslope = 0.75; % slope parameter for logistic utility function
    utilityFunction = @(X) -maxval + maxval*2./(1+exp(-logslope*(X)));
elseif contains(utility , 'logisticH')
    logslope = 1.5; % slope parameter for logistic utility function
    utilityFunction = @(X) -maxval + maxval*2./(1+exp(-logslope*(X)));
elseif contains(utility , 'sqrt')
    utilityFunction = @(X) sign(X).*abs(X).^0.5;
end

Rh{1} = utilityFunction(S{1}); % Expected reward for option 1
Rh{2} = utilityFunction(S{2}); % Expected reward for option 2

% figure;
% x=(-Smax:0.1:Smax);
% plot(x,utilityFunction(x));

%% Run set of tests to measure value-sensitivity in equal alternative case
computeDecisionBoundaries=false; % if true the code will compute the decision boundaries, if false will try to load the decision boundaries from the directory rawData (if it fails, it will recompute the data)
singleDecisions=false; % if true we model single decision (i.e. expected future rewards=0); if false we compute the expected future reward (rho_)
meanValues = -1.5:0.1:1.5; % mean reward values to be tested 
noiseValues = [0 0.5 1 2]; % multiplicative noise strength. Value 0 is not present, for values >0, the value-sensitive noise is active.
savePlots = false; % set true only for few runs (e.g. 6)
numruns = 1000; % number of simulations per test case

% prepare file suffix
if singleDecisions
    singleDecisionsSuffix='-singleDec';
else
    singleDecisionsSuffix='-multiDec';
end
suffix=[singleDecisionsSuffix '_u-' utility];

for multNoise = noiseValues
    allResults = zeros(length(meanValues)*numruns, 3); % structure to store the simulation results
    j=1; % result line counter
    for meanValue = meanValues
        option1Mean = meanValue; option2Mean = meanValue; % set actual mean, from which the evidence data is drawn, equal for both options
        fprintf('mag-sensitive noise: %d and mean value: %d \n',multNoise, meanValue);
        filename = strcat('rawData/D_prior-',num2str(g{1}.meanR),'_S-',num2str(Smax),'-',num2str(resS),'_c-',num2str(c),'_t-',num2str(tmax),'_dt-',num2str(dt),'_tNull-',num2str(tNull),singleDecisionsSuffix,'_u-',utility,'.mat');
        dataLoaded = false;
        if meanValue==meanValues(1) % skip loading/computing the decision matrix after the first loop
            if ~computeDecisionBoundaries % load the decision threshold for all timesteps (matrix D)
                try
                    load(filename, 'D','rho_');
                    dataLoaded = true;
                catch
                    disp('Could not load the decision matrix. Recomputing the data.');
                end
            end    
            if ~dataLoaded % compute the decision threshold for all timesteps
                fprintf('computing boundaries...');
                iS0 = [findnearest(g{1}.meanR, Sscale) findnearest(g{2}.meanR, Sscale)];
                if singleDecisions
                    rho_ = 0; % we assume single decisions
                else
                    rho_ = fzero(@(rho) backwardInduction(rho,c,tNull,g,Rh,S,t,dt,iS0), g{1}.meanR, optimset('MaxIter',10));
                    fprintf('rho for mean %d is %d...',g{1}.meanR, rho_);
                end
                [V0, V, D, EVnext, rho, Ptrans, iStrans] = backwardInduction(rho_,c,tNull,g,Rh,S,t,dt,iS0);
                fprintf('saving boundaries to file...');
                save(filename,'D','rho_', '-v7.3');
                fprintf('done.\n');
            end
        end
        if savePlots % prepare the figure
            figure();
            clf;
            set(gcf, 'PaperUnits', 'inches');
            set(gcf, 'PaperSize', [8 1+numruns*1.5]);
            set(gcf, 'PaperPositionMode', 'manual');
            set(gcf, 'PaperPosition', [0 0 8 1+numruns*1.5]);
        end
        for run = 1:numruns
            r1sum=0; % sum of observations opt 1
            r2sum=0; % sum of observations opt 2
            simTraj = [ ];
            for iT = 1:1:length(t)-1
                r1m = ( (g{1}.meanR * g{1}.varX) + (r1sum * g{1}.varR) ) / (g{1}.varX + t(iT) * g{1}.varR ); % compute the posterior mean 1
                r2m = ( (g{2}.meanR * g{2}.varX) + (r2sum * g{2}.varR) ) / (g{2}.varX + t(iT) * g{2}.varR ); % compute the posterior mean 2
                if savePlots 
                    simTraj = [simTraj; r1m r2m ];
                end
                % find the index of the posterior mean to check in matrix D if decision is made
                r1i = findnearest(r1m, Sscale, -1);
                r2i = findnearest(r2m, Sscale, -1);
                if isempty(r1i)
                    if r1i < -Smax; r1i=1; else; r1i=length(Sscale); end
                end
                if isempty(r2i)
                    if r2i < -Smax; r2i=1; else; r2i=length(Sscale); end
                end
                %fprintf('For values (%d,%d) at time %d the D is %d\n',r1m,r2m, t(iT), D(r1i,r2i,iT) )
                try
                    decisionMade = D(r1i,r2i,iT)==1 || D(r1i,r2i,iT)==2 || D(r1i,r2i,iT)==1.5;
                catch
                    fprintf('ERROR for D=%d, t=%d, r1m=%d, r2m=%d, r1i=%d, r2i=%d,\n',D(r1i,r2i,iT),iT,r1m,r2m,r1i,r2i);
                end
                if decisionMade
                    break
                else
                    r1sum = r1sum + normrnd(option1Mean*dt, sqrt(g{1}.varX + multNoise*(option1Mean^2))*sqrt(dt) ); % add the last observation opt 1 to the sum
                    r2sum = r2sum + normrnd(option2Mean*dt, sqrt(g{2}.varX + multNoise*(option2Mean^2))*sqrt(dt) ); % add the last observation opt 2 to the sum
                end
            end
            if savePlots
                subplot(ceil(numruns/2),2,run); imagesc(Sscale, Sscale, D(:,:,iT), [1 3]); axis square; axis xy; title(['D=' num2str(D(r1i,r2i,iT)) ' t=' num2str(t(iT)) ]); 
                xlabel(Slabel{1},'Interpreter','Latex'); ylabel(Slabel{2},'Interpreter','Latex');
                hold on; plot(simTraj(:,1),simTraj(:,2),'r','linewidth',2);
                hold on; plot(r1m,r2m,'wo','linewidth',2);
%                 filename = strcat('rawData/traj_r1-',num2str(option1Mean),'_r2-',num2str(option2Mean),'_',num2str(run),'.txt');
%                 csvwrite(filename,simTraj);
            end
            allResults(j,:) = [ meanValue D(r1i,r2i,iT) t(iT) ];
            j = j+1;
        end
        if savePlots
            filename = strcat('figures/traj_prior-',num2str(g{1}.meanR),'_value-',num2str(meanValue),'_noise-',num2str(multNoise),'.pdf');
            saveas(gcf,filename)
        end
    end
    filename = strcat('resultsData/dmm-results',suffix,'_msn-',num2str(multNoise),'.txt');
    csvwrite(filename,allResults);
end

%% Plot multiplicative noise results
meanValues = 0:0.1:1.5; % mean reward values to be tested 
MNValues = [0 0.5 1 2];
singleDecisionsSuffix='-singleDec';
singleDecisionsSuffix='-multiDec';
suffix=[singleDecisionsSuffix '_u-' utility];
legendSize=18;
axesSize=18;
figure();
j=0;
colors=[[0, 0.7, 0.9]; [0,0,0.9]; [0,0.7,0]; [0,0.7,0]];
linestyles=['- '; '--'; ': '; '-.'];
for MNdata = MNValues
    j=j+1;
    filename = strcat('resultsData/dmm-results',suffix,'_msn-',num2str(MNdata),'.txt');
    allResults = readtable(filename);
    dataForPlot = [];
    for meanValue = meanValues
        dataForPlot = [ dataForPlot, allResults{ abs(allResults{:,1} - meanValue)<0.0000001, 3}];
    end
    errorbar(meanValues, mean(dataForPlot), std(dataForPlot)/sqrt(height(allResults)/length(meanValues))*1.96,'LineWidth', 2, 'DisplayName',strcat('\Phi=',num2str(MNdata)),'LineStyle',linestyles(j,:));%,'Color',colors(j,:)); 
    %title([MNtitle ' mean']);
    hold on;
end
ax = gca;
ax.FontSize = axesSize;
xlabel('Option value','FontSize',axesSize);
ylabel('Reaction time','FontSize',axesSize);
legend('Location','southwest','FontSize',legendSize)
filename = strcat('figures/vs-multiNoise',singleDecisionsSuffix,'.pdf');
saveas(gcf,filename)


function [V0, V, D, EVnext, rho, Ptrans, iStrans] = backwardInduction(rho_,c,tNull,g,Rh,S,t,dt,iS0)
k = 0;
rho = k*S{1}/tNull + (1-k)*rho_;                                                                        % Reward rate estimate
[V(:,:,length(t)), D(:,:,length(t))] = max_({Rh{1}-rho*tNull, Rh{2}-rho*tNull});                        % Max V~ at time tmax
for iT = length(t)-1:-1:1
    [EVnext(:,:,iT), Ptrans{iT}, iStrans{iT}] = E(V(:,:,iT+1),S,t(iT),dt,g);                            % <V~(t+1)|S(t)> for waiting
    [V(:,:,iT), D(:,:,iT)] = max_({Rh{1}-rho*tNull, Rh{2}-rho*tNull, EVnext(:,:,iT)-(rho+c)*dt});       % [Average-adjusted value (V~), decision] at time t
%     fprintf('%d/%d\t',iT,length(t)-1); toc;
end
V0 = mean(vector(V(iS0(1),iS0(2),1)));
fprintf('rho = %d\tV0 = %d\t', rho_, V0); toc;

function R = extrap(mat, varargin)
    % original function missing; identity function used and code fixed around this
    R = mat;

function [EV, Ptrans, iStrans] = E(V,S,t,dt,g)
g{1}.varRh = g{1}.varR * g{1}.varX / (t * g{1}.varR + g{1}.varX);
g{2}.varRh = g{2}.varR * g{2}.varX / (t * g{2}.varR + g{2}.varX);
v1 = varTrans(g{1}.varRh, g{1}.varR, g{1}.varX, t, dt);
v2 = varTrans(g{2}.varRh, g{2}.varR, g{2}.varX, t, dt);
aSscale = abs(S{1}(1,:));
iStrans{1} = find(aSscale<3*sqrt(v1));
iStrans{2} = find(aSscale<3*sqrt(v2));
Ptrans = normal2({S{1}(iStrans{2},iStrans{1}),S{2}(iStrans{2},iStrans{1})}, [0 0], [v1 0; 0 v2]);
mgn = ceil(size(Ptrans)/2);
% Original code has been patched, as it was not working
% V = extrap(V,mgn,[5 5]);
% EV = conv2(V,Ptrans,'same');
% EV = EV(mgn(1)+1:end-mgn(1), mgn(2)+1:end-mgn(2));
V = extrap(V,mgn,[5 5]); % attempt to patch original code by Tajima et al. (2016)
EV = conv2(V,Ptrans,'same'); % marginalise expected value over probabilities of future states

function v = varTrans(varRh, varR, varX, t, dt)
% v = (varR * (varX + varRh)) / ((1 + t/dt) * varR + varX / dt);
v = (varR ./ (varR*(t+dt) + varX)).^2 .* (varX + varRh * dt) * dt;

function prob = normal2(x, m, C)
d1 = x{1} - m(1);
d2 = x{2} - m(2);
H = -1/2*(C\eye(2)); prob = exp(bsxfun(@plus,d1.*d1*H(1,1), d1.*d2*H(1,2)) + d2.*d1*H(2,1) + d2.*d2*H(2,2));
% prob = exp(-(d1.^2/C(1,1)/2 + d2.^2/C(2,2))/2);
prob = prob ./ sum(prob(:));

function [V, D] = max_(x)
x_ = zeros(size(x{1},1), size(x{1},2), length(x));
for k = 1:length(x)
    x_(:,:,k) = x{k};
end
[V, D] = max(x_,[],3);
D(x{1}==x{2} & D==1) = 1.5;

function dbS2 = detectBoundary(D,S,t)
dS = diff(S{2}(1:2,1));
S_ = repmat(S{2},[1 1 length(t)]); S_(D~=1 & D~=1.5) =  Inf; dbS2(:,:,1) = max(squeeze(min(S_))-dS, 0);                % Decision boundary [min(S2;dec=1); max(S2;dec=2)]
S_ = repmat(S{2},[1 1 length(t)]); S_(D~=2 & D~=1.5) = -Inf; dbS2(:,:,2) = min(squeeze(max(S_))+dS, 0);                %  ... bndS2(iS1, iTime, iDec)
mgn = 1; [sm{1},sm{2}] = meshgrid(-mgn:mgn,-mgn:mgn);
for k=1:2;
    %% Extrapolating:
    % db_ = dbS2(:,:,k); db_(~isfinite(db_) & isfinite([db_(:,2:end) db_(:,end)])) = (-1)^(k+1)*max(vector(S{1}));  dbS2(:,:,k) = db_;
    db_ = dbS2(:,:,k); db_(~isfinite(db_) & isfinite([db_(:,2:end) db_(:,end)])) = (-1)^(k+1)*max(max(S{1}));  dbS2(:,:,k) = db_; % changed vector() call to max() ::: attempt to patch original code by Tajima et al. (2016)
    
    %% Smoothing:
    db_ = conv2(extrap(dbS2(:,:,k),mgn),normal2(sm,[0 0],[1 0; 0 1]),'same');  dbS2(:,:,k) = db_(mgn+1:end-mgn,mgn+1:end-mgn);
    dbS2(:,:,k) = db_; % ::: attempt to patch original code by Tajima et al. (2016)
end

function [dbX, dbR] = transformDecBound(dbS2,Sscale,t,g)
S1 = repmat(Sscale',[1 size(dbS2,2) size(dbS2,3)]);
t_ = repmat(t,[size(dbS2,1) 1 size(dbS2,3)]);
for k=1:2;  mR{k}=g{k}.meanR;  vR{k}=g{k}.varR;  vX{k}=g{k}.varX;  end
dbX(:,:,:,1) = (t_+(vX{1}+vX{2})./(vR{1}+vR{2})) .* (S1+dbS2) - (vX{1}+vX{2})./(vR{1}+vR{2}) .* (mR{1}+mR{2});          % X1 (iS1, iTime, iDec, 1)
dbX(:,:,:,2) = (t_+(vX{1}+vX{2})./(vR{1}+vR{2})) .* (S1-dbS2) - (vX{1}+vX{2})./(vR{1}+vR{2}) .* (mR{1}-mR{2});          % X2 (iS1, iTime, iDec, 2)
dbR(:,:,:,1) = (S1+dbS2);          % R1 (iS1, iTime, iDec, 1)
dbR(:,:,:,2) = (S1-dbS2);          % R2 (iS1, iTime, iDec, 2)

function [x_,y_,v_] = plotSurf(Sscale, Val, iS, col, Slabel)
[x,y] = meshgrid(1:length(Sscale), 1:length(Sscale));
x_ = Sscale(x(x+y==iS+round(length(Sscale)/2)));
y_ = Sscale(y(x+y==iS+round(length(Sscale)/2)));
v_ = Val(x+y==iS+round(length(Sscale)/2));
h = surfl(Sscale, Sscale, Val); hold on; %camproj perspective;
set(h,'FaceColor', col, 'EdgeColor','none'); camlight left; lighting phong; alpha(0.7) % replaced sat(.5,col) with col
if ischar(col);  plot3(x_, y_, v_,         col); hold on;
else             plot3(x_, y_, v_, 'Color',col); hold on;  end
xlabel(Slabel{1}); ylabel(Slabel{2}); %zlim([-50 50]);

