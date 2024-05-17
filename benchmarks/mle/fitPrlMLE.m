% Dependencies:
% - Optimization Toolbox (fmincon)
% - Global Optimization Toolbox
% - Parallel Computing Toolbox (for parallel execution)
clear all;

numTrials = 500;
numAgents = 1000;
% Define prior probability of model parameters
prAlphaPos = @(x) unifpdf(x, 1e-4, 1);
prAlphaNeg = @(x) unifpdf(x, 1e-4, 1);
prBeta = @(x) unifpdf(x, 1e-4, 1);
prStickiness = @(x) unifpdf(x, 1e-4, 1);
pdfs = {prAlphaPos, prAlphaNeg, prBeta, prStickiness};

pMin = [1e-4 1e-4 1e-4 1e-4]; % Minimum values for model parameters
pMax = [1 1 1 1]; % Maximum values for model parameters

targetTrial = numTrials;
betaPriors = {'unif'}
for i = 1:numel(betaPriors)
    % load the data
    prefix = sprintf('../data/4prl/%da_%dt', numAgents, numTrials);
    disp(prefix);
    allAgentData = loaddata(numAgents, numTrials, prefix, targetTrial);
    % start fitting with MLE
    fitparams = fitmodel(numAgents, allAgentData, pMin, pMax);
    
    allParamas = vertcat(fitparams{:});
    allParamTable = array2table(allParamas, 'VariableNames', { 'alpha', 'nalpha', 'beta', 'stickiness'});
    
    writetable(allParamTable, sprintf('../results/mle_%da_%dt_parameters.csv', numAgents, targetTrial), 'delimiter',',');
end


function fittedParams = fitmodel(numAgents, allAgentData, pMin, pMax)
    fittedParams = cell(numAgents, 1);
    parfor i = 1:numAgents
        D = allAgentData{i};
        
        % Sample parameter starting values
        par = zeros(length(pMin), 1);
        for ind = 1:length(pMin)
            par(ind) = unifrnd(pMin(ind), pMax(ind));
        end
        
        myfitfun = @(b) negloglikelihood(b, D);
        rng default % For reproducibility
        fmincon_opts = optimoptions(@fmincon, 'Algorithm', 'sqp');
        problem = createOptimProblem('fmincon', 'objective', myfitfun, 'x0', par, 'lb', pMin, 'ub', pMax, 'options', fmincon_opts);
        gs = GlobalSearch;
        [param, fval] = run(gs, problem);
        % err = sqrt(diag(inv(hessian)));
        % disp(err)
        fittedParams{i} = param';
    end
end

function nle = negloglikelihood(par, D)
    nle = llhRL2apc(par(1),par(2),par(3),par(4),D);
end

function allAgentData = loaddata(numAgents, numTrials, prefix, targetTrials)
    data = readtable(sprintf('data/%s.csv', prefix));

    allAgentData = cell(1, numAgents);
    for i = 1:numAgents
        agentData = data((i-1)*numTrials+1:i*numTrials, :);
        allAgentData{i} = agentData(1:targetTrials, :);
    end
end

%% RL 2 alpha + p + counterfactual
function llh = llhRL2apc(alpha,aneg,beta,p,D)

    beta = beta * 10; % rescale beta

    llh = 0; % init llh

    Q = [.5 .5]; % init Qs

    actions = D.actions+1; % convert 0 to 1 and 1 to 2 for indexing.
    rewards = D.rewards;
    for t = 1:size(D,1) % for each trial
        
        
        a = actions(t); % find column for actions
        ns = 3-a; % not selected action
        r = rewards(t); % find column for rewards
        
        W = Q;  % update softmax w/ persistence-altered Q (W)
        if t > 1
            prev_a = actions(t-1);
            W(prev_a) = W(prev_a) + p;
        end
        softmax = exp(beta*W(a))/sum(exp(beta*W));
        
        if r
            Q(a) = Q(a) + alpha * (r - Q(a)); % update
            Q(ns) = Q(ns) + alpha * ((1-r) - Q(ns)); % counterfactual update
        else
            Q(a) = Q(a) + aneg * (r - Q(a)); % update
            Q(ns) = Q(ns) + aneg * ((1-r) - Q(ns)); % counterfactual update
        end
        
        llh = llh + log(softmax);
        
    end

    llh = -llh;

end