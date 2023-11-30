% This is the matlab code for the multi-objective bonobo optimizer 
% with problem-decomposition approach(MOBO3).
% This is written for solving unconstrained optimization problems. 
% However, it can also solve constrained optimization
% problems with penalty function approaches.
% For details of the MOBO3 algorithm, kindly refer and cite as mentioned below:
% Das, A.K., Nikum, A.K., Krishnan, S.V. et al. Multi-objective Bonobo Optimizer 
%(MOBO): an intelligent heuristic for multi-criteria optimization. 
% Knowl Inf Syst (2020). https://doi.org/10.1007/s10115-020-01503-x
% For any query, please email to: amit.besus@gmail.com

% I acknowledge that this version of MOBO3 has been written using
% a large portion of the following code:

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  MATLAB Code for                                                  %
%                                                                   %
%  Multi-Objective Particle Swarm Optimization (MOPSO)              %
%  Version 1.0 - Feb. 2011                                          %
%                                                                   %
%  According to:                                                    %
%  Carlos A. Coello Coello et al.,                                  %
%  "Handling Multiple Objectives with Particle Swarm Optimization," %
%  IEEE Transactions on Evolutionary Computation, Vol. 8, No. 3,    %
%  pp. 256-279, June 2004.                                          %
%                                                                   %
%  Developed Using MATLAB R2009b (Version 7.9)                      %
%                                                                   %
%  Programmed By: S. Mostapha Kalami Heris                          %
%                                                                   %
%         e-Mail: sm.kalami@gmail.com                               %
%                 kalami@ee.kntu.ac.ir                              %
%                                                                   %
%       Homepage: http://www.kalami.ir                              %
%                                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;close all;clear all;
tic;  % CPU time measure
nobj=2;
if nobj==2
    fobj=@(x)kursave(x);% Objective function
    d=3;  % No. of Variables
    Var_min=-5*ones(1,d);  % Lower variable Boundaries
    Var_max=5*ones(1,d);   % Upper variable Boundaries
    VarSize=[d 1];
    %% Algorithm-specific Parameters for BO (user should set suitable values of the parameters for their problem)
    p_xgm_initial=0.08; % Initial probability for extra-group mating (generally 1/d for higher dimensions)
    scab=1.55  ;  %Sharing cofficient for alpha bonobo (Generally 1-2)
    scsb=1.4;   % Sharing coefficient for selected bonobo(Generally 1-2)
    rcpp=0.004; % Rate of change in  phase probability (Generally 1e-3 to 1e-2)
    tsgs_factor_max=0.07;% Max. value of temporary sub-group size factor
    %% There is no need to change anything below this %%
    npc=0; % Negative phase count
    ppc=0; % Positive phase count
    p_xgm=p_xgm_initial; % Probability for extra-group mating
    tsgs_factor_initial=0.5*tsgs_factor_max; % Initial value for temporary sub-group size factor
    tsgs_factor=tsgs_factor_initial; % Temporary sub-group size factor
    p_p=0.5; % Phase probability
    p_d=0.5; % Directional probability
    %% Other Settings
    MaxIt=200;  % Maximum Number of Iterations
    N=200;    % Population Size (Number of Sub-Problems)
    nArchive=100;
    T=max(ceil(0.15*N),2);    % Number of Neighbors
    T=min(max(T,2),15);
    sp=Generate_SubProblems(nobj,N,T);
    % Empty Individual
    empty_individual.Position=[];
    empty_individual.Cost=[];
    empty_individual.g=[];
    empty_individual.IsDominated=[];
    % Initialize Goal Point
    z=zeros(nobj,1);
    % Create Initial Population
    pop=repmat(empty_individual,N,1);
    newbonobo=repmat(empty_individual,1,1);
    for i=1:N
        pop(i).Position=unifrnd(Var_min,Var_max,[1 d]);
        pop(i).Cost=fobj(pop(i).Position');
        z=min(z,pop(i).Cost);
    end
    for i=1:N
        pop(i).g=Cost_Decomposition(pop(i),z,sp(i).lambda);
    end
    % Determine Population Domination Status
    pop=Determine_Domonation(pop);
    % Initialize Estimated Pareto Front
    EP=pop(~[pop.IsDominated]);
    [~,k]=min([EP.g]);
    alphabonobo=EP(k);
    it=1;
    nem=zeros(MaxIt+1,1);
    tempdv=zeros(MaxIt+1,1);
    nem(it)=min(nArchive,size(EP,1));
    rep_costs=[EP.Cost]';
    ss=minmax(rep_costs');
    tempdv(it)=mean(std(rep_costs)./(ss(:,2)-ss(:,1))');
    for it=1:MaxIt
        tsgs_max=max(2,ceil(N*tsgs_factor));  % Maximum size of the temporary sub-group
        for i=1:N
            B = 1:N;
            B(i)=[];
            %% Determining the actual size of the temporary sub-group
            tsg=randi([2 tsgs_max]);
            %% Selection of pth Bonobo using fission-fusion social strategy & flag value determination
            q=randsample(B,tsg);
            temp_cost=pop(q).g;
            [~,ID1]=min(temp_cost);
            p=q(ID1);
            %% Creation of newbonobo
            if(rand<=p_p)
                r1=rand(1,d); %% Promiscuous or restrictive mating strategy
                newbonobo.Position=pop(i).Position+scab*r1.*(alphabonobo.Position-pop(i).Position+scsb*(1-r1).*(pop(i).Position-pop(p).Position));
            else
                for j=1:d
                    if(rand<=p_xgm)
                        rand_var=rand; %% Extra group mating strategy
                        if(alphabonobo.Position(j)>=pop(i).Position(j))
                            if(rand<=(p_d))
                                beta1=exp(((rand_var)^2)+rand_var-(2/rand_var));
                                newbonobo.Position(j)=pop(i).Position(j)+beta1*(Var_max(j)-pop(i).Position(j));
                            else
                                beta2=exp((-((rand_var)^2))+(2*rand_var)-(2/rand_var));
                                newbonobo.Position(j)=pop(i).Position(j)-beta2*(pop(i).Position(j)-Var_min(j));
                            end
                        else
                            if(rand<=(p_d))
                                beta1=exp(((rand_var)^2)+(rand_var)-2/rand_var);
                                newbonobo.Position(j)=pop(i).Position(j)-beta1*(pop(i).Position(j)-Var_min(j));
                            else
                                beta2=exp((-((rand_var)^2))+(2*rand_var)-2/rand_var);
                                newbonobo.Position(j)=pop(i).Position(j)+beta2*(Var_max(j)-pop(i).Position(j));
                            end
                        end
                    else
                        if(rand<=p_d) %% Consortship mating strategy
                            newbonobo.Position(j)=pop(i).Position(j)+(exp(-rand))*(pop(i).Position(j)-pop(p).Position(j));
                        else
                            newbonobo.Position(j)=pop(p).Position(j);
                        end
                    end
                end
            end
            %% Clipping
            for j=1:d
                if(newbonobo.Position(j)>Var_max(j))
                    newbonobo.Position(j)=Var_max(j);
                end
                if(newbonobo.Position(j)<Var_min(j))
                    newbonobo.Position(j)=Var_min(j);
                end
            end
            newbonobo.Cost= fobj(newbonobo.Position'); % New cost evaluation
            z=min(z,newbonobo.Cost);
            for j=sp(i).Neighbors
                newbonobo.g=Cost_Decomposition(newbonobo,z,sp(j).lambda);
            end
            %% New bonobo acceptance criteria
            for j=sp(i).Neighbors
                newbonobo.g=Cost_Decomposition(newbonobo,z,sp(j).lambda);
                if((newbonobo.g<=pop(j).g)||(rand<=(p_xgm)))
                    pop(j)=newbonobo;
                end
            end
            if(newbonobo.g<alphabonobo.g)
                alphabonobo=newbonobo;
            end
        end
        pop=Determine_Domonation(pop);
        ndpop=pop(~[pop.IsDominated]);
        EP=[EP
            ndpop]; %#ok
        EP=Determine_Domonation(EP);
        EP=EP(~[EP.IsDominated]);
        temprep=[EP.Cost]';
        [~,ind]=unique(temprep,'rows');
        EP=EP(ind);
        if numel(EP)>nArchive
            Extra=numel(EP)-nArchive;
            ToBeDeleted=randsample(numel(EP),Extra);
            EP(ToBeDeleted)=[];
        end
        nem(it+1)=numel(EP);
        if(nem(it+1)>1)
            rep_costs=[EP.Cost]';
            ss=minmax(rep_costs');
            tempdv(it+1)=mean(std(rep_costs)./(ss(:,2)-ss(:,1))');
            if(nem(it+1)>=nem(it) && tempdv(it+1)>tempdv(it))
                pp=1;
            else
                pp=0;
            end
        else
            pp=0;
        end
        %% Parameters updation
        if(pp==1)
            npc=0; %% Positive phase
            ppc=ppc+1;
            cp=min(0.5,(ppc*rcpp));
            pbest=alphabonobo;
            p_xgm=p_xgm_initial;
            p_p=0.5+cp;
            tsgs_factor=min(tsgs_factor_max,(tsgs_factor_initial+ppc*(rcpp^2)));
        else
            npc=npc+1; %% Negative phase
            ppc=0;
            cp=-(min(0.5,(npc*rcpp)));
            p_xgm=min(0.5,p_xgm_initial+npc*(rcpp^2));
            tsgs_factor=max(0,(tsgs_factor_initial-npc*(rcpp^2)));
            p_p=0.5+cp;
        end
        % Display Iteration Information
        disp(['Iteration = ' num2str(it)]);
    end
end
% Pl% Plot of the results
PlotObjectiveFunction(pop,EP,nobj);
toc;