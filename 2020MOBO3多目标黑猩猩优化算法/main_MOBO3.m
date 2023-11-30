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

clear all;
close all;
clc;

%%%% 设置实验参数范围，包括实验次数和实验函数范围
Num_Test=5;   %%%% 每个函数独立进行Num_Test轮?
Num_Experiment=30;   %%%% 函数是从F1-FNum_Functions
AlgorithmName='MOBO3'; %%% 控制函数名?


% gmax = 100;    %最大迭代次数
% % FEE = 5000; %%%最大目标函数评价次数
% n = 50;       %种群规模
max_iter=200;  % Maximum Number of Iterations
N=200;    % Population Size (Number of Sub-Problems)
ArchiveMaxSize=100;  %%%Archive Size(number of rep)

m = 5;   %目标维数


ALLFunction_AllTest=[];

% for ff=[1:21];
% for ff=[10:21];
% for ff=[1:4,6:9];
for ff=[7];
    clearvars -except Num_Test Num_Experiment AlgorithmName ALLFunction_AllTest ff max_iter N ArchiveMaxSize m AllTest_Results  problem_name 
    %%%%% 创建文件夹?
    string_0ALL=['000\',AlgorithmName,'_5维目标800次迭代100种群实验20210923\'];
    dirname00=[string_0ALL,'\F',num2str(ff),'\'];
    display(['**********  ',AlgorithmName,'算法优化F',num2str(ff),'的 ', 'M',num2str(m), ' 维实验   **********']);
   for testi=1:Num_Test   %%%% 控制每次实验测试次数
       dirname0=[dirname00,'test',num2str(testi),'_F',num2str(ff)];
       system(['mkdir ' dirname0]) %创建主文件夹
       dirname1=[dirname0,'\F',num2str(ff),'_fig'];
       system(['mkdir ' dirname1]) %创建文件夹  等待保存实验图像
       dirname2=[dirname0,'\F',num2str(ff),'_data'];
       system(['mkdir ' dirname2]) %创建文件夹  等待保存实验图像
       for kk=1:30 %%%% 控制实验次数的循环
           display(['**********  ',AlgorithmName,'算法优化F',num2str(ff),'的  第  ', num2str(kk), ' 次实验   **********']);
            rand('state',sum(100*clock));
            problem_name=['F',num2str(ff)];
   
           [ Var_max,Var_min,d ] = generate_boundary1( problem_name,m );%Upper and Lower Bound of Decision Variables  %%% 生成决策空间中变量上界、下界和维度
tic;  % CPU time measure
%     m=2;
% if nobj==2
%     fobj=@(x)kursave(x);% Objective function
%     d=3;  % No. of Variables
%     Var_min=-5*ones(1,d);  % Lower variable Boundaries
%     Var_max=5*ones(1,d);   % Upper variable Boundaries
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
%     MaxIt=200;  % Maximum Number of Iterations
%     N=200;    % Population Size (Number of Sub-Problems)
%     nArchive=100;
    T=max(ceil(0.15*N),2);    % Number of Neighbors
    T=min(max(T,2),15);
    sp=Generate_SubProblems(m,N,T);
    % Empty Individual
    empty_individual.Position=[];
    empty_individual.Cost=[];
    empty_individual.g=[];
    empty_individual.IsDominated=[];
    % Initialize Goal Point
    z=zeros(m,1);
    % Create Initial Population
    pop=repmat(empty_individual,N,1);
    newbonobo=repmat(empty_individual,1,1);
    for i=1:N
        pop(i).Position=unifrnd(Var_min,Var_max,[1 d]);
%         pop(i).Cost=fobj(pop(i).Position');
        pop(i).Cost=test_function(pop(i).Position,d,m,problem_name)';
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
%             newbonobo.Cost= fobj(newbonobo.Position'); % New cost evaluation
            newbonobo.Cost=test_function(newbonobo.Position,d,m,problem_name)'; % New cost evaluation
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
%         disp(['Iteration = ' num2str(it)]);
        HisPF{it} = rep_costs;
    end
% end
% % Pl% Plot of the results
% PlotObjectiveFunction(pop,EP,m);

            time=toc;
            PF = rep_costs;
            cg_curve=HisPF; %%% 历史目标函数值
            Time(kk)=time;
            cc=strcat(dirname2,'\',AlgorithmName,'优化次数_',num2str(kk),'.mat');
            result.time=time;
           
            true_PF=TPF(m,ArchiveMaxSize, problem_name);

         %%% using the matlab codes for calculating metric values
         

            hv = HV(PF,true_PF);   %超体积?
            gd = GD(PF, true_PF);                       %世代距离

            sp = Spacing(PF, true_PF);                  %空间分布 
            igd = IGD(PF, true_PF);            %反向世代距离

            hvd(kk)=hv;
            gdd(kk)=gd;
            ssp(kk)=sp;
            igdd(kk)=igd;

             save(cc)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       end
       mean_IGD = mean(igdd);
       display([AlgorithmName,'_Functions_F',num2str(ff),'测试',num2str(kk),'次实验的IGD平均值 : ', num2str(mean_IGD)]);
       std_IGD=std(igdd);
       display([AlgorithmName,'_Functions_F',num2str(ff),'测试',num2str(kk),'次实验的IGD标准差 : ', num2str(std_IGD)]);
       max_IGD=max(igdd);
       display([AlgorithmName,'_Functions_F',num2str(ff),'测试',num2str(kk),'次实验的IGD最大值 : ', num2str(max_IGD)]);
       min_IGD=min(igdd);
       display([AlgorithmName,'_Functions_F',num2str(ff),'测试',num2str(kk),'次实验的IGD最小值 : ', num2str(min_IGD)]);
       display('******************************** ');
       
       
       mean_GD = mean(gdd);
       display([AlgorithmName,'_Functions_F',num2str(ff),'测试',num2str(kk),'次实验的GD平均值 : ', num2str(mean_GD)]);
       std_GD=std(gdd);
       display([AlgorithmName,'_Functions_F',num2str(ff),'测试',num2str(kk),'次实验的GD标准差 : ', num2str(std_GD)]);
       max_GD=max(gdd);
       display([AlgorithmName,'_Functions_F',num2str(ff),'测试',num2str(kk),'次实验的GD最大值 : ', num2str(max_GD)]);
       min_GD=min(gdd);
       display([AlgorithmName,'_Functions_F',num2str(ff),'测试',num2str(kk),'次实验的GD最小值 : ', num2str(min_GD)]);
       display('******************************** ');
      
       
       mean_HV = mean(hvd);
       display([AlgorithmName,'_Functions_F',num2str(ff),'测试',num2str(kk),'次实验的HV平均值 : ', num2str(mean_HV)]);
       std_HV=std(hvd);
       display([AlgorithmName,'_Functions_F',num2str(ff),'测试',num2str(kk),'次实验的HV标准差 : ', num2str(std_HV)]);
       max_HV=max(hvd);
       display([AlgorithmName,'_Functions_F',num2str(ff),'测试',num2str(kk),'次实验的HV最大值 : ', num2str(max_HV)]);
       min_HV=min(hvd);
       display([AlgorithmName,'_Functions_F',num2str(ff),'测试',num2str(kk),'次实验的HV最小值 : ', num2str(min_HV)]);
       display('******************************** ');
       
       mean_SP = mean(ssp);
       display([AlgorithmName,'_Functions_F',num2str(ff),'测试',num2str(kk),'次实验的SP平均值 : ', num2str(mean_SP)]);
       std_SP=std(ssp);
       display([AlgorithmName,'_Functions_F',num2str(ff),'测试',num2str(kk),'次实验的SP标准差 : ', num2str(std_SP)]);
       max_SP=max(ssp);
       display([AlgorithmName,'_Functions_F',num2str(ff),'测试',num2str(kk),'次实验的SP最大值 : ', num2str(max_SP)]);
       min_SP=min(ssp);
       display([AlgorithmName,'_Functions_F',num2str(ff),'测试',num2str(kk),'次实验的SP最小值 : ', num2str(min_SP)]);
       display('******************************** ');
       
       mean_time=mean(Time);
       display([AlgorithmName,'_Functions_F',num2str(ff),'测试',num2str(kk),'次实验的运行时间平均值 : ', num2str(mean_time)]);
       std_time=std(Time);
       display([AlgorithmName,'_Functions_F',num2str(ff),'测试',num2str(kk),'次实验的运行时间标准差 : ', num2str(std_time)]);
       display('******************************** ');
%        mean_X=mean(Best_X);
%         display([AlgorithmName,'_Functions_F',num2str(ff),'测试',num2str(kk),'次实验的最优解平均值 : ', num2str(mean_X)]);
%         std_X=std(Best_X);
%         display([AlgorithmName,'_Functions_F',num2str(ff),'测试',num2str(kk),'次实验的最优解标准差 : ', num2str(std_X)]);
%         %%%%%%%%%%%%%%%%%%
        cd=strcat(dirname0,'\Result汇总结果.mat');
        Result.IGDmean=mean_IGD;
        Result.IGDstd=std_IGD;
        Result.IGDmax=max_IGD;
        Result.IGDmin=min_IGD;
      
        
        Result.GDmean=mean_GD;
        Result.GDstd=std_GD;
        Result.GDmax=max_GD;
        Result.GDmin=min_GD;
        
        Result.HVmean=mean_HV;
        Result.HVstd=std_HV;
        Result.HVmax=max_HV;
        Result.HVmin=min_HV;
        
        Result.SPmean=mean_SP;
        Result.SPstd=std_SP;
        Result.SPmax=max_SP;
        Result.SPmin=min_SP;
        
        Result.Tmean=mean_time;
        Result.Tstd=std_time;
        %         Result.Xmean=mean_X;
        %         Result.Xstd=std_X;
        %         Result.Best_Y=Best_Y;
        %         Result.Best_X=Best_X;
        Result.Time=Time;
        %         Result.ResultVector=[mean_IGD,std_IGD,max_IGD,min_IGD,mean_GD,std_GD,max_GD,min_GD,mean_time,std_time];
        Result.ResultVector=[mean_IGD,std_IGD,max_IGD,min_IGD,mean_GD,std_GD,max_GD,min_GD,mean_HV,std_HV,max_HV,min_HV,mean_SP,std_SP,max_SP,min_SP,mean_time,std_time];
        %         Result.Best_History_Y=History_Y;
        save(cd,'Result')
        
        
        %         AllTest_Results(testi,:)=[mean_IGD,std_IGD,max_IGD,min_IGD,mean_time,std_time];
        % AllTest_Results(testi,:)=[mean_IGD,std_IGD,max_IGD,min_IGD,mean_GD,std_GD,max_GD,min_GD,mean_time,std_time];
        AllTest_Results(testi,:)=[mean_IGD,std_IGD,mean_GD,std_GD,mean_HV,std_HV,mean_SP,std_SP,mean_time,std_time];
    end
    cd=strcat(dirname00,'Result_AllTest.mat');
    save(cd,'AllTest_Results')
    ALLFunction_AllTest=[ALLFunction_AllTest;AllTest_Results];
end


cd=strcat(string_0ALL,'ALLFunction_AllTest.mat');
save(cd,'ALLFunction_AllTest')