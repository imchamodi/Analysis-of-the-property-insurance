# Analysis-of-the-property-insurance
Analysis of the property insurance. (Identifying the features that fraudulent insurance clams.
clc; clear;
data = readtable('E:\3rd year 2nd semester\IS 3053 - Data Mining Techniques\group project\insurence_claims.csv');
%%%%%%%%%%%%%%%%%%%%%---- Data Preprocessing ----%%%%%%%%%%%%%%%%%%%%%%

% Dropping variables
data(:, {'claimid', 'incident_date', 'policyid','job_start_date','occupancy_date'}) = [];

% Creating new variable person_age
dob_year = year(datetime(data.dob, 'InputFormat', 'MM/dd/yyyy'));
current_year = 2008;
person_age = current_year - dob_year;
data.person_age = person_age;

% Creating new variable policy_duration
policy_year = year(datetime(data.policy_date, 'InputFormat', 'MM/dd/yyyy'));
current_year = 2008;
policy_duration = current_year - policy_year;
data.policy_duration = policy_duration;

% Dropping variables Again!
data(:, {'policy_date', 'dob'}) = [];

% One-hot encode the claim_type variable

claim_type_onehot = dummyvar(data.claim_type);
data = [data array2table(claim_type_onehot)];
data.Properties.VariableNames(end-4:end) = {'Wind_Hail', 'Water_damage', 'Fire_Smoke', 'Contamination', 'Theft_Vandalism'};
data(:, 'claim_type') = [];

%converting claim type to nominal categorical variable
%class(data.claim_type);
%claim_type_labels = {'Wind/Hail', 'Water damage', 'Fire/Smoke', 'Contamination', 'Theft/Vandalism'};
%data.claim_type = categorical(data.claim_type, 1:5, claim_type_labels);

%%%%%%%%%%%%%%%%%%%%%---- Q(ii) ----%%%%%%%%%%%%%%%%%%%%%%
% Create a new data set which includes records with all fraudulence claims and 500 randomly
% selected non-fraudulence claims.


% Separate the fraudulent claims and non-fraudulent claims
fraudulent_claims = data(data.fraudulent == 1, :); %total 463 fraud claims 
non_fraudulent_claims = data(data.fraudulent == 0, :);

% Randomly select 500 non-fraudulent claims
rng(10) % set seed for reproducibility
idx_non_fraudulent = randperm(height(non_fraudulent_claims), 500);
%P = randperm(N,K) returns a row vector containing K unique integers
%selected randomly from 1:N.  For example, randperm(6,3) might be [4 2 5].
%here height(non_fraudulent_claims)is no.of non-fr claim. from that select
%500
non_fraudulent_claims = non_fraudulent_claims(idx_non_fraudulent, :);

% Combine the fraudulent claims and randomly selected non-fraudulent claims
new_data = [fraudulent_claims; non_fraudulent_claims];

% Create a new table called 'predictors' containing all predictor variables
responseCol = strcmp(new_data.Properties.VariableNames, 'fraudulent');
predictors = new_data(:, ~responseCol);

% Convert the predictor table to a matrix
X = table2array(predictors);
%X = double(X);

Y = new_data.fraudulent;
%%%%%%%%%%%%%%%---- Splitting the dataset ----%%%%%%%%%%%%%%%%

% Split data into training, testing, and validation sets
cvp = cvpartition(size(X,1),'Holdout',0.2);  % 20% for testing
X_train = X(cvp.training,:);
Y_train = Y(cvp.training,:);
X_test = X(cvp.test,:);
Y_test = Y(cvp.test,:);

% Further divide training set into training and validation sets
cvp2 = cvpartition(size(X_train,1),'Holdout',0.2);  % 20% for validation
X_train_final = X_train(cvp2.training,:);
Y_train_final = Y_train(cvp2.training,:);
X_val = X_train(cvp2.test,:);
Y_val = Y_train(cvp2.test,:);



%%%%%%%%%%%%%%%%%%%%%---- C Tree ----%%%%%%%%%%%%%%%%%%%%%%


%c_tree = fitctree(X_train,Y_train);
c_tree = fitctree(X_train,Y_train,'PredictorNames',{'uninhabitable' ,'claim_amount' ,'coverage','deductible','townsize' ,'gender','edcat','retire', 'income' , 'marital' ,'reside' ,'primary_residence','person_age','policy_duration','Wind_Hail', 'Water_damage', 'Fire_Smoke', 'Contamination', 'Theft_Vandalism'},'Prune','on');
view(c_tree,'Mode','graph');


% Get the variable importance values from the trained tree
imp = c_tree.predictorImportance;

% Get the variable names from the predictor table
varNames = ({'uninhabitable' ,'claim_amount' ,'coverage','deductible','townsize' ,'gender','edcat','retire', 'income' , 'marital' ,'reside' ,'primary_residence','person_age','policy_duration','Wind_Hail', 'Water_damage', 'Fire_Smoke', 'Contamination', 'Theft_Vandalism'});

figure;
bar(imp);
title('Predictor Importance Estimates');
ylabel('Estimates');
xlabel('Predictors');
h = gca;
h.XTickLabel = varNames;
h.XTickLabelRotation = 90;

%Examining Resubstitution Error
resuberror = resubLoss(c_tree)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%holdout validation 
Ypred = predict(c_tree,X_test);

% Calculate accuracy
accuracy = sum(Ypred == Y_test) / numel(Y_test);
disp(['Accuracy full tree: ' num2str(accuracy)])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%https://in.mathworks.com/help/stats/improving-classification-trees-and-regression-trees.html
%Cross Validation - To get a better sense of the predictive accuracy of your tree for new data, cross validate the tree. 
cv_c_tree = crossval(c_tree);
cvloss = kfoldLoss(cv_c_tree)

%Finding optimal tree

leafs = logspace(1,3,50);
N = numel(leafs);
err = zeros(N,1);
for n=1:N
    t = fitctree(X_train,Y_train,'CrossVal','On',...
        'MinLeafSize',leafs(n));
    err(n) = kfoldLoss(t);
end
plot(leafs,err);
xlabel('Min Leaf Size');
ylabel('cross-validated error');

[minErr, minIdx] = min(err);
optimalMinLeafSize = leafs(minIdx); %49.5 is the The best leaf size 

OptimalTree = fitctree(X_train,Y_train,'MinLeafSize',49.5,'PredictorNames',{'uninhabitable' ,'claim_amount' ,'coverage','deductible','townsize' ,'gender','edcat','retire', 'income' , 'marital' ,'reside' ,'primary_residence','person_age','policy_duration','Wind_Hail', 'Water_damage', 'Fire_Smoke', 'Contamination', 'Theft_Vandalism'});
view(OptimalTree,'mode','graph')

resubOpt = resubLoss(OptimalTree);
lossOpt = kfoldLoss(crossval(OptimalTree));
resubDefault = resubLoss(c_tree);
lossDefault = kfoldLoss(crossval(c_tree));
resubOpt,resubDefault,lossOpt,lossDefault


%%%%%%%%%%%%%%%%%%%%%%%%
%holdout validation 
Ypred = predict(OptimalTree,X_test);

% Calculate accuracy
accuracy = sum(Ypred == Y_test) / numel(Y_test);
disp(['Accuracy optimal: ' num2str(accuracy)])

%%%%%%%%%%%%%%%%%%%%%%%%%%%5


% Get the variable importance values from the trained tree
imp_optimal = OptimalTree.predictorImportance;

% Get the variable names from the predictor table
varNames = ({'uninhabitable' ,'claim_amount' ,'coverage','deductible','townsize' ,'gender','edcat','retire', 'income' , 'marital' ,'reside' ,'primary_residence','person_age','policy_duration','Wind_Hail', 'Water_damage', 'Fire_Smoke', 'Contamination', 'Theft_Vandalism'});

figure;
bar(imp_optimal);
title('Predictor Importance Estimates');
ylabel('Estimates');
xlabel('Predictors');
