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

%claim_type_onehot = dummyvar(data.claim_type);
%data = [data array2table(claim_type_onehot)];
%data.Properties.VariableNames(end-4:end) = {'Wind_Hail', 'Water_damage', 'Fire_Smoke', 'Contamination', 'Theft_Vandalism'};
%data(:, 'claim_type') = [];

%converting claim type to nominal categorical variable
class(data.claim_type);
claim_type_labels = {'Wind/Hail', 'Water damage', 'Fire/Smoke', 'Contamination', 'Theft/Vandalism'};
data.claim_type = categorical(data.claim_type, 1:5, claim_type_labels);

%Boxplots
%Coverage vs fraudulent
boxplot(data.coverage,data.fraudulent)
title('Boxplot of Coverage with Fraudulent')
xlabel('Fraudulent')
ylabel('Coverage')
%Claim amount vs fraudulent
boxplot(data.claim_amount,data.fraudulent)
title('Boxplot of Claim amount with Fraudulent')
xlabel('Fraudulent')
ylabel('Claim amount')
%Deductible vs fraudulent
boxplot(data.deductible,data.fraudulent)
title('Boxplot of Deductible with Fraudulent')
xlabel('Fraudulent')
ylabel('Diductible')
%Income vs fraudulent
boxplot(data.income,data.fraudulent)
title('Boxplot of Income with Fraudulent')
xlabel('Fraudulent')
ylabel('Income')
%Reside vs fraudulent
boxplot(data.reside,data.fraudulent)
title('Boxplot of Reside with Fraudulent')
xlabel('Fraudulent')
ylabel('Reside')
%Person Age vs fraudulent
boxplot(data.person_age,data.fraudulent)
title('Boxplot of Age with Fraudulent')
xlabel('Fraudulent')
ylabel('Age')

%Bar Chart

ytown=countcats(categorical(data.townsize))
bar(ytown)
title('Bar graph of Town size');
xlabel=('Town size');
ylabel=('Frequency');

%pie(data.fraudulent)


%Histograms for numerical variables
histogram(data.deductible)
histogram(data.person_age)
histogram(data.claim_amount)
histogram(data.income)
histogram(data.reside)
histogram(data.coverage)
histogram(data.policy_duration)
title('Histogram of Policy Duration')

%Response Variable
y=countcats(categorical(data.fraudulent))
bar(y,'green')

x=countcats(categorical(data.claim_type))
bar(x,'green')

nonfraud=data(data.fraudulent==0,:)
fraud=data(data.fraudulent==1,:)


normplot(nonfraud.person_age)


%Spearman's Rank Correlation
%data1=data(:,[3 5 6 11 15 16 13])
%[RHO,PVAL] = corr(a',b','Type','Spearman');

y1=countcats(categorical(data.fraudulent(data.claim_type=='Wind/Hail')))
y12=[y1(1)/1054 y1(2)/1054];
[y2]=countcats(categorical(data.fraudulent(data.claim_type=='Water damage')))
y22=[y2(1)/627 y2(2)/627]
[y3]=countcats(categorical(data.fraudulent(data.claim_type=='Fire/Smoke')))
y32=[y3(1)/1039 y3(2)/1039]
[y4]=countcats(categorical(data.fraudulent(data.claim_type=='Contamination')))
y42=[y4(1)/404 y4(2)/404]
[y5]=countcats(categorical(data.fraudulent(data.claim_type=='Theft/Vandalism')))
y52=[y5(1)/1291 y5(2)/1291]

Y=[y12;y22;y32;y42;y52]
x=data.claim_type;
bar(Y,'stacked')
title('Stacked bar graph of Fraudulent with Claim type')
set(gca, 'xticklabel', {'Wind/Hail', 'Water damage', 'Fire/Smoke','Contamination','Theft/Vandalism'});
legend('Non fraudulent','fraudulent')
%x=countcats(categorical(data.claim_type))

A=countcats(categorical(data.townsize))


[a1]=countcats(categorical(data.fraudulent(data.townsize==1)))
a12=[a1(1)/1133 a1(2)/1133];
[a2]=countcats(categorical(data.fraudulent(data.townsize==2)))
a22=[a2(1)/1048 a2(2)/1048]
[a3]=countcats(categorical(data.fraudulent(data.townsize==3)))
a32=[a3(1)/887 a3(2)/887]
[a4]=countcats(categorical(data.fraudulent(data.townsize==4)))
a42=[a4(1)/810 a4(2)/810]
[a5]=countcats(categorical(data.fraudulent(data.townsize==5)))
a52=[a5(1)/537 a5(2)/537]

aa=[a12;a22;a32;a42;a52]
x=data.claim_type;
bar(aa)
%bar(aa,'stacked')
title('Stacked bar graph of Fraudulent with Town size')
set(gca, 'xticklabel', {'>250,000', '50,000-249,999', '10,000-49,999','2,500-9,999','<2,500'});
legend('Non fraudulent','fraudulent')

nonfraud=data(data.fraudulent==0,:)
fraud=data(data.fraudulent==1,:)

normplot(nonfraud.income)
title('Normal probability plot of Income')
%not normal

normplot(nonfraud.claim_amount)
title('Normal probability plot of Claim amount')
%not normal

normplot(nonfraud.coverage)
title('Normal probability plot of Coverage')
%not normal

normplot(fraud.income)
title('Normal probability plot of Income')
%not normal

normplot(fraud.claim_amount)
title('Normal probability plot of Claim amount')
%not normal

normplot(fraud.coverage)
title('Normal probability plot of Coverage')
%not normal

% Perform Kruskal-Wallis test for fraudulent and income
[p, tbl, stats] = kruskalwallis([data.fraudulent data.income], [], 'off');

% Print results
fprintf('Kruskal-Wallis Test Results:\n');
fprintf('---------------------------\n');
fprintf('p-value: %.4f\n', p);
fprintf('---------------------------\n');
% medians are different

% Perform Kruskal-Wallis test for fraudulent and age
[p, tbl, stats] = kruskalwallis([data.fraudulent data.person_age], [], 'off');

% Print results
fprintf('Kruskal-Wallis Test Results:\n');
fprintf('---------------------------\n');
fprintf('p-value: %.4f\n', p);
fprintf('---------------------------\n');
% medians are different

% Perform Kruskal-Wallis test for fraudulent and claim amount
[p, tbl, stats] = kruskalwallis([data.fraudulent data.claim_amount], [], 'off');

% Print results
fprintf('Kruskal-Wallis Test Results:\n');
fprintf('---------------------------\n');
fprintf('p-value: %.4f\n', p);
fprintf('---------------------------\n');
% medians are different

x1=data.fraudulent
x2=data.claim_type

xN = grp2idx(x1);
yN = grp2idx(x2);

tab=crosstab(xN,yN)
%
%Scatterplots
plot(data.deductible,data.reside,'*')
title('Deductible Vs Reside')
%xlabel('Deductible')
%ylabel('Reside')

plot(data.deductible,data.coverage,'*')
title('Deductible Vs Reside')
%xlabel('Deductible')
%ylabel('Reside')

plot(data.deductible,data.income,'*')
title('Deductible Vs Income')
%xlabel('Deductible')
%ylabel('Income')

plot(data.deductible,data.claim_amount,'*')
title('Deductible Vs Claim Amount')
%xlabel('Deductible')
%ylabel('Claim Amount')

plot(data.reside,data.coverage,'*')
title('Reside Vs Coverage')
%xlabel('Reside')
%ylabel('Coverage')

plot(data.reside,data.income,'*')
title('Reside Vs Income');
%xlabel('Reside');
%ylabel('Income');

plot(data.reside,data.claim_amount,'*')
title('Reside Vs Claim Amount')
%xlabel('Reside')
%ylabel('Claim Amount')

plot(data.income,data.claim_amount,'*')
title('Income Vs Claim Amount')
%xlabel('Deductible')
%ylabel('Claim Amount')
% calculate slope and intercept of abline
b = regress((data.claim_amount), [(data.income) ones(size((data.income)))]);
slope = b(1);
intercept = b(2);

% Add abline to plot
hold on
line(data.income, slope*data.income + intercept, 'Color', 'r', 'LineWidth', 2);
hold off

plot(data.income,data.coverage,'*')
title('Income Vs Coverage')
%xlabel('Income')
%ylabel('Coverage')

plot(data.coverage,data.claim_amount,'*')
title('Coverage Vs Claim Amount')
%xlabel('Coverage')
%ylabel('Claim Amount')
title('Income Vs Claim Amount')
%xlabel('Deductible')
%ylabel('Claim Amount')
% Calculate slope and intercept of abline
b = regress((data.claim_amount), [(data.coverage) ones(size((data.coverage)))]);
slope = b(1);
intercept = b(2);

% Add abline to plot
hold on
line(data.coverage, slope*data.coverage + intercept, 'Color', 'r', 'LineWidth', 2);
hold off



