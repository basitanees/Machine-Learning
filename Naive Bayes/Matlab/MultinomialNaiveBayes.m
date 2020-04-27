% x_tr = table2array(x_tr);
% y_tr = table2array(y_tr);

indices_neutral = find(y_tr == 'neutral');
indices_positive = find(y_tr == 'positive');
indices_negative = find(y_tr == 'negative');
% kn = 1;
% kneg = 1;
% kp = 1;
% for i = 1:length(y_tr)
%     if y_tr(i) == 'neutral'
%        indices_neutral(kn) = i;
%        kn = kn + 1;
%     elseif y_tr(i) == 'positive'
%        indices_positive(kp) = i;
%        kp = kp + 1;
%     elseif y_tr(i) == 'negative'
%        indices_negative(kneg) = i;
%        kneg = kneg + 1;
%     end
% end

neutral = x_tr(indices_neutral,:);
positive = x_tr(indices_positive,:);
negative = x_tr(indices_negative,:);
%%
alpha = 1;
occ_neutral = sum(neutral) + alpha;
neutral_all = sum(occ_neutral) + (alpha * 5722);
theta_neutral = occ_neutral/neutral_all;

occ_positive = sum(positive)+ alpha;
positive_all = sum(occ_positive) + (alpha * 5722);
theta_positive = occ_positive/positive_all;

occ_negative = sum(negative)+ alpha;
negative_all = sum(occ_negative) + (alpha * 5722);
theta_negative = occ_negative/negative_all;

p_pos = length(indices_positive)/11712;
p_neg = length(indices_negative)/11712;
p_neu = length(indices_neutral)/11712;

priors = [p_neu p_pos p_neg];
likelihoods = [theta_neutral; theta_positive; theta_negative];
%%
labels = ["neutral"; "positive"; "negative"];
labels_pr = ["neutral"];
posteriors = zeros(1,3);
for i = 1:2928
    for k = 1:3
        total = 0;
        for a = 1:5722
            if ~((x_test(i,a) == 0) && (likelihoods(k,a) == 0))
               total = total + (x_test(i,a)*(log(likelihoods(k,a))));
            end
        end
        posteriors(k) = log(priors(k)) + total;
    end
    [maxi, index] = max(posteriors);
    labels_pr(i) = labels(index);
end

trues = (labels_pr == y_test');
percent_true = (sum(trues) / 2928)*100;