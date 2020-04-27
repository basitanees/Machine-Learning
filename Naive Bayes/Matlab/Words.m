indices_neutral = find(y_tr == 'neutral');
indices_positive = find(y_tr == 'positive');
indices_negative = find(y_tr == 'negative');

neutral = x_tr(indices_neutral,:);
positive = x_tr(indices_positive,:);
negative = x_tr(indices_negative,:);

occ_neutral = sum(neutral);
occ_positive = sum(positive);
occ_negative = sum(negative);

[sort_neu, in_neu] = sort(occ_neutral);
[sort_pos, in_pos] = sort(occ_positive);
[sort_neg, in_neg] = sort(occ_negative);

words_neu = vocab1(in_neu);
words_neu = words_neu(5703:5722);
words_pos = vocab1(in_pos);
words_pos = words_pos(5703:5722);
words_neg = vocab1(in_neg);
words_neg = words_neg(5703:5722);
