cv:
extratrees,lr,mlp,rf,xgboost -> feature importance (avg), predictions (agg)
kan -> predictions (agg, kan_pred, kan_prod)
pysr,pypysr -> predictions (agg)

nocv:
kan -> formulas -> predictions (kansym_pred, kansym_prod), feature importance
pysr,pypysr -> formulas -> predictions, formula

best models on f1:
extratrees,lr,mlp,rf,xgboost
kan cv
kansym nocv
pysr,pypysr cv
pysr,pypysr nocv

complexity:
kansym nocv
pysr,pypysr nocv