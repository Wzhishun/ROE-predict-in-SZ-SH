import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
new=pd.read_csv("final05.csv", index_col=0)
eval_true=new.iloc[:,0]
eval_pre=new.iloc[:,1]
test_true=new.iloc[:,2]
test_pre=new.iloc[:,3]
mse = mean_squared_error(eval_true, eval_pre)
r2 = r2_score(eval_true, eval_pre)
print(mse)
print(r2)
mse = mean_squared_error(test_true, test_pre)
r2 = r2_score(test_true, test_pre)
print(mse)
print(r2)