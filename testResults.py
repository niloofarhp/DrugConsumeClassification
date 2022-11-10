from scipy.stats import friedmanchisquare
import pandas as pd
import numpy as np
import scikit_posthocs as sp

f_test = friedmanchisquare([0.681,0.704,0.638,0.817],[0.725,0.71,0.667,0.848],[0.674,0.693,0.794,0.828]
,[0.72,0.708,0.674,0.821],[0.72,0.719,0.673,0.831],[0.726,0.725,0.691,0.858])
f_res = pd.DataFrame({'test':'Friedman','statistic':f_test[0],'pvalue':f_test[1]},index=[0])
print(f_res)

# the results:
#       test      statistic  pvalue
#       Friedman  11.690647  0.039282
#for a = 0.05 : 11.690647 > 10.290(k=6, n=4), 11.690647 > 10.490(k=6,n=5)
#So we can reject the null hypothesis

#Nemenyi CD calculation
# the coefficient value (q) for k = 6 and a = 0.5 is 2.84

data = np.array([[0.681,0.704,0.638,0.817],[0.725,0.71,0.667,0.848],[0.674,0.693,0.794,0.828]
,[0.72,0.708,0.674,0.821],[0.72,0.719,0.673,0.831],[0.726,0.725,0.691,0.858]])
 
# Conduct the Nemenyi post-hoc test
print(sp.posthoc_nemenyi_friedman(data.T))
# the result :
#           0         1         2         3         4         5
# 0  1.000000  0.410222  0.900000  0.799501  0.470014  *0.016584
# 1  0.410222  1.000000  0.900000  0.900000  0.900000  0.744925
# 2  0.900000  0.900000  1.000000  0.900000  0.900000  0.207292
# 3  0.799501  0.900000  0.900000  1.000000  0.900000  0.351731
# 4  0.470014  0.900000  0.900000  0.900000  1.000000  0.690349
# 5  0.016584  0.744925  0.207292  0.351731  0.690349  1.000000
# for a = 0.05 only the methods 0 and 5 (DT and GB) are significantly differnet(shown with *).