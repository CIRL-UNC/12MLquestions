######################################################################################################################
# Author: Alex Keil
# Program: CARTpscore.py
# Language: python 3
# Date: Monday, July 13, 2020
# Data in: digdata.csv
# Description: eAppendix #2: Python code that uses CART to compute propensity scores for 
#   an inverse probability weighted estimate the causal effect of Digitalis on death 
#   in the Digitalis Investigator Group Trial
# Released under the GNU General Public License: http://www.gnu.org/copyleft/gpl.html
######################################################################################################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV
import statsmodels.api as sm

dat = pd.read_csv("../data/digdata.csv", na_values=['NA', '', '.'])

completecases = dat.dropna(axis=0, subset=['BMI'])

Z = np.array(completecases[['AGE', 'SEX', 'BMI']])
x = np.array(completecases.TRTMT)
y = np.array(completecases.DEATH)

# cross validtion to select tree parameters
# python doesn't use cost-complexity pruning, so we'll optimize on the max tree depth
# python fits will be saturated by default, so we'll use a non-default parameterization
tree = DecisionTreeClassifier(min_samples_split=20, min_samples_leaf=round(20/3), random_state=1231)
rkf = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1231)
gs = GridSearchCV(tree,
                  param_grid={'max_depth': range(1, 20)},
                  scoring='accuracy',
                  cv=rkf, return_train_score=True,
                  refit=True
                  )

ft = gs.fit(Z, x)
fit = ft.best_estimator_
treatment_prob = fit.predict_proba(Z)[:, 1]
control_prob = fit.predict_proba(Z)[:, 0]

tp1 = gaussian_kde(treatment_prob)
tp0 = gaussian_kde(control_prob)

# plotting propensity scores
xc = np.linspace(0, 1, 200)
plt.plot(xc, tp1(xc), label="Pr(Trt)")
plt.plot(xc, tp0(xc), label='1-Pr(Trt)')
plt.legend(loc=0)
plt.xlabel("Propensity score")
plt.ylabel("Density")
plt.show()


# MSM
ipw = x / treatment_prob + (1 - x) / (1 - treatment_prob)
D = np.column_stack((np.ones(len(x)), x))
geeid = np.array([i for i in range(len(x))])
fam = sm.families.Binomial(link=sm.genmod.families.links.logit())
cov = sm.cov_struct.Independence()
msm = sm.GEE(y, D, family=fam, weights=ipw, groups=geeid, cov_struct=cov)
msmfit = msm.fit()
print("Compute an outcome model")
print(msmfit.summary())

# Predict outcomes if everyone were treated/untreated
D1 = np.column_stack((np.ones(len(x)), np.ones(len(x))))
D0 = np.column_stack((np.ones(len(x)), np.zeros(len(x))))
predicted_outcome_all_treated = msmfit.predict(D1)
predicted_outcome_all_untreated = msmfit.predict(D0)

print("Predicted outcome, all treated")
print(np.mean(predicted_outcome_all_treated))
print("Predicted outcome, all untreated")
print(np.mean(predicted_outcome_all_untreated))

# Estimate additive effect
print("Additive treatment effect")
print(np.mean(predicted_outcome_all_treated -
              predicted_outcome_all_untreated)) 
