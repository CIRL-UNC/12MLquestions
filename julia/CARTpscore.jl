######################################################################################################################
# Author: Alex Keil
# Program: CARTpscore.jl
# Language: Julia 1.4.2
# Date: Monday, July 13, 2020
# Data in: digdata.csv
# Description: Bonus: Julia code that uses CART to compute propensity scores for 
#   an inverse probability weighted estimate the causal effect of Digitalis on death 
#   in the Digitalis Investigator Group Trial
# Released under the GNU General Public License: http://www.gnu.org/copyleft/gpl.html
######################################################################################################################

# Run once to install packages
# using Pkg
# Pkg.add(["DecisionTree", "Plots", "CSV", "DataFrames", "Plotly", "CovarianceMatrices", "GLM"])

using DecisionTree, StatsPlots, GLM, CovarianceMatrices, Statistics
import CSV, DataFrames # similar to "using" but you have to start the function calls with "CSV."
plotly() # use plotly backend for plotting (plotshows up in browser window)

DIGdata = DataFrames.DataFrame(CSV.File("data/digdata.csv", missingstring="NA"));
cc = DataFrames.dropmissing(DIGdata, :BMI)

# use ten-fold cross validation on the training dataset to pick parameters for a CART model.
y  = string.(cc.TRTMT); # must be string to use classification tree
X = Array(cc[:,[:AGE, :SEX, :BMI]]);

# Julia doesn't use cost-complexity pruning, so we'll optimize on the max tree depth
# will use CV 3 times
n_subfeatures=0; max_depth=-1; min_samples_leaf=1; min_samples_split=2
min_purity_increase=0.0; pruning_purity = 1.0

minTrees = 1
maxTrees = 20
CVaccuracy = [mean(mean(nfoldCV_tree(y,X, 
          10,  # number of folds
          pruning_purity, # pruning_purity (1 = no pruning)
          depth,   # max depth
          min_samples_leaf,
          min_samples_split,
          min_purity_increase
          )) 
          for repeats in 1:3) for depth in minTrees:maxTrees ];

# select best performing tree and fit it        
maxdepth = argmin(CVaccuracy)

model = build_tree(y,X,
          n_subfeatures, # number of subfeatures (0 = all)
          maxdepth,
          min_samples_leaf,
          min_samples_split,
          min_purity_increase
                   )

# Compute propensity scores as probability of treatment (propensity scores)
probs = apply_tree_proba(model, X, ["1", "0"]);
pscore = probs[:,1];


# plot the propensity scores (will just be peaks)
density(pscore[cc.TRTMT .== 1], label="Treated")
density!(pscore[cc.TRTMT .== 0], label="Unreated")

cc.weights = [x == 1 ? 1.0 / pscore[i] : 1.0 / (1.0 - pscore[i]) for (i,x) in enumerate(cc.TRTMT)];


# fit MSM
msmfit = glm(@formula(DEATH ~ TRTMT), cc, Bernoulli(), wts=cc.weights)
# get robust standard error estimates and construct confidence intervals
beta = coef(msmfit)
robse = stderror(msmfit, HC0())
ci = beta .+ [-1.96 1.96; -1.96 1.96] .* robse

# odds ratio, 95% robust CI
exp(beta[2]), exp.(ci[2,:])

# Estimtate outcomes if everyone were treated
cc0 = copy(cc)
cc0.TRTMT .= 0
pr0 = GLM.predict(msmfit, cc0)
cc1 = copy(cc)
cc1.TRTMT .= 1
pr1 = GLM.predict(msmfit, cc1)
# Estimate additive effect
mean(pr1) - mean(pr0)