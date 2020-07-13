######################################################################################################################
# Author: Alex Keil
# Program: CARTpscore.jl
# Language: Julia 1.4.2
# Date: Monday, July 13, 2020
# Data in: digdata.csv
# Description: Bonus: Julia code that uses CART to compute propensity scores for 
#   an inverse probability weighted estimate the causal effect of Digitalis on death 
#   in the Digitalis Investigator Group Trial
# Note: if you have not used Julia before, the first time loading a package calls
#  a compiler, which can take a significant amount of time (esp. with plots). The second time
#  running the code will go much more quickly.
# Released under the GNU General Public License: http://www.gnu.org/copyleft/gpl.html
######################################################################################################################

# Run once to install packages
# using Pkg
# Pkg.add(["DecisionTree", "Plots", "CSV", "DataFrames", "Plotly", "CovarianceMatrices", "GLM", "Suppressor"])

using DecisionTree, StatsPlots, GLM, CovarianceMatrices, Statistics, Suppressor
import CSV, DataFrames # similar to "using" but you have to start the function calls with "CSV."
plotly() # use plotly backend for plotting (plotshows up in browser window)

DIGdata = DataFrames.DataFrame(CSV.File("../data/digdata.csv", missingstring="NA"));
cc = DataFrames.dropmissing(DIGdata, :BMI)

# use ten-fold cross validation on the training dataset to pick parameters for a CART model.
y  = string.(cc.TRTMT); # must be string to use classification tree
X = Array(cc[:,[:AGE, :SEX, :BMI]]);

# Julia doesn't use cost-complexity pruning, so we'll optimize on the max tree depth
# will use CV 3 times

println("Cross-validated selection of tree depth")
function get_cvaccuracy(;minTrees = 1, maxTrees = 20, nrepeats=3, K=10)
  # convenience function to calculate 'K' fold cross validated accuracy averaged over 'nrepeats'
  # due to use of nfoldCV_tree function, this will print a lot of info to output
  n_subfeatures=0; 
  min_samples_leaf=1; 
  min_samples_split=2
  min_purity_increase=0.0; 
  pruning_purity = 1.0
  along = collect(minTrees:maxTrees)
  res = zeros(length(along))
  for repeat in 1:nrepeats
    for (j, depth) in enumerate(along)
      foldaccuracy = nfoldCV_tree(y,X, 
                  K,              # number of folds
                  pruning_purity, # pruning_purity (1 = no pruning)
                  depth,          # max depth
                  min_samples_leaf,
                  min_samples_split,
                  min_purity_increase
                )
      res[j] += inv(nrepeats) * mean(foldaccuracy)
    end
  end
  res
end


CVaccuracy = @suppress get_cvaccuracy(minTrees = 1, maxTrees = 20);

# select best performing tree and fit it        
maxdepth = argmin(CVaccuracy)

n_subfeatures=0; 
min_samples_leaf=1; 
min_samples_split=2
min_purity_increase=0.0; 

model = build_tree(y,X,
          n_subfeatures, # number of subfeatures (0 = all)
          maxdepth,
          min_samples_leaf,
          min_samples_split,
          min_purity_increase
        )

# Compute probability of treatment (propensity scores) and non-treatment
probs = apply_tree_proba(model, X, ["1", "0"]);
pscore = probs[:,1];


# plot the propensity scores (will just be peaks)
density(pscore[cc.TRTMT .== 1], label="Treated");
show(density!(pscore[cc.TRTMT .== 0], label="Unreated"))

cc.weights = [x == 1 ? 1.0 / pscore[i] : 1.0 / (1.0 - pscore[i]) for (i,x) in enumerate(cc.TRTMT)];


# fit MSM
msmfit = glm(@formula(DEATH ~ TRTMT), cc, Bernoulli(), wts=cc.weights)
# get robust standard error estimates and construct confidence intervals
beta = coef(msmfit)
robse = stderror(msmfit, HC0())
ci = beta .+ [-1.96 1.96; -1.96 1.96] .* robse

println("odds ratio, 95% robust CI")
println(exp(beta[2]), exp.(ci[2,:]))

# Estimtate outcomes if everyone were treated
cc0 = copy(cc);
cc0.TRTMT .= 0;
pr0 = GLM.predict(msmfit, cc0);
cc1 = copy(cc);
cc1.TRTMT .= 1;
pr1 = GLM.predict(msmfit, cc1);
# Estimate additive effect
println("Predicted outcome, all treated")
println(mean(pr1))
println("Predicted outcome, all untreated")
println(mean(pr0))
println("Additive treatment effect")
println(mean(pr1) - mean(pr0))