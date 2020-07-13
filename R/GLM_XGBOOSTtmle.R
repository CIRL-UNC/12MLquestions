######################################################################################################################
# Author: Steve Mooney
# Program: GLM_XGBOOSTtmle.R
# Language: R
# Date: Monday, July 13, 2020
# Data in: DIGdata (asympTest package)
# Description: eAppendix #4: R code that uses TMLE to apply two different machine learning 
#  algorithms to estimate the causal effect of Digitalis on death in the Digitalis 
#  Investigator Group Trial
# Released under the GNU General Public License: http://www.gnu.org/copyleft/gpl.html
######################################################################################################################

# Run once
install.packages("tmle")
install.packages("asympTest")

# Main code
# Load Libraries and dataset
library(tmle)
library(asympTest)
data(DIGdata)
set.seed(12345)


# Drop 1 observation with missing data
complete_case <- DIGdata[!is.na(DIGdata$BMI),]

# pull out the specific data to hand to TMLE
outcome <- complete_case$DEATH
exposure <- complete_case$TRTMT
covariates <- as.matrix(complete_case[,c("SEX", "AGE", "BMI")])

# Example of TMLE using GLM as the statistical model 
result_using_glm_only <- tmle(outcome,exposure,covariates, Q.SL.library = "SL.glm", g.SL.library = "SL.glm")
result_using_glm_only

# Example of TMLE using Extreme Gradient Boosting as the statistical model
result_using_xgboost <- tmle(outcome,exposure,covariates, Q.SL.library = "SL.xgboost", g.SL.library = "SL.xgboost")
result_using_xgboost

# Note slower model fit time and narrower confidence intervals in the xgb version
