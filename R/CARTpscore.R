######################################################################################################################
# Author: Steve Mooney
# Program: CARTpscore.R
# Language: R
# Date: Monday, July 13, 2020
# Data in: DIGdata (asympTest package)
# Description:eAppendix #1: R code that uses CART to compute propensity scores for an 
#   inverse probability weighted estimate the causal effect of Digitalis on death in 
#   the Digitalis Investigator Group Trial
# Released under the GNU General Public License: http://www.gnu.org/copyleft/gpl.html
######################################################################################################################

#install.packages("caret")
#install.packages("asympTest")
# Main code
library(caret)
library(asympTest)
set.seed(12345)
data(DIGdata)

# In the default dataset, the TRTMT variable is coded as 0/1, which makes R think it is a continuous rather than a 
# dichotomous variable.  We recode it here
DIGdata$TRTMT_Dichotomous <- as.factor(DIGdata$TRTMT)
complete_case <- DIGdata[!is.na(DIGdata$BMI),]

# split data
training_dataset <- DIGdata[1:3400,]
test_dataset <- DIGdata[3401:6800,]

# use ten-fold cross validation on the training dataset to pick parameters for a CART model.
control <- trainControl(method="repeatedcv", number=10, repeats=3)
cart_model <- train(TRTMT_Dichotomous~AGE+SEX+BMI, data=complete_case, method="rpart", metric="Accuracy", trControl=control)
print(cart_model)

# Comput propensity scores as probability of treatment (propensity scores)
treatment_prob <- predict(cart_model, complete_case, type="prob")

# Not surprisingly for an RCT, treatment probabilities are decently balanced
plot(density(treatment_prob[,'1']))
lines(density(treatment_prob[,'0']), col='red')

# Use propensity scores for IPW analysis.
weights <- 1/treatment_prob
complete_case$IPW[complete_case$TRTMT_Dichotomous == 1] <- weights[complete_case$TRTMT_Dichotomous == 1, "1"]
complete_case$IPW[complete_case$TRTMT_Dichotomous == 0] <- weights[complete_case$TRTMT_Dichotomous == 0, "0"]
table(is.na(complete_case$IPW))

# Compute an outcome model
outcome_model <- glm(DEATH~TRTMT, data=complete_case, family="binomial", weights=IPW)
all_treated <- complete_case
all_treated$TRTMT <- 1
all_untreated <- complete_case
all_untreated$TRTMT <- 0

# Estimtate outcomes if everyone were treated
predicted_outcome_all_treated <- predict(outcome_model, all_treated, weights=IPW, type='response')
mean(predicted_outcome_all_treated)
predicted_outcome_all_untreated <- predict(outcome_model, all_untreated, weights=IPW, type='response')
mean(predicted_outcome_all_untreated)

# Estimate additive effect
mean(predicted_outcome_all_treated - predicted_outcome_all_untreated)
