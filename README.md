### Code to accompany the manuscript:
#### Steve Mooney, Daniel Westreich, and Alexander Keil. 13 Questions About Using Machine Learning in Causal Research (You Won’t Believe the Answer to Number 10!). Am J Epidemiol. 2021. [Link](https://doi.org/10.1093/aje/kwab047)

### File descriptions:

#### R/CARTpscore.py
- eAppendix #1: R code that uses CART (classification and regression trees) to compute propensity scores for an inverse probability weighted estimate the causal effect of Digitalis on death in the Digitalis Investigator Group Trial

#### python/CARTpscore.py
- eAppendix #2: Python code that uses CART (classification and regression trees) to compute propensity scores for an inverse probability weighted estimate the causal effect of Digitalis on death in the Digitalis Investigator Group Trial

#### sas/CARTpscore.sas
- eAppendix #3: SAS code that uses CART (classification and regression trees) to compute propensity scores for an inverse probability weighted estimate the causal effect of Digitalis on death in the Digitalis Investigator Group Trial

#### R/GLM_XGBOOSTtmle.R
- eAppendix #4: R code that uses TMLE (targeted minimum loss estimation) to apply two different machine learning algorithms to estimate the causal effect of Digitalis on death in the Digitalis Investigator Group Trial

#### julia/CARTpscore.jl
- Bonus: Julia code that uses CART (classification and regression trees) to compute propensity scores for an inverse probability weighted estimate the causal effect of Digitalis on death in the Digitalis Investigator Group Trial



#### data/digdata.csv
- CSV (comma separated text file) containing data from the Digitalis Investigator Group Trial, as found in the R package `asympTest` v0.1.4

