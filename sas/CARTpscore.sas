/**********************************************************************************************************************
* Author: Alex Keil
* Program: CARTpscore.sas
* Language: SAS 9.4, SAS/STAT 14.2
* Date: Monday, July 13, 2020
* Data in: digdata.csv
* Description: eAppendix #3: SAS code that uses CART to compute propensity scores for an 
*   inverse probability weighted estimate the causal effect of Digitalis on death in the 
*   Digitalis Investigator Group Trial
* Released under the GNU General Public License: http://www.gnu.org/copyleft/gpl.html
**********************************************************************************************************************/


/* 
First, digdata.csv needs to be on your local computer - this will read from the github file structure
*/
%LET pathname = %SYSGET(SAS_EXECFILEPATH);
%LET n_remove = %LENGTH(%SCAN(&pathname, -3, \));
%LET pathname = %SUBSTR(&pathname, 1, %LENGTH(&pathname) - &n_remove)\data;

*or explicitly put in the pathname;
*%LET pathname = <Your pathname here>;
PROC IMPORT OUT=digdat FILE= "&pathname.\digdata.csv" REPLACE;
RUN;

/*
Use ten-fold cross validation on the training dataset to pick 
parameters for a CART model.
SAS note: while the sas procedure HPSPLIT has built in 
cross validated selection of cost-complexity parameters,
here we give an example done manually with a SAS macro.
*/

%MACRO cvtree(dat, minL=1, maxL=30, nfolds=10, reps=3);
  TITLE;
  DATA temp;SET &dat;
   CALL STREAMINIT(12321);
   __r = RAND('UNIFORM');
  %LET rep = 1;
  %DO %WHILE((&rep <= &reps));
    PROC SORT DATA = temp; BY __r;
    DATA temp;
      SET temp;
      CALL STREAMINIT(%EVAL(12321+&REP));
	  RETAIN fold&rep;
	  IF _n_ = 1 OR fold&rep=&nfolds THEN fold&rep=0;
      fold&rep = fold&rep+1;
      __r = RAND('UNIFORM');
    %LET rep = %EVAL(&rep + 1);
  %END;

  %MACRO tree(train=, test=, totleaves=);
    DATA train; SET &TRAIN;
	%IF "&test" ^= "" %THEN %DO;
	  DATA test; SET &test; _ot = trtmt;trtmt = .;
      DATA treetemp;
	    SET train test;
	%END;
	%IF "&test" = "" %THEN %DO;
      DATA treetemp;
	    SET train ;
		_ot = trtmt;
	%END;
    PROC HPSPLIT DATA = treetemp SEED=12321
      MAXDEPTH=30 MINCATSIZE=20 MINLEAFSIZE=7 CVMETHOD=NONE;
	  CLASS trtmt sex;
	  MODEL trtmt (EVENT='1') = age sex bmi;
	  GROW GINI;
	  PRUNE COSTCOMPLEXITY(LEAVES=&totleaves);
	  OUTPUT OUT = treetemp;
      ID trtmt _ot;
  %MEND;
  ODS SELECT NONE;
  %LET maxleaves = &minL;
  %DO %WHILE(%EVAL(&maxleaves <= &maxL));
    %LET rep = 1;
	%PUT HPSPLIT fit with COSTCOMPLEXITY (LEAVES=&maxleaves);
    %DO %WHILE((&rep <= &reps));
      %LET k = 1;
      %DO %WHILE((&k <= &nfolds));
        %TREE(train=temp(WHERE=(fold&rep ^= &k)), 
              test=temp(WHERE=(fold&rep = &k)), 
              totleaves=&maxleaves);
		DATA treetemp;
		  SET treetemp;
		  acc = _ot*(P_TRTMT1>0.5) + (1-_ot)*(P_TRTMT0>0.5);
		PROC MEANS DATA = treetemp NOPRINT;
		  VAR acc;
		  OUTPUT OUT = __accfold MEAN = ;
		DATA __accfold;
		  SET __accfold(KEEP=acc);
		  fold = &k;
		  rep = &rep;
		  maxleaves = &maxleaves;
		PROC APPEND DATA = __accfold BASE = accuracy; RUN;
        %LET k = %EVAL(&k + 1);
      %END;
      %LET rep = %EVAL(&rep + 1);
    %END;
    %LET maxleaves = %EVAL(&maxleaves + 1);
  %END;
  ODS SELECT ALL;
  PROC MEANS DATA = accuracy NOPRINT;
    CLASS maxleaves;
    VAR acc;
    OUTPUT OUT = cvaccuracy(WHERE=(maxleaves>.z)) MEAN = ;
  PROC SORT DATA = cvaccuracy; BY DESCENDING acc; RUN;
  DATA cvaccuracy;
   SET cvaccuracy;
   IF _N_=1 THEN CALL SYMPUT('cvleaves', PUT(maxleaves, 3.0));
  RUN;
  TITLE "Cross validation selected classification tree";
  %TREE(train=&DAT, test=, totleaves=&CVLEAVES);
  RUN;
%MEND;

OPTIONS NOMPRINT NONOTES;
/* run the macro here */
%CVTREE(dat=digdat, minL=1, maxL=10 , nfolds=10, reps=3);
OPTIONS NOMPRINT NOTES;

PROC PRINT DATA = cvaccuracy NOOBS;
  TITLE "Cross-validated accuracy";
  VAR maxleaves acc;

* Compute propensity scores as probability of treatment 
(propensity scores);

DATA merged;
 MERGE digdat treetemp(RENAME=(P_TRTMT1=pscore) KEEP=p_trtmt1);
 ipw = trtmt/pscore + (1-trtmt)/(1-pscore);

*Use propensity scores for IPW analysis.;
*create copies of data for getting predictions of outcomes;
DATA preddata;
 SET merged
     merged(in=ina)
     merged(in=inb);
 int = 0;
 IF ina THEN DO;
   int=1;
   death = .;
   trtmt = 1;
 END;
 ELSE IF inb THEN DO;
   int=1;
   death = .;
   trtmt = 0;
 END;
 id = _N_;

*compute an outcome model;
PROC GENMOD DATA = preddata;
  TITLE "MSM to get expected outcomes under treatment/no treatment";
  CLASS id;
  MODEL death = trtmt;
  WEIGHT ipw;
  /*Estimtate outcomes if everyone were treated/untreated*/
  OUTPUT OUT = predictions(WHERE=(int=1)) P=pdeath;
  REPEATED SUBJECT=id / type=ind;
*Estimate additive effect;
PROC MEANS DATA = predictions NOPRINT;
  CLASS trtmt;
  VAR pdeath;
  OUTPUT OUT = eout(WHERE=(trtmt>.z)) MEAN=;
DATA final;
  SET eout END=eof;
  KEEP predicted_outcome_all_treated predicted_outcome_all_untreated ATE;
  RETAIN predicted_outcome_all_treated predicted_outcome_all_untreated ATE;
  IF trtmt = 1 THEN predicted_outcome_all_treated = pdeath;
  ELSE IF trtmt = 0 THEN predicted_outcome_all_untreated = pdeath;
  ATE = predicted_outcome_all_treated-predicted_outcome_all_untreated;
  IF eof THEN OUTPUT;
PROC PRINT DATA = final NOOBS;
  TITLE "Expected outcomes under treatment/no treatment, additive effect";
RUN;
 
