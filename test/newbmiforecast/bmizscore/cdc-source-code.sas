*************************************************************
* This sas program calculates CDC percentiles and z-scores based on 
* the 2000 cdc growth charts
* (http://www.cdc.gov/growthcharts/cdc_charts.htm. 
* The reference population is children examined in NCHS studies 
* from 1963-65 to 1988-94. 

* Biologically implausible values are flagged, but these values are not necessarily incorrect.  
* Please see the information * on 'Extreme or biologically implausible values' on the web page.

* Unless you have a very good reason to alter this file (and this is unlikely),
* please do NOT make changes.  This file is meant to be called with a %include
* statement from your SAS program.  

/* 
   Revised in April 2015 to fix problems with labels and formats
   Revised in Dec 2015 to change upper cut points for BIVs 
   Revised in Nov 2016 to add bmip95 and bmi95 (and drop SFs and ACs) to output
   Revised in 2022 to add extended BMIz and other new metrics. Also added 'agemos >= 24' to LMS calculations
*/


**********************************************;
*********  macros for calculations ***********;
**********************************************;
%macro _zscore(var,l, m, s, z, p, f); 
	if &var >0 & agemos >=24 then do;  
		if abs(&L) ge 0.01 then &z=((&var / &M)**&L-1)/(&L * &S);
		  else if .z < abs(&L) < 0.01 then &z=log(&var / &M)/ &S;  
		&p=probnorm(&z)*100;
	
		sdl=((&M - &M*(1 - 2 * &L * &S)**(1 / &L)) / 2); 
		sdh=((&M * (1 + 2 * &L * &S)**(1 / &L) - &M) / 2);
			if &var lt &M then &f= (&var - &M) / sdl; else &f= (&var - &M) / sdh; 
	end; 
%mend _zscore;

%macro _cuts(var,out,l,u);
	if &L <= &var <= &u then &out=0; 
		else if &var> &u then &out=1; else if .< &var < &L then &out= -1; 
%mend _cuts;
************ End of Macros ************;

data _orig _mydata _old; set mydata; _id=_n_; length agemos 8 sex 3; 
	* length and stand_ht are used later in deciding to use 
    * wt-for-stature or wt-for-len reference;
  if (. < agemos < 24) then do; length=height; end; 
	if agemos>=24 then stand_ht=height;

  if agemos lt 240 then output _mydata; else output _old; 
  output _orig;

	 
data _cinage _cinlen _cinht; set _mydata; 
if agemos ge 0 and agemos lt 0.5 then _agecat=0;
    else _agecat=int(agemos+0.5)-0.5;
if bmi < 0 & ( weight>0 & height >0 & agemos >=24) then bmi=weight/(height/100)**2;
output _cinage;

if length > . then do;
    if length >= 45 then _htcat=int(length+0.5)-0.5;
        if 45 <= length < 45.5 then _htcat=45;
        output _cinlen; 
	end;
if stand_ht > . then do;
    if stand_ht ge 77.5 then _htcat=int(stand_ht+0.5)-0.5;
        else if 77<= stand_ht < 77.5 then _htcat=77;
        output _cinht;
    end;
	
***********************************************************************;
*** begin the for-age calcs - note that this calls up the refdir libname;
data crefage; set refdir.CDCref_d; where denom='age'; 
	* contains all merged LMS data - use 'denom' variable;
	length sex 3; 
	
Proc sort; by sex _agecat; 	proc sort data=_cinage; by sex _agecat;
	
data finfage; merge _cinage (in=a) crefage (in=b); by sex _agecat;  if a;
    ageint = _agemos2-_agemos1; dage=agemos- _agemos1;
    
array l0  _llg  _mlg  _slg  _lht  _mht  _sht  _lwt  _mwt  _swt _lhc  _mhc  _shc  
	_lbmi  _mbmi  _sbmi;
array l1 _llg1 _mlg1 _slg1 _lht1 _mht1 _sht1 _lwt1 _mwt1 _swt1 _lhc1 _mhc1 _shc1 
	_lbmi1 _mbmi1 _sbmi1;
array l2 _llg2 _mlg2 _slg2 _lht2 _mht2 _sht2 _lwt2 _mwt2 _swt2 _lhc2 _mhc2 _shc2 
	_lbmi2 _mbmi2 _sbmi2;
do over l0; l0= l1 + (dage * (l2 - l1)) / ageint; end;

if agemos < 24 then _mbmi=.; *theres a valid value for 23.5 months! ;

* note that upper cutpoints were changed in 2016 to +4 (ht) and +8 (wt, BMI);
%_zscore(length, _llg, _mlg, _slg, lgz, lgpct, mod_lenz); 
	%_cuts(mod_lenz, _bivlg, -5, 4);  
%_zscore(stand_ht, _lht, _mht, _sht, stz, stpct, mod_statz); 
	%_cuts(mod_statz, _bivst, -5, 4);
%_zscore(weight, _lwt, _mwt, _swt, waz, wapct, mod_waz); 
	%_cuts(mod_waz, _bivwt, -5, 8);
%_zscore(headcir, _lhc, _mhc, _shc, headcz, headcpct, mod_headcz); 
	%_cuts(mod_headcz, _bivhc, -5, 5); * no change to head_circ;
%_zscore(bmi, _lbmi, _mbmi, _sbmi, bmiz, bmipct, mod_bmiz); 
	%_cuts(mod_bmiz, _bivbmi, -4, 8);
 
bmi50 = _mbmi * ((1 + _lbmi*_sbmi*probit(0.50))**(1/_lbmi));  
bmip50 = 100 * (bmi/bmi50); * % of 50th percentile;
bmi95 = _mbmi * ((1 + _lbmi*_sbmi*probit(0.95))**(1/_lbmi));    
bmip95 = 100 * (bmi/bmi95);  * % of 95th percentile;
bmi120 = 1.2 * bmi95; * 120% of 95th percentile;

* other BMI metrics, PMID 31439056;
* www.cdc.gov/growthcharts/percentile_data_files.htm ;
if sex=1 then do; mref= 23.02029; sref = 0.13454; end; *reference values at age 240 mos;
if sex=2 then do; mref= 21.71700; sref = 0.15297; end;

z1=((bmi/_mbmi) - 1) / _sbmi; * LMS formula when L=1: ((BMI/M)-1)/S;
adj_perc_median = z1 * 100 * sref; * adjusted %distance from median; 

  /* other metrics that aren't calculated 
perc_median = z1 * 100 * _sbmi; * unadjusted %distance from median; 
dist = z1 * _mbmi * _sbmi; * unadjusted distance from median with L=1;
adj_dist = z1 * sref * mref; * adjusted (to age 20y) dist from median; 
z0 = log(bmi/_mbmi)/_sbmi; *LMS transformation with L=0;
log_perc_median = z0 * 100 * _sbmi; * unadjusted %distance from median with L=0 (log scale);
adj_log_perc_median = z0 * 100* sref;  * adjusted %distance from median w L=0 (log scale);   
  */

* calculations for extended BMIz and other metrics;
original_bmiz=bmiz; 
original_bmipct=bmipct;
agey=agemos/12;

if sex=1 then sigma = 0.3728 + 0.5196*agey - 0.0091*agey**2;		
 else if sex=2 then sigma = 0.8334 + 0.3712*agey - 0.0011*agey**2;

if bmipct > 95 then do; 
  bmipct = 90 + 10 *  (probnorm((bmi - bmi95) / sigma));
  if bmipct <= 99.999999999999992 then bmiz = probit(bmipct/100);
end;
if bmipct > 99.9999999 & bmiz <= .z then bmiz=8.21;
  /* above line is to account for computer precision that can't distinguish between 1.0 and 0.99999999999999993
     in the probit function.  SAS spits out a missing value for these very high percentiles  */


drop _llg1 _mlg1 _slg1 _lht1 _mht1 _sht1 _lwt1 _mwt1 _swt1 _lhc1 _mhc1 _shc1 
	_lbmi1 _mbmi1 _sbmi1 _llg2 _mlg2 _slg2 _lht2 _mht2 _sht2 _lwt2 _mwt2 _swt2 
	_lhc2 _mhc2 _shc2 _lbmi2 _mbmi2 _sbmi2 _lwht1  _mwht1  _swht1  _lwht2 
	_mwht2  _swht2  _lwlg1  _mwlg1  _swlg1 _lwlg2  _mwlg2  _swlg2;


******************************************;
*** begin for-length and for-stand_ht calcs;
*****************************************;

*** begin for-length calcs, birth to 36 mos;
proc sort data=_cinlen; by sex _htcat;
data creflg; set refdir.CDCref_d (keep=denom sex _lg1--_swlg2);  
	where denom='length'; _htcat=_lg1; 	length sex 3; 
    proc sort data=creflg; by sex _htcat; 

data finflg; merge _cinlen (in=a) creflg; by sex _htcat;
    if a & (43 < length <104); 
    lenint = _lg2- _lg1; dlen=length - _lg1;
array l  _lwl  _mwl  _swl; 
array l1  _lwlg1  _mwlg1  _swlg1; 
array l2  _lwlg2  _mwlg2  _swlg2;
do over l; l = l1 + (dlen * (l2 - l1)) / lenint; end;

%_zscore(weight, _lwl, _mwl, _swl, wlz, wlpct, mod_wlz); 
	%_cuts(mod_wlz,_bivwlg, -4, 8); 
keep _id sex _agecat agemos weight mod_wlz _bivwlg wlz wlpct;

*** begin for-stand_ht calcs, zwtstat.xls;
proc sort data=_cinht; by sex _htcat;
data crefht; set refdir.CDCref_d 
	(keep=denom sex _ht1 _ht2 _lwht1 _lwht2 _mwht1 _mwht2 _swht1 _swht2); 
	where denom='height'; 	_htcat=_ht1;  length sex 3; 
	proc sort data=crefht; by sex _htcat; 

data finfht; merge _cinht (in=a) crefht; by sex _htcat;
    if a & (77 < height <122);
    htint = _ht2- _ht1; dht=height - _ht1; 
array l   _lwh     _mwh   _swh; 
array l1  _lwht1  _mwht1  _swht1; 
array l2  _lwht2  _mwht2  _swht2;
do over l; l = l1 + (dht * (l2 - l1)) / htint; end;

%_zscore(weight, _lwh, _mwh, _swh, wstz, wstpct, mod_wstz);   
	%_cuts(mod_wstz,_bivwst,-4,8); 
keep _id sex _agecat agemos weight mod_wstz _bivwst wstz wstpct;

*** combine the for-age, for-length, and for-height calcs;
proc sort data=finflg; by _id; proc sort data=finfht; by _id;
data lenht; merge finflg finfht; by _id; 
		
proc sort data=finfage; by _id; 

data _outdata; * define height vars as max of standing height and length vars;
  merge finfage lenht; by _id; 
 	
    array a stz  stpct _bivst mod_statz wstz wstpct _bivwst mod_wstz ;
    array b lgz  lgpct _bivlg mod_lenz  wlz  wlpct  _bivwlg mod_wlz;
    array c haz  hapct _bivht mod_haz   whz  whpct  _bivwh mod_whz;
	do over c; if agemos ge 24 then c=a; else c=b; end;

	if .z < weight< 0.01 then do; waz=.; wapct=.; bmiz=.; bmipct=.; whz=.; whpct=.; end;
	if .z < height< 0.01 then do; haz=.; hapct=.; bmiz=.; bmipct=.; whz=.; whpct=.; end;
	if .z < headcir< 0.01 then do; headcz=.; headcpct=.; end;
	
	  /*
	rename bmiz=original_bmiz bmipct=original_bmipct ext_bmiz=bmiz ext_bmip=bmipct;
	  */
	 
data _outdata; set _outdata;   
   keep _id mod_bmiz mod_haz mod_headcz mod_lenz mod_waz mod_whz mod_wstz mod_wlz _bivbmi _bivhc 
	_bivht _bivlg _bivwh _bivst _bivwlg _bivwst _bivwt agemos bmi bmipct 
	bmip50 bmip95 bmiz bmi50 bmi95 bmi120 height haz lgz headcir headcpct headcz
	lgpct sex wapct waz whpct whz hapct wstz wstpct mod_statz
	original_bmiz original_bmipct 
	 /* 
	perc_median adj_perc_median dist adj_dist log_perc_median adj_log_perc_median
	 */
	; 
	
 data _outdata; set _outdata _old; * combine with older, excluded people;
	label 
	  waz='weight-for-age Z' 
	   _bivbmi='BIV BMI-for-age' 
	   _bivhc='BIV head_circ' 
	   _bivht='BIV height-for-age' 
	   _bivwh='BIV weight-for-height' 
	   _bivwt='BIV weight-for-age'
	   
	   mod_bmiz='modified BMI-for-age Z'  
	   mod_haz='modified height-for-age Z' 
	   mod_headcz='modified head_circ Z'
	   mod_waz='modified weight-for-age Z'
	   mod_whz='modified weight-for-height Z' 
	   
	   bmi50 = 'CDC median BMI-for-age'
	   bmi95 = 'CDC 95th pctl BMI-for-age'
	   bmi120 = '120% of the CDC 95th pctl'

	   bmiz='BMI-for-age Z'	  
	   bmipct='BMI-for-age percentile' 
       bmip50 = '% of 50th percentile'
	   bmip95='% of 95th BMI percentile' 
	   
	   original_bmiz='original LMS BMI Z'
	   original_bmipct='original LMS BMI percentile'	  
	   	   
	   /* adj_perc_median='Adjusted BMI % from median' */
	   
	   hapct='height-for-age percentile'  
	   haz='height-for-age Z' 
	   headcir='head circumference'
	   headcz='head_circ-for-age Z' 
	   headcpct='head_circ-for age perc' 	   
       wapct='weight-for-age percentile'
       whpct='weight-for-height percentile' 	 	 
	   whz='weight-for-height Z' 
      ; 
proc sort data= _outdata; by _id; proc sort data=_orig; by _id;

data _cdcdata; update _outdata _orig; by _id;  
	* variables in _orig dataset will overwrite any changes that were made;
	drop _id _bivst mod_statz wstz wstpct mod_wstz  lgz  lgpct _bivlg mod_lenz  
		 mod_wlz length stand_ht _bivwst _bivwlg;
