# Copyright 2023 Hanan Ahmed, John Einmahl and Chen Zhou

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and 
# associated documentation files (the "Software"), to deal in the Software without restriction, including 
# without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do so, subject to the 
# following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial 
# portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR 
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
# OTHER DEALINGS IN THE SOFTWARE

# modified code from Paper_code_JASA.R
library("LaplacesDemon")
library("fMultivar")
library("evd")
library("condmixt")
library("ismev")
library("mev")
rm(list = ls())

est <- function(i, Y_T, Y_S_nm, Y_S_n, n, m, k, g_values) {
    set.seed(i)

    repl <- 1
    results_tab <- matrix(NA, repl, 2)

    x <- matrix(NA, n, repl)
    y_tilde <- matrix(NA, n, repl)
    y <- matrix(NA, n, repl)

    for(le in 1:repl) {
        x[, le] = Y_T
        y_all = Y_S_nm
        y_ = Y_S_n
        y[, le] = y_
     
        y3_rank = rank(y_all)
        F_nm = rep(NA,(n+m))
        for(t in 1:(n+m)){F_nm[t]=y3_rank[t]/(n+m)}
        F_nm1 = F_nm-(1/(2*(n+m)))
        y1 = -log(1-F_nm1)
        y_tilde1 = y1[1:n]
        y_tilde[,le] = y_tilde1
    }

    a_vec = matrix(g_values,nrow=1,ncol=2)
    estimators = matrix(NA,length(a_vec),2)
    for (l in 1:length(a_vec)){
    # when g=0
    if(a_vec[l]==0) {for (re in 1:repl){
        #############gamma_1 estimate#########
        x_sort=sort(x[,re])
        ml=gp.fit(x[,re],x_sort[n-k],method="zhang")
        gamma_1=ml$approx.mean[2]
        ############gamma_2 estimate########
        y_tilde_sort=sort(y_tilde[,re])
        ml1=gp.fit(y_tilde[,re],y_tilde_sort[n-k],method="zhang")
        gamma_2=ml1$approx.mean[2]
        ###########(0)###############
        y_sort=sort(y[,re])
        x_rank=rank(x[,re])
        y1_rank=rank(y[,re])
        ########calculation of tail dependence#########
        indicator<-rep(NA,n)
        for (j in 1:n){if (x[j,re]>x_sort[n-k] && y[j,re]>y_sort[n-k]) {indicator[j]=1} else{indicator[j]=0}} 
        R1_y<-sum(indicator)/k #R_xy(1,1)
        
        indicator1=matrix(NA,n,1)
        for(i in 1:n){ if(x[i,re]>x_sort[n-k] && y[i,re]>y_sort[n-k]){indicator1[i]=((n-x_rank[i]+1)/k)^gamma_1} else{indicator1[i]=0}}
        R_s_y_int=(1/gamma_1)*(R1_y-(sum(indicator1)/k)) #\int_0^1 R_xy(s,1)/s^{1-\gamma}
        
        indicator4t=matrix(NA,n,1)
        for(i in 1:n){ if(x[i,re]>x_sort[n-k] && y[i,re]>y_sort[n-k]){indicator4t[i]=(log((n-y1_rank[i]+1)/k))} else{indicator4t[i]=0}}
        R_t_y=(-(sum(indicator4t)/k)) #\int_0^1 R_xy(1,t)/t 

        ###########improved estimator of the extreme value index################
        R_1=((gamma_1/(gamma_1+1))*((((2*gamma_1)+1)*R_s_y_int)-R_t_y))-R1_y
        gamma_imp_2d=gamma_1+((gamma_1+1)*R_1*gamma_2)
        
        #########################################################################################
        # store results
        results_tab[re,1]=gamma_1
        results_tab[re,2]=gamma_imp_2d
    }
        # store estimators
        estimator = matrix(NA,1,2)
        estimator[1,1] = results_tab[re,1]
        estimator[1,2] = results_tab[re,2]
    } 

    # if g neq 0
    else{a=a_vec[l]
    y_tilde2=matrix(NA,n,repl)
    y_tilde3=matrix(NA,n,repl)
    for (lk in 1:repl){ 
        y_tilde2[,lk]=1-exp(-y_tilde[,lk])
        y_tilde3[,lk]=(1-((1-y_tilde2[,lk])^(-a)))/-a
    } 
    
    for (re in 1:repl){
        ###################################improved estimator extreme value index#####################################
        ###############Extreme value index###########
        #############gamma_1 estimate#########
        x_sort=sort(x[,re])
        ml=gp.fit(x[,re],x_sort[n-k],method="zhang") 
        gamma_1=ml$approx.mean[2]
        ############gamma_2 estimate########
        y_tilde_sort=sort(y_tilde3[,re])
        ml1=gp.fit(y_tilde3[,re],y_tilde_sort[n-k],method="zhang") 
        gamma_2=ml1$approx.mean[2]
        ###########dependence calculations###############
        y_sort=sort(y[,re])
        x_rank=rank(x[,re])
        y1_rank=rank(y[,re])
        
        indicator<-rep(NA,n)
        for (i in 1:n){if (x[i,re]>x_sort[n-k] && y[i,re]>y_sort[n-k]) {indicator[i]=1} else{indicator[i]=0}}
        R1_y<-sum(indicator)/k #R_xy(1,1)
        
        indicator1=matrix(NA,n,1)
        for(i in 1:n){ if(x[i,re]>x_sort[n-k] && y[i,re]>y_sort[n-k]){indicator1[i]=((n-x_rank[i]+1)/k)^gamma_1} else{indicator1[i]=0}}
        R_s_y_int=(1/gamma_1)*(R1_y-(sum(indicator1)/k)) #R_xy(s,1)/s^{1-\gamma}
        
        indicator3=matrix(NA,n,1)
        for(i in 1:n){ if(x[i,re]>x_sort[n-k] && y[i,re]>y_sort[n-k]){indicator3[i]=((n-y1_rank[i]+1)/k)^a} else{indicator3[i]=0}}
        R_t_y_int=(1/a)*(R1_y-(sum(indicator3)/k)) #R_xy(1,t)/t^{1-g}
        
        ###########improved extreme value index estimator################
        R_1=R1_y+(((a-gamma_1)/(gamma_1+a+1))*((((2*gamma_1)+1)*R_s_y_int)-(((2*a)+1)*R_t_y_int)))
        gamma_imp_2d=gamma_1+(((gamma_1+1)/(a+1))*R_1*(a-gamma_2))
        
        #########################################################################################
        # store results
        results_tab[re,1]=gamma_1
        results_tab[re,2]=gamma_imp_2d
    }
        # store estimators
        estimator = matrix(NA,1,2)
        estimator[1,1] = results_tab[re,1]
        estimator[1,2] = results_tab[re,2]
    } 
    estimators[l,]=estimator
    }
    return(list(est=estimators))
}