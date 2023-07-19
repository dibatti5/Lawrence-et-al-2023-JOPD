#code for Larewnce et al 2023 JOPD manuscript

#Relevant libraries
library(tidyverse)
library(gt)
library(gtsummary)
library(viridis)
library(hrbrthemes)
library(likert)
library(scales)
library(ggrepel)
library(forcats)
library(scales)
library(rstan)
library(rethinking)
library(cowplot)
library(bayesplot)
library(tidybayes)
library(modelr) 
library(dagitty)
library(label.switching)
library(poLCA)

#for speed with mcmc
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

#functions
HDILow<- function(x, HDI=0.9) {
  sortedPts = sort( x)
  ciIdxInc = ceiling( HDI * length( sortedPts ) )
  nCIs = length( sortedPts ) - ciIdxInc 
  ciWidth = rep( 0 , nCIs )
  for ( i in 1:nCIs ) {
    ciWidth[ i ] = sortedPts[ i + ciIdxInc ] - sortedPts[ i ] }
  HDImin = sortedPts[ which.min( ciWidth ) ]
  HDImax = sortedPts[ which.min( ciWidth ) + ciIdxInc ] 
  return( HDImin)
} #for shortest distance credible intervals
HDIHigh<- function(x, HDI=0.9) {
  sortedPts = sort( x)
  ciIdxInc = ceiling( HDI * length( sortedPts ) )
  nCIs = length( sortedPts ) - ciIdxInc 
  ciWidth = rep( 0 , nCIs )
  for ( i in 1:nCIs ) {
    ciWidth[ i ] = sortedPts[ i + ciIdxInc ] - sortedPts[ i ] }
  HDImin = sortedPts[ which.min( ciWidth ) ]
  HDImax = sortedPts[ which.min( ciWidth ) + ciIdxInc ] 
  return( HDImax)
}

#load file
df <- read.csv('dmt_data.csv',stringsAsFactors = FALSE)
colnames(df)
df2 <- df[-c(10,13,18,22)] #remove missing columns

###########LCA MODELLING############
##The LCA modelling here was taken from:
#Ji, F., A. Amanmyradova, and S. Rabe-Hesketh. 
#Bayesian Latent Class Models and Handling of Label Switching. 2021; 
#Available from: https://mc-stan.org/users/documentation/case-studies/Latent_class_case_study.html#.

#Variational bayes version (not in paper)
#explored 2,3 and 4 class models

#2 class model
#Variational algorithm
dat <- list(I = 19, J = 227, C = 2,y = as.matrix(df2[-c(1,2)]))
stan_vb <- stan_model(fil = 'lca.stan')
vb_fit <- vb(stan_vb,data = dat, iter = 50000,elbo_samples = 1000,
             algorithm = c('fullrank'),output_samples = 10000, 
             tol_rel_obj = 0.00001)

print(vb_fit, c("alpha", "p"))
rstan::traceplot(vb_fit)

#3 class model (bad paretos)
dat <- list(I = 19, J = 227, C = 3,y = as.matrix(df2[-c(1,2)]))
stan_vb <- stan_model(fil = 'lca.stan')
vb_fit3 <- vb(stan_vb,data = dat, iter = 50000,elbo_samples = 1000,
              algorithm = c('fullrank'),output_samples = 10000, 
              tol_rel_obj = 0.00001)

print(vb_fit3, c("alpha", "p", "p_prod"))
rstan::traceplot(vb_fit3)

#4 class model (bad paretos)
dat <- list(I = 19, J = 227, C = 4,y = as.matrix(df2[-c(1,2)]))
stan_vb <- stan_model(fil = 'lca.stan')
vb_fit4 <- vb(stan_vb,data = dat, iter = 15000,elbo_samples = 1000,
              algorithm = c('fullrank'),output_samples = 10000, 
              tol_rel_obj = 0.00001)

#Full Simulation (used in manuscript)
#explored 2,3 and 4 class models

#2 class full mcmc
dat2 <- list(I = 19, J = 227, C = 2,y = as.matrix(df2[-c(1,2)]))
mod_2class <- stan('lca.stan',data = dat2, chains = 4, iter = 15000,cores = 4)

#check model
print(mod_2class,c("alpha","p")) 
traceplot(mod_2class,c("alpha","p"))

#for the analysis used in the manuscript, the 2-class model ran without the
#need for label switching. However, if label switching is needed when replicating, run the 
#code below from lines 109 - 168

#Post-hoc label switch
# extract stan fit as the required format of the input
pars <- mod_2class %>% names %>% `[`(1:40)
post_par <- rstan::extract(mod_2class,
                           c("alpha", "p", "pred_class", "pred_class_dis", "lp__"),
                           permuted = TRUE)

# simulated allocation vectors
post_class <- ((post_class_p[,,1] > 0.5)*1) + 1

# classification probabilities
post_class_p <- post_par$pred_class

m = 30000 # of draws
K = 2 # of classes
J =20 # of component-wise parameters

# initialize mcmc arrays
mcmc <- array(data = NA, dim = c(m = m, K = K, J = J))

# assign posterior draws to the array
mcmc[, , 1] <- post_par$alpha
for (i in 1:(J - 1)) {
  mcmc[, , i + 1] <- post_par$p[, , i]
}

# set of selected relabeling algorithm
set <-
  c("PRA",
    "ECR",
    "ECR-ITERATIVE-1",
    "ECR-ITERATIVE-2")

# find the MAP draw as a pivot
mapindex = which.max(post_par$lp__)

# switch labels
ls_lcm2 <-
  label.switching(
    method = set,
    zpivot = post_class[mapindex,],
    z = post_class,
    K = K,
    prapivot = mcmc[mapindex, ,],
    constraint = 1,
    mcmc = mcmc,
    p = post_class_p,
    data = dat2$y
  )

ls_lcm2$similarity
mcmc_permuted <- permute.mcmc(mcmc, ls_lcm2$permutations$ECR)

# change dimension for each parameter defined as in the Stan code
mcmc_permuted_2class <-
  array(
    data = mcmc_permuted$output,
    dim = c(30000, 1, 40),
    dimnames = list(NULL, NULL, pars))

fit_permuted <-
  monitor(mcmc_permuted_2class, warmup = 0,  digits_summary = 4)

#3 class full mcmc
dat3 <- list(I = 19, J = 227, C = 3,y = as.matrix(df2[-c(1,2)]))
mod_3class <- stan('lca.stan',data = dat3, chains = 4, iter = 15000,cores = 4)

#check model
print(mod_3class,c("alpha","p","p_prod")) 
traceplot(mod_3class,c("alpha","p"))

#Post-hoc label switch
# extract stan fit as the required format of the input
pars <- mod_3class %>% names %>% `[`(1:60)
post_par <- rstan::extract(mod_3class,
                           c("alpha", "p", "pred_class", "pred_class_dis", "lp__"),
                           permuted = TRUE)

# simulated allocation vectors
post_class <- post_par$pred_class_dis
# classification probabilities
post_class_p <- post_par$pred_class

m = 30000 # of draws
K = 3 # of classes
J =20 # of component-wise parameters

# initialize mcmc arrays
mcmc <- array(data = NA, dim = c(m = m, K = K, J = J))

# assign posterior draws to the array
mcmc[, , 1] <- post_par$alpha
for (i in 1:(J - 1)) {
  mcmc[, , i + 1] <- post_par$p[, , i]
}

# set of selected relabeling algorithm
set <-
  c("PRA",
    "ECR",
    "ECR-ITERATIVE-1",
    "ECR-ITERATIVE-2")

# find the MAP draw as a pivot
mapindex = which.max(post_par$lp__)

# switch labels
ls_lcm3 <-
  label.switching(
    method = set,
    zpivot = post_class[mapindex,],
    z = post_class,
    K = K,
    prapivot = mcmc[mapindex, ,],
    constraint = 1,
    mcmc = mcmc,
    p = post_class_p,
    data = dat3$y
  )

ls_lcm3$similarity
mcmc_permuted <- permute.mcmc(mcmc, ls_lcm3$permutations$ECR)

# change dimension for each parameter defined as in the Stan code
mcmc_permuted_3class <-
  array(
    data = mcmc_permuted$output,
    dim = c(30000, 1, 60),
    dimnames = list(NULL, NULL, pars))

fit_permuted <-
  monitor(mcmc_permuted_3class, warmup = 0,  digits_summary = 3)

#4 class full mcmc
dat4 <- list(I = 19, J = 227, C = 4,y = as.matrix(df2[-c(1,2)]))
mod_4class <- stan('lca.stan',data = dat4, chains = 4, iter = 15000,cores = 4)

#check model
print(mod_4class,c("alpha","p","p_prod")) 
traceplot(mod_4class,c("alpha","p"))

#Post-hoc label switch
# extract stan fit as the required format of the input
pars <- mod_4class %>% names %>% `[`(1:80)
post_par <- rstan::extract(mod_4class,
                           c("alpha", "p", "pred_class", "pred_class_dis", "lp__"),
                           permuted = TRUE)

# simulated allocation vectors
post_class <- post_par$pred_class_dis
# classification probabilities
post_class_p <- post_par$pred_class

m = 30000 # of draws
K = 4 # of classes
J =20 # of component-wise parameters

# initialize mcmc arrays
mcmc <- array(data = NA, dim = c(m = m, K = K, J = J))

# assign posterior draws to the array
mcmc[, , 1] <- post_par$alpha
for (i in 1:(J - 1)) {
  mcmc[, , i + 1] <- post_par$p[, , i]
}

# set of selected relabeling algorithm
set <-
  c("PRA",
    "ECR",
    "ECR-ITERATIVE-1",
    "ECR-ITERATIVE-2")

# find the MAP draw as a pivot
mapindex = which.max(post_par$lp__)

# switch labels
ls_lcm4 <-
  label.switching(
    method = set,
    zpivot = post_class[mapindex,],
    z = post_class,
    K = K,
    prapivot = mcmc[mapindex, ,],
    constraint = 1,
    mcmc = mcmc,
    p = post_class_p,
    data = dat4$y
  )

ls_lcm4$similarity
mcmc_permuted <- permute.mcmc(mcmc, ls_lcm4$permutations$ECR)
dim(mcmc_permuted$output)
tail(mcmc_permuted$output[,,1])
# change dimension for each parameter defined as in the Stan code
mcmc_permuted_4class <-
  array(
    data = mcmc_permuted$output,
    dim = c(30000, 1, 80),
    dimnames = list(NULL, NULL, pars))

fit_permuted <-
  monitor(mcmc_permuted_4class, warmup = 0,  digits_summary = 3)

#comparison to LCA using max likelihood (poLCA function)
#recode to categories 1 and 2
df3 <- df2[-c(1,2)]
df3 <- as.data.frame(ifelse(df3==0,1,2))

f <- cbind(Sense.of.familiarity,Feels.like.home.or.going.home,Sense.of.welcoming,
           Sense.of.belonging,Sense.of.comfort,Sense.of.nostalgia,
           Sense.of.remembering.something.associated.with.the.experience.you.may.have.forgotten.or.will.forget,
           Intuitive.sense.that.you.are.returning.to.a.state..place..space..or.environment.before,
           Intuitive.sense.that.you.will.return.to.the.state..place..space..or.environemnt.again,
           Feels.like.you.have.gone.through.this.experience.or.act.before,
           Feels.like.you.will.go.through.this.experience.or.act.again,
           Feels.like.you.have.done.this.many.many.times.before,A.sense.of.deja.vu,
           A.sense.that.you.visited.a.place.that.exists.preconception.or.after.death,
           A.sense.that.this.is.a.place.your.consciousness.resides.either.now.or.has.previously,
           A.sense.you.have.experienced.a.place.that.is.eternal.or.infinite,
           Encounter.with.a.entity.or.presence.that.you.knew..felt.familiar.with..or.have.an.estahblished.bond.with,
           Enctouner.with.an.entity.or.presence.that.knew.you.or.expresed.a.familiarity.or.bond.with.you,
           Encoutner.with.an.entity.or.presence.that.feels.like.your.family)~1
f_out2 <- poLCA(f,data=df3,nclass=2)
f_out3 <- poLCA(f,data=df3,nclass=3)
f_out4 <- poLCA(f,data=df3,nclass=4)

#Plot out probability signatures for each model
#extract posterior for 2-class model
mod_2class@model_pars
p_2class <- data.frame(extract(mod_2class,pars=c('alpha','p')))
colnames(p_2class) <- c("Class 1",'Class 2','Sense of familiarity','Sense of familiarity',
                        'Feels like home','Feels like home','Sense of comfort',
                        'Sense of comfort','Sense of welcoming','Sense of welcoming',
                        'Sense of belonging','Sense of belonging','Sense of nostalgia',
                        'Sense of nostalgia','Sense of remembering...','Sense of remembering...',
                        'Intuitive sense you are returning','Intuitive sense you are returning',
                        'Intuitive sense you will return','Intuitive sense you will return',
                        'Sense of deja vu','Sense of deja vu','Feels like gone through before',
                        'Feels like gone through before','Feels like done this many times',
                        "Feels like done this many times",'Feels like will happen again',
                        'Feels like will happen again','Sense of visiting preconception/death',
                        'Sense of visiting preconception/death','Experience place that is eternal',
                        'Experience place that is eternal','Experience place consciousness resides',
                        'Experience place consciousness resides','Encounter with familiar entity - established bond',
                        'Encounter with familiar entity - established bond',
                        'Encounter with entity - had bond with you','Encounter with entity - had bond with you',
                        'Encounter with entity - feels like family','Encounter with entity - feels like family')
#separate because of double column names, gather, and re-bind
p2_class1 <- p_2class[c(seq(1,ncol(p_2class),by = 2))]
p2_class2 <- p_2class[c(seq(2,ncol(p_2class),by = 2))]
p2_class1 <- gather(p2_class1,key = 'Items', value = 'Probability of Answering Yes')
p2_class2 <- gather(p2_class2,key = 'Items', value = 'Probability of Answering Yes')
p2_class_df <- rbind.data.frame(p2_class1,p2_class2)
mean(p_2class$`Class 2`) #check both classes for class label creation below
precis(p_2class,2,probs = 0.9) #for class uncertainty (first 2 rows)
p2_class_df$class <- rep(c('Class 1 (24%)','Class 2 (76%)'), each = 600000)
p2_class_df$category <- NA
p2_class_df$category[p2_class_df$Items%in%colnames(p_2class)[c(3,5,7,9,11,13,15)]]<- 
  'Feeling/Emotion/Knowledge'
p2_class_df$category[p2_class_df$Items%in%colnames(p_2class)[c(17,19)]]<- 
  'Place/Space/State/Environment'
p2_class_df$category[p2_class_df$Items%in%colnames(p_2class)[c(21,23,25,27)]]<- 
  'Experience/Act'
p2_class_df$category[p2_class_df$Items%in%colnames(p_2class)[c(29,31,33)]]<- 
  'Transcendent Experience'
p2_class_df$category[p2_class_df$Items%in%colnames(p_2class)[c(35,37,39)]]<- 
  'Entity Encounter'
p2_class_df$Items <- factor(p2_class_df$Items, 
                            levels=colnames(p_2class)[c(seq(1,ncol(p_2class),by = 2))])
p2_class_df$category <- factor(p2_class_df$category,
                               levels = c('Feeling/Emotion/Knowledge',
                                          'Place/Space/State/Environment',
                                          'Experience/Act',
                                          'Transcendent Experience',
                                          'Entity Encounter' ))

#Plot
theme_set(theme_tidybayes() + panel_border())
fig1 <- p2_class_df[!(is.na(p2_class_df$category)),] %>%
  ggplot(aes(y = fct_rev(Items),
             x = `Probability of Answering Yes`))+
  geom_vline(xintercept = 0.5,alpha= 0.4,linetype =2 ) +
  geom_vline(xintercept = 0.25,alpha= 0.4,linetype =2 ) +
  geom_vline(xintercept = 0.75,alpha= 0.4,linetype =2 ) +
  theme(strip.text.y = element_text(face = 'bold',size = 7),
        strip.text.x = element_text(face = 'bold',size = 9),
        axis.title.x = element_text(size = 9,face = 'bold'),
        axis.title.y = element_text(size = 9,face = 'bold'),
        axis.text.y = element_text(face = 'bold',size = 8),
        legend.position = 'none',
        legend.text = element_text(face = 'bold',size = 10),
        axis.text.x = element_text(face = 'bold',size = 10))+
  ylab("Items") + 
  stat_gradientinterval(.width = c(0.70,0.9),
                        fill = 'blue',colour = 'black') +
  facet_grid(category~class,scales = 'free')
fig1
ggsave('fig1.jpg',fig1,dpi = 600,width = 8,height = 9)

#two class table
#remake df in wide format
p2_class1 <- p_2class[c(seq(1,ncol(p_2class),by = 2))]*100
p2_class2 <- p_2class[c(seq(2,ncol(p_2class),by = 2))]*100

#first class
tbl_class1 <- 
  tbl_summary(p2_class1[-c(1)],
              missing = 'no',
              digits = all_continuous() ~ 1,
              statistic = all_continuous()~c("{mean} ({HDILow},{HDIHigh})"))%>%
  modify_footnote(all_stat_cols() ~ "Mean (90% Compatibility Interval) from posterior draws")%>%
  modify_header(label = "**Question**",stat_0 = '**Yes (%)**')%>%
  modify_caption("**Table X. Question responses by latent class**")

#second class
tbl_class2 <- 
  tbl_summary(p2_class2[-c(1)],
              missing = 'no',
              digits = all_continuous() ~ 1,
              statistic = all_continuous()~c("{mean} ({HDILow},{HDIHigh})"))%>%
  modify_footnote(all_stat_cols() ~ "Mean (90% Compatibility Interval) from posterior draws")%>%
  modify_header(label = "**Question**", stat_0 ='**Yes (%)**')

#merge
tbl_merge <-
  tbl_merge(
    tbls = list(tbl_class1,tbl_class2),
    tab_spanner = c("**Class 1 (24%)**", "**Class 2 (76%)**"))%>%
  modify_table_body(mutate, groupname_col = 
                      case_when(variable == 'Sense of familiarity'|
                                  variable == 'Feels like home'|
                                  variable =='Sense of comfort'|
                                  variable == 'Sense of welcoming'|
                                  variable =='Sense of belonging'|
                                  variable == 'Sense of nostalgia'|
                                  variable == 'Sense of remembering...'~ 
                                  "Feeling/Emotion/Knowing",
                                variable == 'Intuitive sense you are returning'|
                                  variable =='Intuitive sense you will return'~
                                  'Place/Space/State/Environment',
                                variable =='Sense of deja vu'|
                                  variable =='Feels like gone through before'|
                                  variable =='Feels like done this many times'|
                                  variable =='Feels like will happen again'~ 
                                  'Experience/Act',
                                variable =='Sense of visiting preconception/death'|
                                  variable =='Experience place that is eternal'|
                                  variable=='Experience place consciousness resides' ~
                                  'Transcendent Experience',
                                variable=='Encounter with familiar entity - established bond'|
                                  variable =='Encounter with entity - had bond with you'|
                                  variable=='Encounter with entity - feels like family'~
                                  'Entity Encounter'))%>%
  modify_table_styling(rows = variable == "Feeling/Emotion/Knowing", text_format = c('bold'))%>%
  as_gt()%>%
  gt::gtsave(filename='tblx.html')


#Supplementary methods (Found in the paper supplement)
#3 class
#derive from permuted array
p_3class <- as.data.frame(mcmc_permuted_3class[,1,])
colnames(p_3class) <- c("Class 1",'Class 2','Class 3',
                        rep('Sense of familiarity',times = 3),
                        rep('Feels like home',times = 3),
                        rep('Sense of comfort',times= 3),
                        rep('Sense of welcoming',times = 3),
                        rep('Sense of belonging',times = 3),
                        rep('Sense of nostalgia',times = 3),
                        rep('Sense of remembering...',times = 3),
                        rep('Intuitive sense you are returning',times = 3),
                        rep('Intuitive sense you will return', times = 3),
                        rep('Sense of deja vu',times = 3),
                        rep('Feels like gone through before',times = 3),
                        rep('Feels like done this many times',times =3),
                        rep('Feels like will happen again',times = 3),
                        rep('Sense of visiting preconception/death',times = 3),
                        rep('Experience place that is eternal',times = 3),
                        rep('Experience place consciousness resides',times = 3),
                        rep('Encounter with familiar entity - established bond',times = 3),
                        rep('Encounter with entity - had bond with you',times = 3),
                        rep('Encounter with entity - feels like family',times = 3))

#separate because of double column names, gather, and re-bind
p3_class1 <- p_3class[c(seq(1,ncol(p_3class),by = 3))]
p3_class2 <- p_3class[c(seq(2,ncol(p_3class),by = 3))]
p3_class3 <- p_3class[c(seq(3,ncol(p_3class),by = 3))]

p3_class1 <- gather(p3_class1,key = 'Items', value = 'Probability of Answering Yes')
p3_class2 <- gather(p3_class2,key = 'Items', value = 'Probability of Answering Yes')
p3_class3 <- gather(p3_class3,key = 'Items', value = 'Probability of Answering Yes')

p3_class_df <- rbind.data.frame(p3_class1,p3_class2,p3_class3)
mean(p_3class$`Class 1`)
mean(p_3class$`Class 2`)
mean(p_3class$`Class 3`)

p3_class_df$class <- rep(c('Class 1 (23%)','Class 2 (3%)','Class 3 (23%)'), each = 600000)
p3_class_df$category <- NA

colnames(p_3class)
p3_class_df$category[p3_class_df$Items%in%colnames(p_3class)[c(4,7,10,13,16,19,22)]]<- 
  'Feeling/Emotion/Knowledge'
p3_class_df$category[p3_class_df$Items%in%colnames(p_3class)[c(25,28)]]<- 
  'Place/Space/State/Environment'
p3_class_df$category[p3_class_df$Items%in%colnames(p_3class)[c(31,34,37,40)]]<- 
  'Experience/Act'
p3_class_df$category[p3_class_df$Items%in%colnames(p_3class)[c(43,46,49)]]<- 
  'Transcendent Experience'
p3_class_df$category[p3_class_df$Items%in%colnames(p_3class)[c(52,55,58)]]<- 
  'Entity Encounter'
p3_class_df$Items <- factor(p3_class_df$Items, 
                            levels=colnames(p_3class)[c(seq(1,ncol(p_3class),by = 3))])
p3_class_df$category <- factor(p3_class_df$category,
                               levels = c('Feeling/Emotion/Knowledge',
                                          'Place/Space/State/Environment',
                                          'Experience/Act',
                                          'Transcendent Experience',
                                          'Entity Encounter' ))

#Plot
theme_set(theme_tidybayes() + panel_border())
fig2<- p3_class_df[!(is.na(p3_class_df$category)),] %>%
  ggplot(aes(y = fct_rev(Items),
             x = `Probability of Answering Yes`))+
  geom_vline(xintercept = 0.5,alpha= 0.4,linetype =2 ) +
  geom_vline(xintercept = 0.25,alpha= 0.4,linetype =2 ) +
  geom_vline(xintercept = 0.75,alpha= 0.4,linetype =2 ) +
  theme(strip.text.y = element_text(face = 'bold',size = 7),
        strip.text.x = element_text(face = 'bold',size = 9),
        axis.title.x = element_text(size = 9,face = 'bold'),
        axis.title.y = element_text(size = 9,face = 'bold'),
        axis.text.y = element_text(face = 'bold',size = 8),
        legend.position = 'none',
        legend.text = element_text(face = 'bold',size = 10),
        axis.text.x = element_text(face = 'bold',size = 10))+
  ylab("Items") + 
  stat_gradientinterval(.width = c(0.70,0.9),
                        fill = 'blue',colour = 'black') +
  facet_grid(category~class,scales = 'free')
fig2
ggsave('fig2.jpg',fig2,dpi = 600,width = 10,height = 9)

#4 class
#derive from permuted array
p_4class <- as.data.frame(mcmc_permuted_4class[,1,])
colnames(p_4class) <- c("Class 1",'Class 2','Class 3','Class 4',
                        rep('Sense of familiarity',times = 4),
                        rep('Feels like home',times = 4),
                        rep('Sense of comfort',times= 4),
                        rep('Sense of welcoming',times = 4),
                        rep('Sense of belonging',times = 4),
                        rep('Sense of nostalgia',times = 4),
                        rep('Sense of remembering...',times = 4),
                        rep('Intuitive sense you are returning',times = 4),
                        rep('Intuitive sense you will return', times = 4),
                        rep('Sense of deja vu',times = 4),
                        rep('Feels like gone through before',times = 4),
                        rep('Feels like done this many times',times =4),
                        rep('Feels like will happen again',times = 4),
                        rep('Sense of visiting preconception/death',times = 4),
                        rep('Experience place that is eternal',times = 4),
                        rep('Experience place consciousness resides',times = 4),
                        rep('Encounter with familiar entity - established bond',times = 4),
                        rep('Encounter with entity - had bond with you',times = 4),
                        rep('Encounter with entity - feels like family',times = 4))

#separate because of double column names, gather, and re-bind
p4_class1 <- p_4class[c(seq(1,ncol(p_4class),by = 4))]
p4_class2 <- p_4class[c(seq(2,ncol(p_4class),by = 4))]
p4_class3 <- p_4class[c(seq(3,ncol(p_4class),by = 4))]
p4_class4 <- p_4class[c(seq(4,ncol(p_4class),by = 4))]

p4_class1 <- gather(p4_class1,key = 'Items', value = 'Probability of Answering Yes')
p4_class2 <- gather(p4_class2,key = 'Items', value = 'Probability of Answering Yes')
p4_class3 <- gather(p4_class3,key = 'Items', value = 'Probability of Answering Yes')
p4_class4 <- gather(p4_class4,key = 'Items', value = 'Probability of Answering Yes')

p4_class_df <- rbind.data.frame(p4_class1,p4_class2,p4_class3,p4_class4)
mean(p_4class$`Class 1`)
mean(p_4class$`Class 2`)
mean(p_4class$`Class 3`)
mean(p_4class$`Class 4`)

p4_class_df$class <- rep(c('Class 1 (3%)',
                           'Class 2 (8%)',
                           'Class 3 (22%)',
                           'Class 4 (67%)'), each = 600000)
p4_class_df$category <- NA

colnames(p_4class)
p4_class_df$category[p4_class_df$Items%in%colnames(p_4class)[c(5,9,13,17,21,25,29)]]<- 
  'Feeling/Emotion/Knowledge'
p4_class_df$category[p4_class_df$Items%in%colnames(p_4class)[c(33,37)]]<- 
  'Place/Space/State/Environment'
p4_class_df$category[p4_class_df$Items%in%colnames(p_4class)[c(41,45,49,53)]]<- 
  'Experience/Act'
p4_class_df$category[p4_class_df$Items%in%colnames(p_4class)[c(57,61,65)]]<- 
  'Transcendent Experience'
p4_class_df$category[p4_class_df$Items%in%colnames(p_4class)[c(69,73,77)]]<- 
  'Entity Encounter'
p4_class_df$Items <- factor(p4_class_df$Items, 
                            levels=colnames(p_4class)[c(seq(1,ncol(p_4class),by = 4))])
p4_class_df$category <- factor(p4_class_df$category,
                               levels = c('Feeling/Emotion/Knowledge',
                                          'Place/Space/State/Environment',
                                          'Experience/Act',
                                          'Transcendent Experience',
                                          'Entity Encounter' ))

#Plot
theme_set(theme_tidybayes() + panel_border())
fig3<- p4_class_df[!(is.na(p4_class_df$category)),] %>%
  ggplot(aes(y = fct_rev(Items),
             x = `Probability of Answering Yes`))+
  geom_vline(xintercept = 0.5,alpha= 0.4,linetype =2 ) +
  geom_vline(xintercept = 0.25,alpha= 0.4,linetype =2 ) +
  geom_vline(xintercept = 0.75,alpha= 0.4,linetype =2 ) +
  theme(strip.text.y = element_text(face = 'bold',size = 7),
        strip.text.x = element_text(face = 'bold',size = 9),
        axis.title.x = element_text(size = 9,face = 'bold'),
        axis.title.y = element_text(size = 9,face = 'bold'),
        axis.text.y = element_text(face = 'bold',size = 8),
        legend.position = 'none',
        legend.text = element_text(face = 'bold',size = 10),
        axis.text.x = element_text(face = 'bold',size = 10))+
  ylab("Items") + 
  stat_gradientinterval(.width = c(0.70,0.9),
                        fill = 'blue',colour = 'black') +
  facet_grid(category~class,scales = 'free')
fig3
ggsave('fig3.jpg',fig3,dpi = 600,width = 12,height = 9)




