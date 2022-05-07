##### The following R codes might be useful
##### when you conduct this take-home exam of ISyE 7406
#####
library(ggplot2)
library(caret)
library("randomForest")
library(gbm)
library("neuralnet")
library(e1071)
library(nnet)
library(mgcv)
library(lares)
#library(gam)

### Read Training Data
## Assume you save the training data in the folder "C:/temp" in your local laptop
traindata <- read.table(file = "ISyE7406train.csv", sep=",");
dim(traindata);
## dim=10000*202
## The first two columns are X1 and X2 values, and the last 200 columns are the Y valus

### Some example plots for exploratory data analysis
### please feel free to add more exploratory analysis
X1 <- traindata[,1];
X2 <- traindata[,2];

## note that muhat = E(Y) and Vhat = Var(Y)
muhat <- apply(traindata[,3:202], 1, mean);
Vhat  <- apply(traindata[,3:202], 1, var);

## You can construct a dataframe in R that includes all crucial
##    information for our exam
data0 = data.frame(X1 = X1, X2=X2, muhat = muhat, Vhat = Vhat);

## we can plot 4 graphs in a single plot
par(mfrow = c(2, 2));
plot(X1, muhat,col="blue");
plot(X2, muhat,col="blue");
plot(X1, Vhat,col="blue");
plot(X2, Vhat,col="blue");

hist(X1)
hist(X2)
hist(muhat)
hist(Vhat)

#correlation between predictor and target

corr_var(data0, # name of dataset
         muhat,  # name of variable to focus on
         ignore = 'Vhat'
)

corr_var(data0, # name of dataset
         Vhat,  # name of variable to focus on
         ignore = 'muhat'
)
## Or you can first create an initial plot of one line
##         and then iteratively add the lines
##
##   below is an example to plot X1 vs. muhat for different X2 values
##
## let us reset the plot
dev.off()
##
## now plot the lines one by one for each fixed X2
##
flag <- which(data0$X2 == 0);
plot(data0[flag,1], data0[flag, 3], type="l",
     xlim=range(data0$X1), ylim=range(data0$muhat), xlab="X1", ylab="muhat");
for (j in 1:99){
  flag <- which(data0$X2 == 0.01*j);
  lines(data0[flag,1], data0[flag, 3]);
}

## You can also plot figures for each fixed X1 or for Vhat

### Then you need to build two models:
##  (1) predict muhat from X1 and X2
##  (2) predict Vhat from X1 and X2.

## split train data into train and test
smp_size <- floor(0.75 * nrow(data0))
set.seed(123)
train_ind <- sample(seq_len(nrow(data0)), size = smp_size)
data0_train <- data0[train_ind, ]
data0_test <- data0[-train_ind, ]

#model_reg <- lm(muhat~X1+X2, data= data0_train)
#model_reg<- cv.lm(df = data0_train, form.lm = formula(muhat~X1+X2), m=3, dots = 
#        FALSE, seed=29, plotit=TRUE, printit=TRUE)

tc_lm <- trainControl(method = "cv", number = 5)
model_reg <- train(muhat~X1+X2, data = data0_train, method = "lm",
                trControl = tc_lm) # here
summary(model_reg)
reg_mu <- predict(model_reg,newdata = data0_test[,1:2])
MSE_reg_mu <- sum((data0_test$muhat-reg_mu)**2)/nrow(data0_test)

ggplot(data=data0_train, aes(x=fitted(model_reg), y=(muhat-fitted(model_reg)))) +
       geom_point(alpha=I(0.4),color='darkorange') +
       xlab('Fitted Values') +
       ylab('Residuals') +
       ggtitle('Residual Plot lm muhat') +
       geom_hline(yintercept=0)

model_nl1<- train(muhat~X1+X2+I(X1^2)+I(X2^2), data = data0_train, method = "lm",
                  trControl = tc_lm)
model_nl2<- train(muhat~X1+X2+sqrt(X1), data = data0_train, method = "lm",
                  trControl = tc_lm)

reg_nl_fitted <- predict(model_nl2,newdata = data0_train[,1:2])
reg_nl_resid <- data0_train[,3]-reg_nl_fitted
ggplot(data=data0_train, aes(x=reg_nl_fitted, y=reg_nl_resid)) +
  geom_point(alpha=I(0.4),color='darkorange') +
  xlab('Fitted Values') +
  ylab('Residuals') +
  ggtitle('Residual Plot polynomial lm') +
  geom_hline(yintercept=0)
#Better spread. We will check MSE
reg_nl1 <- predict(model_nl1,newdata = data0_test[,1:2])
MSE_nl1 <- mean((data0_test$muhat-reg_nl1)**2)
reg_nl2 <- predict(model_nl2,newdata = data0_test[,1:2])
MSE_nl2 <- mean((data0_test$muhat-reg_nl2)**2)
##nl2 is the best.. residual plot for this one.
##try loess as the relationship looks nonlinear
te_loess = NULL
spanlist=seq(0.15,0.85, by=.1)
for (i in spanlist) {
  model_loess <- loess(muhat~X1+X2,data0_train,span=i)
  loess_mu <- predict(model_loess,newdata = data0_test[,1:2])
  MSE_loess_mu <- mean((data0_test$muhat-loess_mu)**2)
  te_loess = c(te_loess, MSE_loess_mu)
}
te_loess
MSE_loess_mu = min(te_loess)
#best span = 0.15
#The spread looks much better

#Try random forest
model_rf <- randomForest(muhat~X1+X2, data=data0_train,importance = TRUE)
model_rf
rf_mu <- predict(model_rf,newdata = data0_test[,1:2])
MSE_rf_mu <- mean((data0_test$muhat-rf_mu)**2)

#gbm

model_gbm <- gbm(muhat ~ X1+X2,data=data0_train,
                       n.trees = 5000, 
                       shrinkage = 0.01, 
                       interaction.depth = 6,
                       cv.folds = 5)

model_gbm
summary(model_gbm)
#optimal number of iterations
bestiter=gbm.perf(model_gbm, method="cv") 
gbm_mu <- predict(model_gbm,newdata = data0_test[,1:2],n.trees = bestiter)
MSE_gbm_mu <- mean((data0_test$muhat-gbm_mu)**2)


#Neural network
# First, tune using 10-fold cv
library("neuralnet")
library("nnet")
#model_nn <- neuralnet(muhat ~ X1+X2, data= data0_train,hidden=5)
norm.fun = function(x){ 
  (x - min(x))/(max(x) - min(x)) 
}

data0_train.norm = as.data.frame(lapply(data0_train, norm.fun))
data0_test.norm = as.data.frame(lapply(data0_test, norm.fun))

#model_nn <- neuralnet(muhat~X1+X2, data0_train.norm, hidden = 2)

te_nn = NULL
te_nn_decay = NULL
sizel=seq(2,15, by=1)
decayl= c(0,0.1,0.01,0.001)
for (j in decayl){
  for (i in sizel) {
    model_nn <- nnet(muhat~X1+X2,data=data0_train.norm,size=i, decay=j,trace=FALSE)
    nn_mu <- predict(model_nn,newdata = data0_test.norm[,1:2])
    #de-normalize predicted data
    nn_mu_denorm = min(data0_test[, 'muhat']) + nn_mu * diff(range(data0_test[, 'muhat']))
    # MSE with data in the original scale
    MSE_nn_mu <- mean((data0_test$muhat-nn_mu_denorm)**2)
    te_nn = c(te_nn, MSE_nn_mu)
  }
  te_nn_decay = c(te_nn_decay, min(MSE_nn_mu))
  cat("\n best node size for decay ", j, "is ", which.min(te_nn)+1)
  cat("\n Minimum MSE for decay ", j, "is ", min(te_nn))
  te_nn = NULL
  
}
MSE_nn_mu = min(te_nn_decay)
#minimum MSE 1.915905
#best size = 7 and best decay = 0
te_dnn=NULL
#Try DNN with multiple hidden layers
# for (i in 1:7){
#   for (j in 1:7){
#     print(c(i,j))
#     model_dnn <- neuralnet(muhat~X1+X2,data=as.matrix(data0_train.norm), hidden=c(i,j), threshold = 0.01)
#     dnn_mu <- predict(model_dnn,newdata = as.matrix(data0_test.norm[,1:2]))
#     #de-normalize predicted data
#     dnn_mu_denorm = min(data0_test[, 'muhat']) + dnn_mu * diff(range(data0_test[, 'muhat']))
#     # MSE with data in the original scale
#     MSE_dnn_mu <- mean((data0_test$muhat-nn_mu_denorm)**2)
#     if (MSE_dnn_mu< min(te_dnn)){
#       cat("\n best node size is ", i, ",", j)
#     te_dnn = c(te_dnn, MSE_dnn_mu)
# 
#     }
# 
#   }
# }

model_dnn <- neuralnet(muhat~X1+X2,data=data0_train.norm, hidden=c(4,3), threshold = 0.01)
dnn_mu <- predict(model_dnn,newdata = data0_test.norm[,1:2])
#de-normalize predicted data
nn_mu_denorm = min(data0_test[, 'muhat']) + dnn_mu * diff(range(data0_test[, 'muhat']))
# MSE with data in the original scale
MSE_dnn_mu <- mean((data0_test$muhat-nn_mu_denorm)**2)
#GAM for muhat

model_gam <- gam(muhat~X1+X2+s(X1)+s(X2),data=data0_train,method="REML",select=TRUE)
summary(model_gam)

gam_muhat <- predict(model_gam,newdata = data0_test[,1:2])
MSE_gam_mu <- mean((data0_test$muhat-gam_muhat)**2)

#knn
model_knn <- train(
  muhat ~X1+X2, data = data0_train, method = "knn",
  trControl = trainControl("cv", number = 10),
  preProcess = c("center", "scale"),
  tuneLength = 15
)
# Plot model accuracy vs different values of k
plot(model_knn)

# Print the best tuning parameter k that
# maximizes model accuracy
model_knn$bestTune
#check accuracy of predictions
knn_muhat <- predict(model_knn, newdata = data0_test)
MSE_knn_muhat <- mean((data0_test$muhat-knn_muhat)**2)

## Fit models for Vhat

#1. Base regression model even though we know the relationship may not be linear.
model_reg_Vhat <-  lm(Vhat ~ X1+X2, data=data0_train)
reg_Vhat <- predict(model_reg_Vhat, newdata = data0_test[,1:2])
MSE_reg_Vhat <- mean((data0_test$Vhat-reg_Vhat)**2)

#polynomial regression
model_nl1_Vhat<- train(Vhat~X1+X2+I(X1^2)+I(X2^2), data = data0_train, method = "lm",
                  trControl = tc_lm)
model_nl2_Vhat<- train(Vhat~X1+X2+sqrt(X1), data = data0_train, method = "lm",
                  trControl = tc_lm)
nl1_Vhat <- predict(model_nl1_Vhat, newdata = data0_test[,1:2])
MSE_nl1_Vhat <- mean((data0_test$Vhat-nl1_Vhat)**2)
nl2_Vhat <- predict(model_nl2_Vhat, newdata = data0_test[,1:2])
MSE_nl2_Vhat <- mean((data0_test$Vhat-nl2_Vhat)**2)

te_loess_Vhat = NULL
for (i in spanlist) {
  model_loess_Vhat <- loess(Vhat~X1+X2,data0_train,span=i)
  loess_Vhat <- predict(model_loess_Vhat,newdata = data0_test[,1:2])
  MSE_loess_Vhat <- mean((data0_test$Vhat-loess_Vhat)**2)
  te_loess_Vhat = c(te_loess_Vhat, MSE_loess_Vhat)
}
te_loess_Vhat
#best span = 0.15
model_loess_Vhat <- loess(Vhat~X1+X2,data0_train,span=0.15)
loess_fitted_Vhat <- predict(model_loess_Vhat,newdata = data0_train[,1:2])
loess_resid_Vhat <- data0_train[,4]-loess_fitted_Vhat
ggplot(data=data0_train, aes(x=loess_fitted_Vhat, y=loess_resid_Vhat)) +
  geom_point(alpha=I(0.4),color='darkorange') +
  xlab('Fitted Values') +
  ylab('Residuals') +
  ggtitle('Residual Plot Loess Vhat') +
  geom_hline(yintercept=0)
#the residual plot shows heteroschedasticity. 
#gam

model_gam_Vhat <- gam(Vhat~X1+X2+s(X1)+s(X2),data=data0_train)
summary(model_gam_Vhat)
gam_fitted_Vhat <- predict(model_gam_Vhat,newdata = data0_train[,1:2],method="REML",select=TRUE)
gam_resid_Vhat <- data0_train[,4]-gam_fitted_Vhat
ggplot(data=data0_train, aes(x=gam_fitted_Vhat, y=gam_resid_Vhat)) +
  geom_point(alpha=I(0.4),color='darkorange') +
  xlab('Fitted Values') +
  ylab('Residuals') +
  ggtitle('Residual Plot gam Vhat') +
  geom_hline(yintercept=0)
#residual plot shows pattern. tried with combinations of nonlinear and linear combinations of X1 and X2
gam_Vhat <- predict(model_gam_Vhat,newdata = data0_test[,1:2])
MSE_gam_Vhat <- mean((data0_test$Vhat-gam_Vhat)**2)
par(mfrow = c(2,2))
plot(model_gam, se = TRUE)
plot(model_gam_Vhat, se = TRUE)
# Random forest
model_rf_Vhat <- randomForest(Vhat~X1+X2, data=data0_train,importance = TRUE)
model_rf_Vhat
rf_Vhat <- predict(model_rf_Vhat,newdata = data0_test[,1:2])
MSE_rf_Vhat <- mean((data0_test$Vhat-rf_Vhat)**2)

#gbm
model_gbm_Vhat <- gbm(Vhat ~ X1+X2,data=data0_train,
                 n.trees = 5000, 
                 shrinkage = 0.01, 
                 interaction.depth = 2,
                 cv.folds = 5)

model_gbm_Vhat
summary(model_gbm_Vhat)
#optimal number of iterations
gbm.perf(model_gbm_Vhat, method="cv") 
gbm_Vhat <- predict(model_gbm_Vhat,newdata = data0_test[,1:2],n.trees = 4846)
MSE_gbm_Vhat <- mean((data0_test$Vhat-gbm_Vhat)**2)

#Neural Network for Vhat
te_nn_Vhat = NULL
te_nn_decay_Vhat = NULL
sizel=seq(2,15, by=1)
decayl= c(0,0.1,0.01,0.001)
for (j in decayl){
  for (i in sizel) {
    model_nn_Vhat <- nnet(Vhat~X1+X2,data=data0_train.norm,size=i, decay=j,trace=FALSE)
    nn_Vhat <- predict(model_nn_Vhat,newdata = data0_test.norm[,1:2])
    #de-normalize predicted data
    nn_Vhat_denorm = min(data0_test[, 'Vhat']) + nn_Vhat * diff(range(data0_test[, 'Vhat']))
    # MSE with data in the original scale
    MSE_nn_Vhat <- mean((data0_test$Vhat-nn_Vhat_denorm)**2)
    te_nn_Vhat = c(te_nn_Vhat, MSE_nn_Vhat)
  }
  te_nn_decay_Vhat = c(te_nn_decay_Vhat, min(te_nn_Vhat))
  cat("\n best node size for decay for Vhat ", j, "is ", which.min(te_nn_Vhat)+1)
  cat("\n Minimum MSE for decay for Vhat ", j, "is ", min(te_nn_Vhat))
  te_nn_Vhat = NULL
  
}
MSE_nn_Vhat = min(te_nn_decay_Vhat)

#DNN
model_dnn_Vhat <- neuralnet(Vhat~X1+X2,data=as.matrix(data0_train.norm), hidden=c(4,3), threshold = 0.05)
dnn_Vhat <- predict(model_dnn_Vhat,newdata = as.matrix(data0_test.norm[,1:2]))
#de-normalize predicted data
nn_Vhat_denorm = min(data0_test[, 'Vhat']) + dnn_Vhat * diff(range(data0_test[, 'Vhat']))
# MSE with data in the original scale
MSE_dnn_Vhat <- mean((data0_test$Vhat-nn_Vhat_denorm)**2)
#knn_Vhat
model_knn_Vhat <- train(
  Vhat ~X1+X2, data = data0_train, method = "knn",
  trControl = trainControl("cv", number = 10),
  preProcess = c("center", "scale"),
  tuneLength = 15
)
# Plot model accuracy vs different values of k
plot(model_knn_Vhat)

# Print the best tuning parameter k that
# maximizes model accuracy
model_knn_Vhat$bestTune
#check accuracy of predictions
knn_Vhat <- predict(model_knn_Vhat, newdata = data0_test)
MSE_knn_Vhat <- mean((data0_test$Vhat-knn_Vhat)**2)



## Testing Data: first read testing X variables
testX  <- read.table(file = "ISyE7406test.csv", sep=",");
dim(testX)
## This should be a 2500*2 matrix

## Next, based on your models, you predict muhat and Vhat for (X1, X2) in textX.
## Suppose that will lead you to have a new data.frame
##   "testdata" with 4 columns, "X1", "X2", "muhat", "Vhat"
## The best model obtained for myhat and Vhat was LOESS with a span of 0.15.
##I will use this model to predict muhat and Vhat for test data
colnames(testX) <- c('X1','X2')
testdata <- data.frame(testX)

testdata$muhat <- predict(model_loess, newdata=testX)
testdata$Vhat <- predict(model_loess_Vhat, newdata=testX)

#We will look at the plots again to compare with the original set of plots
# created to examine relationship between predictor and response variables
## we can plot 4 graphs in a single plot
par(mfrow = c(2, 2));
plot(testdata$X1, testdata$muhat);
plot(testdata$X2, testdata$muhat);
plot(testdata$X1, testdata$Vhat);
plot(testdata$X2, testdata$Vhat);

## Then you can write them in the csv file as follows:
## Then you can upload the .csv file to the Canvas
## (please use your own Last Name and First Name)
##  
write.table(testdata, file="C:/temp/1.Jacob.Tinju.csv",
            sep=",",  col.names=F, row.names=F)

## Note that in your final answers, you essentially add two columns for your estimation of
##     $mu(X1,X2)=E(Y)$ and $V(X1, X2)=Var(Y)$
##  to the testing  X data file "7406test.csv".
## Please save your predictions as a 2500*4 data matrix
##     in a .csv file "without" headers or extra columns/rows.

