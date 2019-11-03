#rmuh

library("factoextra")
library("Hmisc")
library(arules)
library(rpart)
library(rpart.plot)


#Choose the classification method which gives the best accuracy 


#Read the dataset with the number economic units by ageb
dat <- read.csv("runidades_clara3_ageb.csv", header = TRUE, colClasses=c("cluster"="factor"))

#Normalize data
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
dat1 <- as.data.frame(lapply(dat[,6:85], normalize))
dat1 <- cbind(dat[,1:5],dat1)


#--Outlier detection
library(OutlierDetection)
md.2 <- dens(dat1[,6:85],cutoff = 0.98)
print(md.2$`Location of Outlier`)
dat.pca <- prcomp(dat1[,6:85], center = TRUE)
plot(dat.pca$x[,1],dat.pca$x[,2])
points(dat.pca$x[md.2$`Location of Outlier`,], col="red")

plot(dat.pca$x[-md.2$`Location of Outlier`,1],dat.pca$x[-md.2$`Location of Outlier`,2])

plot(dat1[,"X811"],dat1[,"X465"])
points(dat1[md.2$`Location of Outlier`,], col="red")
#Data after outlier detection
dat1 = dat1[-md.2$`Location of Outlier`,]


#Randomly select a subset of the dataset for training and another for testing
set.seed(2411)
smp_size <- floor(0.70 * nrow(dat1))
train_ind <- sample(nrow(dat1), size = smp_size)
train.df <- as.data.frame(dat1[train_ind, ])
test.df <- as.data.frame(dat1[-train_ind, ])


#-------------------------- Random Forest ------------------------------------
library(randomForest)
#Create and train the model
model_rf <- randomForest(cluster ~ ., data = train.df[,5:85], importance = TRUE,ntree=700)
#print(model_rf)

pred_rf <- predict(model_rf, test.df[,6:85]) #Testing

#Print the confusion matrix
cm_rf = as.matrix(table(Actual = test.df$cluster, Predicted = pred_rf))
print(cm_rf)
n = sum(cm_rf) # number of instances
nc = nrow(cm_rf) # number of classes
diag = diag(cm_rf) # number of correctly classified instances per class 
rowsums = apply(cm_rf, 1, sum) # number of instances per class
colsums = apply(cm_rf, 2, sum) # number of predictions per class
p = rowsums / n # distribution of instances over the actual classes
q = colsums / n # distribution of instances over the predicted classes

#Accuracy of the model with the testing set
accuracy_rf = sum(diag) / n
print("rf test accuracy:")
print(accuracy_rf)

#Class precision, recall and f-score
precision_rf = diag / colsums 
recall_rf = diag / rowsums 
f1_rf = 2 * precision_rf * recall_rf / (precision_rf + recall_rf)
data.frame(precision_rf, recall_rf, f1_rf)

#Accuracy of the model with the training set
pred_rf_1 <- predict(model_rf, train.df[,6:85])
cm_rf_1 = as.matrix(table(Actual = train.df$cluster, Predicted = pred_rf_1))
accuracy_rf_1 = sum(diag(cm_rf_1)) / sum(cm_rf_1)
print("rf train accuracy:")
print(accuracy_rf_1)

#---------------------------- SVM ---------------------------------------------
library(e1071)

#Create and train the model
model_svm = svm(cluster~., data = train.df[,5:85])
pred_svm <- predict(model_svm, test.df[,6:85]) #Testing

#Print the confusion matrix
cm_svm = as.matrix(table(Actual = test.df$cluster, Predicted = pred_svm))
print(cm_svm)

n = sum(cm_svm) 
nc = nrow(cm_svm) 
diag = diag(cm_svm) 
rowsums = apply(cm_svm, 1, sum) 
colsums = apply(cm_svm, 2, sum) 
p = rowsums / n 
q = colsums / n 

#Accuracy of the model with the testing set
accuracy_svm = sum(diag) / n 
print("svm test accuracy:")
print(accuracy_svm)

#Class precision, recall and f-score
precision_svm = diag / colsums 
recall_svm = diag / rowsums 
f1_svm = 2 * precision_svm * recall_svm / (precision_svm + recall_svm)
data.frame(precision_svm, recall_svm, f1_svm)

#Accuracy of the model with the training set
pred_svm_1 <- predict(model_svm, train.df[,6:85])
cm_svm_1 = as.matrix(table(Actual = train.df$cluster, Predicted = pred_svm_1))
accuracy_svm_1 = sum(diag(cm_svm_1)) / sum(cm_svm_1)
print("svm train accuracy:")
print(accuracy_svm_1)

#--------------------------- GBM ----------------------------------------------
library(gbm)
#Create and train the model
model_gbm = gbm(cluster ~.,
                data = train.df[,5:85],n.trees=1000)

#Testing
pred_gbm = predict.gbm(object = model_gbm,
                       test.df[,6:85],n.trees=1000)

labels = colnames(pred_gbm)[apply(pred_gbm, 1, which.max)]

#Print the confusion matrix
cm_gbm = as.matrix(table(Actual = test.df$cluster, Predicted = as.factor(labels)))
cm_gbm

n = sum(cm_gbm) 
nc = nrow(cm_gbm) 
diag = diag(cm_gbm) 
rowsums = apply(cm_gbm, 1, sum) 
colsums = apply(cm_gbm, 2, sum) 
p = rowsums / 
q = colsums / n 

#Accuracy of the model with the testing set
accuracy_gbm = sum(diag) / n 
print("gbm test accuracy")
print(accuracy_gbm)

#Class precision, recall and f-score
precision_gbm = diag / colsums 
recall_gbm = diag / rowsums 
f1_gbm = 2 * precision_gbm * recall_gbm / (precision_gbm + recall_gbm)
data.frame(precision_gbm, recall_gbm, f1_gbm)

pred_gbm_1 = predict.gbm(object = model_gbm,
                       train.df[,6:85],n.trees=1000)
labels = colnames(pred_gbm_1)[apply(pred_gbm_1, 1, which.max)]
cm_gbm_1 = as.matrix(table(Actual = train.df$cluster, Predicted = as.factor(labels)))

#Accuracy of the model with the training set
accuracy_gbm_1 = sum(diag(cm_gbm_1)) / sum(cm_gbm_1)
print("gbm train accuracy:")
print(accuracy_gbm_1)

#----------------------------- Neural Networks -------------------------------------------
library(nnet)
#Create and train the model
mod_nnet = nnet(cluster~., data = train.df[,5:85], size=21,MaxNWts=9000)

pred_nnet <- predict(mod_nnet, test.df[,6:85], type="class") #Testing

#Print the confusion matrix
cm_nnet = as.matrix(table(Actual = test.df$cluster, Predicted = as.numeric(pred_nnet) ))
print(cm_nnet)

n = sum(cm_nnet) 
nc = nrow(cm_nnet) 
diag = diag(cm_nnet) 
rowsums = apply(cm_nnet, 1, sum) 
colsums = apply(cm_nnet, 2, sum) 
p = rowsums / n 
q = colsums / n 

#Accuracy of the model with the testing set
accuracy_nnet = sum(diag) / n
print("nnet test accuracy")
print(accuracy_nnet)

#Class precision, recall and f-score
precision_nnet = diag / colsums 
recall_nnet = diag / rowsums 
f1_nnet = 2 * precision_nnet * recall_nnet / (precision_nnet + recall_nnet)
data.frame(precision_nnet, recall_nnet, f1_nnet)

#Accuracy of the model with the training set
pred_nnet_1 <- predict(mod_nnet, train.df[,6:85], type="class")
cm_nnet_1 = as.matrix(table(Actual = train.df$cluster, Predicted = as.numeric(pred_nnet_1) ))
accuracy_nnet_1 = sum(diag(cm_nnet_1)) / sum(cm_nnet_1)
print("nnet train accuracy:")
print(accuracy_nnet_1)

#--------------------------- C5.0 Decision Tree --------------------------------
require(C50)
#Create and train the model
model_c5 = C5.0(train.df[,6:85],as.factor(train.df$cluster),trials = 10)
#summary(model_c5)

pred_c5 = predict(model_c5, test.df[,6:85]) #Testing

#Print the confusion matrix
cm_c5 = as.matrix(table(Actual = test.df$cluster, Predicted = pred_c5))
print(cm_c5)

n = sum(cm_c5) 
nc = nrow(cm_c5) 
diag = diag(cm_c5) 
rowsums = apply(cm_c5, 1, sum) 
colsums = apply(cm_c5, 2, sum) 
p = rowsums / n 
q = colsums / n 

#Accuracy of the model with the testing set
accuracy_c5 = sum(diag) / n 
print("c5 test accuracy")
print(accuracy_c5)

#Class precision, recall and f-score
precision_c5 = diag / colsums 
recall_c5 = diag / rowsums 
f1_c5 = 2 * precision_c5 * recall_c5 / (precision_c5 + recall_c5) 
data.frame(precision_c5, recall_c5, f1_c5)

#Accuracy of the model with the training set
pred_c5_1 = predict(model_c5, train.df[,6:85])
cm_c5_1 = as.matrix(table(Actual = train.df$cluster, Predicted = pred_c5_1))
accuracy_c5_1 = sum(diag(cm_c5_1)) / sum(cm_c5_1)
print("c5 train accuracy:")
print(accuracy_c5_1)

#---------------------------------------------------------------------
#Saving datsets with the best prediction

a <- test.df[[1]]
b <- test.df[[2]]
c <- test.df[[3]]
d <- test.df[[4]]
e <- test.df[[5]]
y <- pred_rf
df <- do.call(rbind, Map(data.frame, ENT=a, MUN=b, LOC=c, ageb=d, c_original=e, pred=y))
write.csv(df, file="rf_class_test.csv", row.names = FALSE)

a <- train.df[[1]]
b <- train.df[[2]]
c <- train.df[[3]]
d <- train.df[[4]]
e <- train.df[[5]]
y <- pred_rf_1
df <- do.call(rbind, Map(data.frame, ENT=a, MUN=b, LOC=c, ageb=d, c_original=e, pred=y))
write.csv(df, file="rf_class_train.csv", row.names = FALSE)