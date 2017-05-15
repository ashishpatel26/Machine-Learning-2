# Regression Template

# Importing the dataset

# Importing the dataset
dataset = read.csv('Data.csv')

head(dataset)

# Taking care of missing data

dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age)
dataset$Salary = ifelse(is.na(dataset$Salary),
                        ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                        dataset$Salary)

# Encoding categorical value
dataset$Country= factor(dataset$Country,levels=c('France','Spain','Germany'),labels=c(1,2,3))
dataset$Purchased = factor(dataset$Purchased,levels=c('Yes','No'),labels=c(1,0))


# Splitting the dataset into the Training set and Test set
# # install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Salary, SplitRatio = 2/3)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting the Regression Model to the dataset
# Create your regressor here
# Linear regression
reg<-lm(formula=Salary ~ YearsExperience,data=dataset)

y_pred<- predict(reg,newdata=test_set)
                         
                            
 
# Predicting a new result
y_pred = predict(regressor, data.frame(Level = 6.5))

                            
 # Multiple Regression
  reg<-lm(formula=Profit ~.,data=dataset)
summary(reg)
y_pred<- predict(reg,newdata=test_set)

# backward elimination
regressor<-lm(formula=Profit ~R.D.Spend+Administration+Marketing.Spend+State,data=dataset)
print(summary(regressor))
regressor<-lm(formula=Profit ~R.D.Spend+Administration+Marketing.Spend,data=dataset)
print(summary(regressor))
regressor<-lm(formula=Profit ~R.D.Spend+Marketing.Spend,data=dataset)
print(summary(regressor))
                            
   # Polynomial regression
 Fit Polynomial  Regression
# add new column which wil be level square
dataset$Level2<- dataset$Level^2
dataset$Level3<- dataset$Level^3
poly_reg<-lm(formula=Salary ~.,data=dataset)
summary(poly_reg)
                            
# Fitting SVR to the dataset
# install.packages('e1071')
library(e1071)
regressor = svm(formula = Salary ~ .,
                data = dataset,
                type = 'eps-regression',
                kernel = 'radial')

  # Fitting Decision Tree 
  
 library(rpart)
reg<-rpart(formula=Kyphosis ~.,data=dataset)

y_pred<- predict(reg,newdata=test_set)              
 # Fitting Random Forest Regression to the dataset
# install.packages('randomForest')
library(randomForest)
set.seed(1234)
regressor = randomForest(x = dataset[-2],
                         y = dataset$Salary,
                         ntree = 500)
                            
                            
                            
# Visualising the Regression Model results
# install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(regressor, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Regression Model)') +
  xlab('Level') +
  ylab('Salary')

# Visualising the Regression Model results (for higher resolution and smoother curve)
# install.packages('ggplot2')
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Regression Model)') +
  xlab('Level') +
  ylab('Salary')