install.packages("e1071")                 # download the e1071 library
install.packages("SparseM")               # download the SparseM library
library(e1071)
library(SparseM)                          # load the libraries
train <- read.csv("C:/Users/rezkar/Downloads/letterdata.csv" )

df2 <- transform(df, id=match(sample, unique(sample)))
dim(train)
# load the csv dataset into memory
train$letter <- as.numeric(as.factor(train$letter))     # convert the labels into numeric format 
                  # from any other format (int in my case)
x <- as.matrix(train)            # convert from data.frame format to 
                  # matrix format
y <- train[,17]                         # put the labels in a separate vector
xs <- as.matrix.csr(x)                  # convert to compressed sparse row format
write.matrix.csr(xs, y = y, file="C:/Users/rezkar/Downloads/Letterdata.data") # write the output libsvm format file
