## Rename columns
install.packages("tm")
library(tm)
library(tictoc)

#dataset: https://github.com/udacity/machine-learning/blob/master/projects/practice_projects/naive_bayes_tutorial/smsspamcollection/SMSSpamCollection
sms_raw= read.csv("C:\\Users\\47406\\Downloads\\spam.csv", stringsAsFactors = FALSE)

str(sms_raw)
sms_raw$type <- factor(sms_raw$v1)
str(sms_raw$v1)
table(sms_raw$v1)

#Create the corpus

#> Loading required package: NLP
#converts the 'text' column (multibyte) to utf8 form 
sms_raw$v2 <- iconv(enc2utf8(sms_raw$v2),sub="byte")
sms_corpus <- VCorpus(VectorSource(sms_raw$v2))
print(sms_corpus)

#See some documents
inspect(sms_corpus[1:2])

#show some text
as.character(sms_corpus[[1]])
lapply(sms_corpus[1:3], as.character)

#Conversion of the text
sms_corpus_clean <- tm_map(sms_corpus, content_transformer(tolower))
as.character(sms_corpus[[1]])
as.character(sms_corpus_clean[[1]])
sms_corpus_clean <- tm_map(sms_corpus_clean, removeNumbers) #remove of some numbers

#Transformations
getTransformations()

#Now, remove stop words
# remove punctuation and stop words
sms_corpus_clean <- tm_map(sms_corpus_clean, removeWords, stopwords())

sms_corpus_clean <- tm_map(sms_corpus_clean, removePunctuation)

#Initialize stemming
library(SnowballC)
wordStem(c("learn", "learned", "learning", "learns"))
sms_corpus_clean <- tm_map(sms_corpus_clean, stemDocument)
sms_corpus_clean <- tm_map(sms_corpus_clean, stripWhitespace)

#Convert to Document
sms_dtm <- DocumentTermMatrix(sms_corpus_clean)
sms_dtm

#Data preprocessing
d1=dim(sms_dtm)[1]*0.75
d2=dim(sms_dtm)[1]*0.25

#text features
sms_dtm_train <- sms_dtm[1:d1, ]
sms_dtm_test <- sms_dtm[d1:dim(sms_dtm)[1], ]

#labels
sms_train_lab <- sms_raw[1:d1, ]$type
sms_test_lab <- sms_raw[d1:dim(sms_dtm)[1], ]$type

prop.table(table(sms_train_lab))
prop.table(table(sms_test_lab))

# convert dtm to matrix
sms_mat_train <- as.matrix(t(sms_dtm_train))
dtm.rs <- sort(rowSums(sms_mat_train), decreasing=TRUE)

# dataframe with word-frequency
dtm.df <- data.frame(word = names(dtm.rs), freq = as.integer(dtm.rs),
                     stringsAsFactors = FALSE)

#Naive bayes initialization

sms_freq_words <- findFreqTerms(sms_dtm_train, 5)
str(sms_freq_words)

sms_dtm_freq_train <- sms_dtm_train[ , sms_freq_words]
sms_dtm_freq_test <- sms_dtm_test[ , sms_freq_words]

# convert the matrix to “yes” and “no” categorical variables.
convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}

sms_train <- apply(sms_dtm_freq_train, MARGIN = 2, convert_counts)
sms_test <- apply(sms_dtm_freq_test, MARGIN = 2, convert_counts)

install.packages("e1071")
library(e1071)
sms_classifier <- naiveBayes(sms_train, sms_train_lab)

#Prediction and evaluation
sms_test_pred <- predict(sms_classifier, sms_test)

#FINAL
install.packages("gmodels")
library(gmodels)
CrossTable(sms_test_pred, sms_test_lab, prop.chisq = FALSE, prop.t = FALSE, dnn = c('predicted', 'actual'))

unclass(sms_test_pred)
unclass(sms_test_lab)


#PLOT 
install.packages("ggplot2")                          # Install & load ggplot2 package
library("ggplot2")
install.packages("ggpubr")
library("ggpubr")

#DATAFRAME FOR THE PREDICTED AND TEST VALUES
data_m=data.frame(actual= unclass(sms_test_lab)
                  , predicted=unclass(sms_test_lab))

#Take a  test sample of the independent variables 
x1=sms_dtm_test$i[1:nrow(data_m)]
x2=sms_dtm_test$j[1:nrow(data_m)]


#plot(sms_dtm_train$i[1:nrow(data_m)],sms_dtm_train$j[1:nrow(data_m)], col=scales::alpha(sms_dtm_train$v[1:nrow(data_m)],0.2), pch=17) #plotting train data
#points(sms_dtm_test$i[1:nrow(data_m)],sms_dtm_test$j[1:nrow(data_m)], col=scales::alpha(unclass((sms_test_lab)),0.2), pch=17) #plotting train data
#points(sms_dtm_test$i[1:nrow(data_m)],sms_dtm_test$j[1:nrow(data_m)], col=scales::alpha(unclass((sms_test_pred)),0.2), pch=17) #plotting train data


dt=data.frame(actual= x1, predicted=x2)

fg1=ggplot(dt, aes(x = x1, y = x2)) +
  geom_point(aes(color = as.factor(unclass(sms_test_lab))))

fg2=ggplot(dt, aes(x = x1, y = x2)) +
  geom_point(aes(color = as.factor(unclass(sms_test_pred))))

plot_list = list(fg1,fg2) 
ggarrange(plotlist=plot_list, labels = c("a", "b"))

plot(x1,x2, col=scales::alpha(unclass(sms_test_lab),0.2), pch=17) #plotting train data
points(x1,x2, col=scales::alpha(unclass(sms_test_pred),0.2), pch=19) #plotting train data


ggarrange(fg1, fg2, 
          labels = c("A", "B"),
          ncol = 1, nrow = 2)
