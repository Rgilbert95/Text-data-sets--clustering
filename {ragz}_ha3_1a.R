# Creating the empty dataset with the formatted columns 
dataframe <- data.frame(ID=character(), 
                        datetime=character(), 
                        content=character(), 
                        label=factor()) 
source.url <- 'https://archive.ics.uci.edu/ml/machine-learning-databases/00438/Health-News-Tweets.zip' 
target.directory <- '/tmp/clustering-r' 
temporary.file <- tempfile() 
download.file(source.url, temporary.file) 
unzip(temporary.file, exdir = target.directory) 
# Reading the files 
target.directory <- paste(target.directory, 'Health-Tweets', sep = '/') 
files <- list.files(path = target.directory, pattern='.txt$') 
# Filling the dataframe by reading the text content 
for (f in files) { 
  news.filename = paste(target.directory , f, sep ='/') 
  news.label <- substr(f, 0, nchar(f) - 4) # Removing the 4 last characters => '.txt' 
  news.data <- read.csv(news.filename, 
                        encoding = 'UTF-8', 
                        header = FALSE, 
                        quote = "", 
                        sep = '|', 
                        col.names = c('ID', 'datetime', 'content')) 
  news.data <- news.data[news.data$content != "", ] 
  news.data['label'] = news.label # We add the label of the tweet  
  #Loading with a few of data
  news.data <- head(news.data, floor(nrow(news.data) * 0.05)) 
  dataframe <- rbind(dataframe, news.data) # Row appending 
} 
unlink(target.directory, recursive =  TRUE) # Deleting the temporary directory

#To get rid of these URLs

sentences <- sub("http://([[:alnum:]|[:punct:]])+", '', dataframe$content)


#To start with text mining

corpus = tm::Corpus(tm::VectorSource(sentences)) 

# Cleaning up 
# Handling UTF-8 encoding problem from the dataset 
#utf8text <- iconv(corpus, to='UTF-8', sub = "byte")
corpus.cleaned <- tm::tm_map(corpus, function(x) iconv(x, to='UTF-8', sub='byte'))  
corpus.cleaned <- tm::tm_map(corpus.cleaned, tm::removeWords, tm::stopwords('english')) # Removing stop-words 
corpus.cleaned <- tm::tm_map(corpus, tm::stemDocument, language = "english") 
corpus.cleaned <- tm::tm_map(corpus.cleaned, tm::stripWhitespace) # Trimming excessive whitespaces

# Building the feature matrices
tdm <- tm::DocumentTermMatrix(corpus.cleaned)
tdm.tfidf <- tm::weightTfIdf(tdm)
# We remove A LOT of features. R is natively very weak with high dimensional matrix
tdm.tfidf <- tm::removeSparseTerms(tdm.tfidf, 0.999)
# There is the memory-problem part
# - Native matrix isn't "sparse-compliant" in the memory
# - Sparse implementations aren't necessary compatible with clustering algorithms
tfidf.matrix <- as.matrix(tdm.tfidf)
# Cosine distance matrix (useful for specific clustering algorithms)
dist.matrix = proxy::dist(tfidf.matrix, method = "cosine")

#RUNNING K MEANS
truth.K <- 16
clustering.kmeans <- kmeans(tfidf.matrix, truth.K)


#PLOT
points <- cmdscale(dist.matrix, k = 2)
master.cluster <- clustering.kmeans$cluster

plot(points,
     main = 'K-Means clustering',
     col = as.factor(master.cluster))

#CENTROIDS

dfCluster<-kmeans(tfidf.matrix,centers=3, iter.max = 1)
plot(points[,1], points[,2], col=dfCluster$cluster,pch=19,cex=2, main="iter 1")
points(dfCluster$centers,col=1:5,pch=3,cex=3,lwd=3)

#Loop for iteration and new plots

max_iter = 100

for (i in 2:max_iter){
  tryCatch({
    dfCluster <- kmeans(tfidf.matrix,centers = dfCluster$centers, iter.max = 1)
    done <- TRUE
  }, 
  warning=function(w) {done <- FALSE})
  plot(points[,1], points[,2], col=dfCluster$cluster,pch=19,cex=2, main=paste("iter",i))
  points(dfCluster$centers,col=1:5,pch=3,cex=3,lwd=3)
  if(done) break
}

