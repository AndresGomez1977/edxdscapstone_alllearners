###################################
# Create edx set and validation set
###################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

#set.seed(1) # if using version lower than R 3.6.0: 
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

############ edx and validation data frames are created ######################
#####Data exploration
## Let's start answering the quiz
#Q1
dim(edx)
#Q2
edx %>% filter(rating == 3) %>% tally()  
#Q3 & Q4
n_distinct(edx$movieId)
n_distinct(edx$userId)
#Q5
edx %>% separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(count = n()) %>%
  arrange(desc(count))
#Q6
edx %>% group_by(movieId, title) %>%
  summarize(count = n()) %>%
  arrange(desc(count))
#Q7
edx %>% group_by(rating) %>% summarize(count = n()) %>% top_n(5) %>%
  arrange(desc(count))
#Q8
edx %>%
  group_by(rating) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = rating, y = count)) +
  geom_line()
#End quiz
#more data exploration
#histogram of ratings
library(dslabs)
library(scales)
edx %>% ggplot()+
  geom_histogram(mapping=aes(x=rating), binwidth = 0.5, color="blue", fill="cyan")+
  xlab("Rating")+ylab("Frecuency")+scale_y_continuous(labels = number)

#let's prepare data and rmse function to build the models
library(caret)
library(knitr)
set.seed(1, sample.kind = "Rounding")

train_set <- edx
test_set <- validation

test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
} ##this will be our rmse funtion that will evaluate the model on the test set

### Building the Recommendation System

## First approach: naive model.

#train
mu_hat <- mean(train_set$rating)
mu_hat
#evaluate on the test set
naive_rmse <- RMSE(test_set$rating, mu_hat)
naive_rmse

## Second approach: movie bias

#train
mu <- mean(train_set$rating) 
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i =mean(rating - mu))

qplot(b_i, data = movie_avgs, bins = 10, color = I("black")) ## plot estimates to see variations

#evaluate on the test 
predicted_ratings <- test_set %>%
  left_join(movie_avgs, by = "movieId") %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)
Movie_Effect_rmse <- RMSE(predicted_ratings, test_set$rating)
Movie_Effect_rmse 

## Third approach: movie plus user effect

#train
user_avgs <- train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

#evaluate on the test
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
RMSE_Movie_Users <- RMSE(predicted_ratings, test_set$rating)
RMSE_Movie_Users

## Fourth approach: Penalized least squares

#train
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  mu <- mean(train_set$rating)
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- train_set %>%
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  #evaluate on test set
  predicted_ratings <-test_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  return(RMSE(predicted_ratings, test_set$rating))
})

#show lamda that optimizes model
lambda <- lambdas[which.min(rmses)]
lambda

#get rmse 
min(rmses)



