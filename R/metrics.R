#' @title Normalized entropy calculator
#' @name NormalizedEntropy
#' @description Calculate normalized entropy, i.e. binary
#' cross entropy divided by entropy of background probability of positive class label;
#' defined in He, Pan, et al 2014
#' @param trueVec numeric vector of true binary class labels
#' @param predVec numeric vector of binary class probability prediction values
#' @return numeric vector of normalized cross entropy values
#' @export
NormalizedEntropy <- function(trueVec, predVec){
  if(length(trueVec) != length(predVec)){
    stop('Actual and predicted vectors need to have the same length')
  }
  
  crossEntropy <- -((1 + trueVec) / 2) * log(predVec) - ((1 - trueVec) / 2) * log(1 - predVec)
  ctr <- mean(trueVec == 1)
  ctrEntropy <- -ctr * log(ctr) - (1 - ctr) * log(1 - ctr)
  return(mean(crossEntropy) / ctrEntropy)
}

#' @title Calculate normalized entropy across samples, for use with caret::trainControl (for use with caret::train).
#' @name NormalizedEntropySummary
#' @description summary function to use normalized entropy metric in caret::train. Normalized (binary) entropy controls for
#' the background positive class rate, helping eliminate possible bias when class imbalance is severe.
#' 
#' Binary cross entropy (i.e. binary log loss) for a given set of predictions and 
#' ground-truth labels (assuming labeling scheme $y \in \{-1, 1}$) is given by $$LL = -\frac{1}{n}\sum_{i=1}^{n}(ylog(\hat{y}) + (1-y)\log(1-\hat{y}))$$
#' 
#' Meanwhile, normalized cross entropy simply divides binary cross entropy by the background positive class rate:
#' $$bg = \frac{1}{n}\sum_{i=1}^{n}I\{y_{i} == 1\}$$
#' $$NE = \frac{LL}{-bg\log(bg) - (1-bg)\log(1-bg)}$$
#'
#' @param data data.table provided by caret::train
#' @param lev str positive class label
#' @param model NULL required by caret::train
#' @return named list
#' @seealso ?caret::defaultSummary
#' @export
#' 
#' @examples
#' 
#' \dontrun{
#' require(caret)
#' require(data.table)
#' 
#' DT <- as.data.table(iris)
#' DT[, Species := ifelse(Species == 'setosa', 1, 0)]
#' 
#' # set up training control object.
#' trControl <- caret::trainControl(method = 'cv'
#' , number = 3
#' , verboseIter = TRUE
#' , classProbs = TRUE
#' , summaryFunction = NormalizedEntropySummary)
#' 
#' # caret makes you specify 7 tuning parameters for xgboost model (shruggie)
#' tuneGrid <- expand.grid(nrounds = c(5, 20)
#' , max_depth = c(3, 7)
#' , eta = c(0.2)
#' , gamma = c(1)
#' , colsample_bytree = c(0.7)
#' , subsample = c(0.5)
#' , min_child_weight = c(1))
#' 
#' # select training set, and format target vector (caret *very* picky about target vector)
#' trIdx <- sample(1:nrow(DT), 100)
#' target <- factor(DT[trIdx, get('Species')])
#' levels(target) <- c('negative', 'positive')
#' 
#' # run caret CV hyperparameter search.
#' cvResults <- caret::train(DT[trIdx, -'Species']
#' , y = target
#' , method = 'xgbTree'
#' , metric = 'NormalizedEntropy'
#' , trControl = trControl
#' , tuneGrid = tuneGrid
#' , maximize = FALSE)
#' }
#' 
#' @export
#' 
NormalizedEntropySummary <- function(data, lev = NULL, model = NULL){
  levels(data$obs) <- c('-1', '1')
  out <- NormalizedEntropy(as.numeric(levels(data$obs))[data$obs]
                           , predVec = data[, lev[2]])
  names(out) <- 'NormalizedEntropy'
  return(out)
}