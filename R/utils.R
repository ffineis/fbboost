#' @name GetDefaultLevels
#' @description Default levels of data for CTR marketing data.
#' For use with a target vector whose classes are 'pass' and 'click', e.g.
#' in web advertising datasets.
GetDefaultLevels <- function(){
	return(c('pass', 'click'))
}


#' @name DetectCatVars
#' @description Find likely categorical variables within a data.frame or data.table.
#' @param x data.frame or data.table
#' @return character vector of column names likely corresponding to categorical variables
DetectCatVars <- function(x){
  return(names(x)[which(lapply(x, class) %in% c('factor', 'character'))])
}


#' @name PeelTargetVar
#' @description Peel off target vector from input features to a dataset.
#' @param x input data: matrix, data.frame, or data.table
#' @param y character
#' @return list of 'x' (matrix/data.frame/data.table of features) and target 'y', a vector.
#' @importFrom dplyr select pull
PeelTargetVar <- function(x, y){

  # fail out if y is vector-valued.
  if ((length(y) != 1)){
    stop('y must be either a single variable name or column index.')
  }
  
  # if y is the name of a column, find the integer column index.
  if (is.character(y)){
    if(!(y %in% names(x))){
      stop('supplied y "', y, '" is not the name of a column in x.')
    } else {
      y <- which(names(x) == y)[1]
    }
  # if y is a column index, check that the index is 1 <= y <= ncol(x)
  } else if (is.numeric(y)) {
    if(y < 1 | y > ncol(x)){
      stop('x only has ', ncol(x), ' columns... cannot select target column y = ', y)
    }
    y <- names(x)[y]
  } else {
    stop('supplied y "', y, '" is not a target variable name nor column index')
  }

  # can't use dplyr::select with matrix input.
  if (!'matrix' %in% class(x)){
    out <- list('x' = dplyr::select(x, -y)
                , 'y' = dplyr::pull(x, y))
  } else {
    dat <- list('x' = x[, setdiff(1:ncol(x), targetIdx)]
                , 'y' = x[, targetIdx]) 
  }

  return(out)
}


#' @name CheckJobCount
#' @description Check if desired number of cores exceeds those available.
#' @param nJobs int number of cores requested, check for viability.
#' @return int viable number of cores for parallel backend
#' @importFrom parallel detectCores
CheckJobCount <- function(nJobs = 1){
  availCores <- parallel::detectCores()
  
  if (nJobs > availCores){
    warning('nJobs must be <= number of available cores. Setting to ', availCores - 1)
    nJobs <- availCores - 1
  }

  return(nJobs)
}


#' @name DetermineObjective
#' @description Determine the learning objective of a modeling task based on target vector:
#' - 'binary:logistic' for binary classification
#' - 'multi:softmax' for multiclass classification
#' - 'reg:linear' for regression (continuous, real-valued output)
#' For example, if y is a factor vector with levels 'cat' and 'dog', returns the 'binary:logistic' objective.
#' @param y vector of target data
#' @return character objective name suitable to pass as the xgboost `obj` argument
DetermineObjective <- function(y){
  if (length(unique(y)) == 1){
    stop('Cannot determine objective when y takes on only 1 unique value')
  }
  
  if (length(dim(y)) > 1){
    stop('y must be a vector (cannot have more than 1 dimension)')
  }
  
  # determine if classification problem or not.
  classifier <- FALSE
  if (is.factor(y) | is.character(y)){
    classifier <- TRUE
  } else if (all(y - as.integer(y) == 0) & length(unique(y)) < 100) {
    classifier <- TRUE
  }
  
  if (classifier) {
    objective <- ifelse(length(unique(y)) == 2
                        , 'binary:logistic'
                        , 'multi:softmax')
  } else {
    objective <- 'reg:linear'
  }
  
  return(objective)
}


#' @name XgbTreeParams
#' @description Specify set of valid paramers for a single xgb tree learner.
XgbTreeParams <- function(){
  return(c('eta'
             , 'gamma'
             , 'max_depth'
             , 'min_child_weight'
             , 'max_delta_step'
             , 'subsample'
             , 'colsample_bytree'
             , 'colsample_bylevel'
             , 'lambda'
             , 'alpha'
             , 'tree_method'
             , 'sketch_eps'
             , 'scale_pos_weight'
             , 'updater'
             , 'refresh_leaf'
             , 'process_type'
             , 'grow_policy'
             , 'max_leaves'
             , 'max_bin'
             , 'predictor'))
}

#' @name XgbTreeTunables
#' @description Specify set of all tunable (via grid search) hyperparameters for an xgb tree learner,
#' a superset of XgbTreeParams()
XgbTreeTunables <- function(){
  return(c(XgbTreeParams()
           , 'early_stopping_rounds'
           , 'nrounds'))
}

#' @name XgbTaskParams
#' @description Specify set of xgboost task parameters
XgbTaskParams <- function(){
  return(c('objective'
           , 'base_score'
           , 'eval_metric'))
}

#' @name GetXgbTreeDefaults
#' @description Generate list of default settings for an Xgboost tree model, to use as a go-to
#' hyperparameter space + objective function for a hyperparameter search under cross-validation.
#' @param objective type of modeling problem, e.g.
#' - 'binary:logistic' for binary classification
#' - 'multi:softmax' for multiclass classification
#' - 'reg:linear' for regression (continuous, real-valued output)
#' @importFrom data.table as.data.table
#' @return data.table whose rows are combinations of xgboost arguments
GetXgbTreeDefaults <- function(objective
                               , numClass = NULL){
  
  # define standard set of hyperparameters.
  params <- list(nrounds = c(50, 250, 500)
                 , max_depth = c(3, 6)
                 , eta = c(0.3, 1)
                 , silent = 1
                 , verbose = FALSE
                 , subsample = 0.75
                 , colsample_bytree = 0.75
                 , booster = 'gbtree'
                 , objective = objective)

  return(data.table::as.data.table(expand.grid(params)))
}
