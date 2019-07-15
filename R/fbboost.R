#' @title Fit a penalized least squares model to data in high-dimensional XgBooster tree space
#' @name Fbboost
#' @description The Fbboost model wrapper function. Fits a penalized least squares model to data in high-dimensional tree embedding space using a
#' boosted tree model, fits the boosted tree model if not supplied. Basically, a wrapper for xgb.train, EmbedBoosterData, and cv.glmnet.
#' @param x input data (matrix, data.frame, or data.table). Shape is n x p. All categorical features
#' should be dummy- or one-hot-encoded.
#' @param y target variable vector or name of target variable present in x.
#' @param booster a fitted xgb.Booster. If NULL, Fbboost will automatically run a 
#' grid search over XgBoost hyperparameters and fit the combination with the best test CV mean
#' metrics. If objective should be `maximize[d]`, create a `maximize` column in xgbHyperDT and set
#' the column = TRUE.
#' @param xgbHyperDT data.table containing a parameter grid to search via \code{\Link{XgbCvGridSearch}}.
#' @param family character string the `family` argument passed to glmnet::glmnet
#' @param nfold integer number of k-fold cross validation sets. Used in xgboost CV parameter search if booster not supplied,
#' also passed to cv.glmnet for linear model selection.
#' @param nJobs integer number of cores with which to run grid search, for both xgboost grid search (if booster not supplied)
#' and cv.glmnet for linear model selection.
#' @param seed integer random seed initializer
#' @param ... named list of arguments passed to glmnet::cv.glmnet
#' @return 'fbbooster' list with five elements:
#' @details
#' \itemize{
#'  \item{"booster"}{fitted xgb.Booster model}
#'  \item{"enet_cv"}{glmnet::cv.glmnet object, result of running cross-validation on glmnet hyperparamters}
#'  \item{"xgbcvgrid"}{results of xgboost hyperparameter grid search if `booster` was not supplied, i.e. output from \code{\Link{XgbCvGridSearch}}}
#'  \item{"xgbcvgrid_selection_id"}{xgboost hyperparameter combination ID selected to fit final booster, given `booster` was not supplied}
#'  \item{"xgbcvgrid_selection_iter"}{xgboost training rounds used to fit final booster, given `booster` was not supplied}
#' }
#' 
#' @importFrom glmnet cv.glmnet
#' @importFrom parallelstopCluster
#' @importFrom doMC registerDoMC
#' 
#' @example
#' 
Fbboost <- function(x
                    , y
                    , booster = NULL
                    , xgbHyperDT = NULL
                    , family = 'binomial'
                    , nfold = 5
                    , nJobs = CheckJobCount()
                    , seed = NULL
                    , ...){
  
  # ensure no categorical features were passed through.
  # (OHE transformation will not be clear to user when they want to use predict.Fbboost with test data later on)
  catVars <- DetectCatVars(x)
  if (length(catVars) > 0){
    stop('Categorical features in input data x were found: '
         , paste0(catVars, collapse = ', ')
         , ' Recommended passing a model.matrix.')
  }
  
  # extract vectorized y so we can determine if multiclass classification objective.
  if (length(y) == 1) {
    dat <- PeelTargetVar(x
                         , y = y)
    x <- dat$x
    y <- dat$y
    rm('dat')
  }

  # If no xgbooster has been supplied, train one. Use hyperparameter grid search if need be.
  cross <- NULL
  best <- NULL
  if (is.null(booster)) {
    message('No xgb.Booster passed to Fbboost...')
    flush.console()
    
    # Search hyperparameter grid.
    if ((!is.null(hyperDT) && nrow(hyperDT) > 1) | (is.null(hyperDT))){
      message('Running cross-validated hyperparameter grid search...')
      flush.console()
    }
    cross <- XgbCvGridSearch(x
                             , y = y
                             , nfold = nfold
                             , hyperDT = xgbHyperDT
                             , nJobs = nJobs
                             , seed = seed)

    # Determine if `objective` should be minimized or maximized. Default is minimization.
    if (!'maximize' %in% names(xgbHyperDT)) {
      maximize <- xgbHyperDT[1, maximize]
    } else {
      maximize <- FALSE
    }
    
    testMeanCol <- names(cross$log)[grepl('^test_.*_mean$', names(cross$log))[1]]
    minMaxFun <- ifelse(maximize
                        , yes = which.max
                        , no = which.min)
    best <- cross$log[minMaxFun(get(testMeanCol)), c(setting_id, iter)]
    
    message('Fitting xgb.Booster model...')
    flush.console()
    booster <- TrainFromSearch(x
                               , y = y
                               , xgbcvgrid = cross
                               , id = best[1]
                               , nrounds = best[2])
  }

  # embed the data in the "tree space" according to the booster.
  embedMat <- EmbedBoosterData(x
                               , model = booster
                               , nJobs = nJobs)

  message('Running cv.glmnet to fit final-stage ElasticNet model...')
  flush.console()
  
  enetArgs <- as.list(...)
  if ('standardize' %in% names(enetArgs)) {
    enetArgs$standardize <- NULL
  }
  # Determine the number of cores to run parallel cross val search on, register parallel backend.
  if(nJobs > 1){
    nJobs <- CheckJobCount(nJobs)
    doMC::registerDoMC(nJobs)
  }
  
  enet <- do.call(glmnet::cv.glmnet, c(list(x = embedMat
                                            , y = y
                                            , family = family
                                            , standardize = FALSE
                                            , nfolds = nfold
                                            , parallel = nJobs > 1)
                                       , enetArgs))
  
  # clean up messages, backend.
  if (nJobs > 1){
    parallel::stopCluster()
  }
  
  # return fbbooster object, basically a container for
  # xgb.Booster model, glmnet model, and the xgboost cross-validation summary.
  out <- structure(list(booster = booster
                        , enet_cv = enet
                        , xgbcvgrid = cross
                        , xgbcvgrid_selection_id = best[1]
                        , xgbcvgrid_selection_iter = best[2])
                   , class = 'fbbooster')
  
  return(out)
}


#' @title Make predictions from an "fbbooster" model object.
#' @name Fbboost
#' @description 
#' @param fbbooster a fitted fbbooster model. See Fbboost.
#' @param newx new data to be sent through `fbbooster`. Should have same features as
#' fbbooster$booster$feature_names or at least the same number of features as the constituent booster.
#' @param ... named list of arguments passed to glmnet::cv.glmnet
#' @return 'fbbooster' list with five elements:
#' 
#' @example
#' 
predict.fbbooster <- function(fbbooster
                              , newx
                              , s = c('lambda.1se', 'lambda.min')
                              , ...){
  
  if (is.null(fbbooster$booster$featureNames)) {
    fbbooster$booster$nfeatures
  }
  
}
