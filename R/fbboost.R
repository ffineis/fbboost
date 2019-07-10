#' @title Fit a penalized least squares model to data in high-dimensional tree space
#' @name Fbboost
#' @description Fit a penalized least squares model to data in high-dimensional tree space using a
#' boosted tree model
#' @param x input data (matrix, data.frame, or data.table). Shape is n x p.
#' @param y input labels or name of target variable present in x.
#' @param booster a fitted xgb.Booster
#' @param glmFolds number of k-folds used to cross validate the linear model
#' @return fitted glmnet model
# Fbboost <- function(x
#                     , y
#                     , glmFolds = 3
#                     , nJobs = CheckJobCount()
#                     , hyperDT = NULL){
# 
#   # separate linear model hyperparameters from xgboost hyperparameters.
#   if(!is.null(hyperDT)){
#     glmnetHyperDT <- unique(hyperDT[, .SD
#                                     , .SDcols = intersect(names(hyperDT)
#                                                           , names(formals(glmnet::glmnet)))])
#     xgbHyperDT <- unique(hyperDT[, .SD
#                                  , .SDcols = c(setdiff(names(hyperDT)
#                                                        , names(glmnetHyperDT)))])
#   }
#   
#   # If no xgbooster has been supplied, train one. Use hyperparameter grid search if need be.
#   if(is.null(booster)){
#     
#     # If multiple xgboost settings are supplied, perform cross validation.
#     if(dim(xgbHyperDT)[1] > 1){
#       
#       # Search hyperparameter grid.
#       cv <- SearchXgbHyperparams(x
#                                  , y = y
#                                  , nfold = nFold
#                                  , hyperDT = xgbHyperDT
#                                  , nJobs = nJobs)
#       testCvMetrics <- grep('test', names(cv[[1]]), value = TRUE)
#       testCvMetric <- testCvMetrics[!grepl('std', testCvMetrics)]
#       
#       # Determine if `objective` was minimized or maximized, if this info
#       # was not provided directly.
#       if(!'maximize' %in% names(hyperDT)){
#         maximize <- any(grepl('auc', names(cv[[1]])))
#       } else {
#         maximize <- xgbHyperDT[1, maximize]
#       }
#   
#       # Find best cross validated score over grid.
#       cvScores <- lapply(cv, FUN = function(cvDT){
#         minMaxFun <- ifelse(maximize
#                             , max
#                             , min)
#         whichMinMaxFun <- ifelse(maximize
#                                  , which.max
#                                  , which.min)
#         
#         return(list('bestCvScore' = minMaxFun(cvDT[, get(testCvMetric)])
#                     , 'bestIter' = whichMinMaxFun(cvDT[, get(testCvMetric)])))
#       })
#       
#       # Scan list of CV model scores and back out which setting is the best at which nrounds.
#       bestCvScores <- unlist(lapply(cvScores
#                                     , FUN = function(score){score$bestCvScore}))
#       bestCvScoreIdx <- ifelse(maximize
#                                , which.max(bestCvScores)
#                                , which.min(bestCvScores))
#       
#       # Subset xgbArgs to the best cv tuning, train a model.
#       xgbHyperDT <- xgbHyperDT[bestCvScoreIdx, ]
#       xgbHyperDT[, nrounds := cvScores[[bestCvScoreIdx]]$bestIter]
#     }
#     
#     booster <- XgbWrapper(x
#                           , y = y
#                           , as.list(xgbHyperDT))
#   
#   }
#   
#   # embed the data in the "tree space" according to the booster.
#   embedMat <- EmbedBooster(x
#                            , model = booster)
#   
#   if dim()
#   if(crossVal){
#     linModel <- cv.glmnet(embedMat
#                           , y = y # fix this
#                           , nfolds = nFolds)
#   } else {
#     linModel <- glmnet(embedMat
#                        , y = y # fix this
#                        , nfolds = nFolds)
#   }
#   
#   return(linModel)
# }
