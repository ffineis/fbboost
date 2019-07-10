#' @title xgb.train or xgb.cv wrapper
#' @name XgbWrapper
#' @description Simple wrapper for xgboost::xgb.train or xgb.cv
#' @param x input data (matrix, data.frame, or data.table) possibly containing a target variable column.
#' @param y vector of (numeric) target values, or the character name or column index of the target variable present in x.
#' @param cv boolean run xgb.cv instead of xgb.train
#' @param ... arguments to xgb.train or xgb.cv depending on value of `cv`
#' @return xgb.Booster model if cv is FALSE, o/w a data.table with cv evaluation_log results.
#' @importFrom xgboost xgb.train xgb.cv xgb.DMatrix
#' @importFrom dplyr select
XgbWrapper <- function(x, y, cv = FALSE, ...){
  
  # Separate feature/target data.
  if(length(y) == 1){
    dat <- PeelTargetVar(x
                         , y = y)

  } else if (length(y) == nrow(x)){
    dat <- list('x' = x
                , 'y' = y)
  } else {
    stop('length(y) != nrow(x) and y is not a target variable name nor column index')
  }
  
  if (!class(dat$y) %in% c('numeric', 'integer')) {
    stop('y must be an integer or numeric feature (must encode character or factor as numeric).')
  }
  
  # Format bulk args to xgb.train.
  args <- list(...)

  # construct a model matrix.
  if (!'matrix' %in% class(dat$x)) {
  
    # one-hot encode categorical vars.
    catVars <- DetectCatVars(dat$x)
    if (length(catVars) > 0){
      warning('Discovered categorical features -- '
              , paste0(catVars, collapse = ', ')
              , '. Recommended encoding these yourself.')
      contArg <- lapply(dplyr::select(dat$x, catVars)
                        , contrasts
                        , contrasts = FALSE)
    } else {
      contArg <- NULL
    }
    modelMat <- model.matrix( ~ . - 1
                              , data = dat$x
                              , contrasts.arg = contArg)
  } else {
    modelMat <- dat$x
  }
    
  # cast data to xgb.DMatrix for input to xgb.train/cv.
  dat <- xgboost::xgb.DMatrix(modelMat, label = dat$y)
  rm('modelMat')
  args[['data']] <- dat
  
  if(cv){
    out <- do.call(xgboost::xgb.cv, args)
  } else {
    out <- do.call(xgboost::xgb.train, args)
  }

  return(out)
}


#' @name XgbCvGridSearch
#' @description Grid search a set of tree-based xgboost hyperparameters via k-fold cross validation.
#' @param x input data (matrix, data.frame, or data.table). Shape is n x p. Categorical features are automatically one-hot encoded.
#' @param y vector of (numeric) target values or character name or column index of target variable present in x.
#' @param nfold integer number of k-fold cross validation sets
#' @param objective character string training objective, e.g. 'reg:linear', 'reg:logistic', 'binary:logistic', etc.
#' Note that xgboost expects 'objective' to be passed with other hyperparameters, but the fbboost API separates out
#' the learning task objective from the broader pool of tuning parameters - that is, do *not* include objective within `hyperDT`.
#' @param hyperDT data.table, each row is a unique combination of tunable xgboost hyperparameters
#' where columns are names of those settings, e.g. `nrounds`, `gamma` and `max_depth`. See \code{\Link{XgbTreeTunables}}.
#' Note that these hyperparameters are a *superset* of the xgb.cv `params` hyperparameters, as the following
#' xgb.cv arguments can be included in a grid search:
#' - nrounds
#' - early_stopping_rounds
#' @param nJobs integer number of cores with which to run grid search
#' @param ... optional arguments to xgb.train to be *FIXED* across all hyperparameter combinations.
#' @return 'xgbcvgrid' list with two elements:
#' @details
#' \itemize{
#'  \item{"log"}{data.table comprised of stacked xgb.cv eval log data.tables, one for each unique parameter setting}
#'  \item{"calls"}{list of arguments passed to xgb.train, unique to each parameter setting. Use for easily training a selected model.}
#' }
#' @importFrom foreach foreach %dopar%
#' @importFrom xgboost xgboost
#' @importFrom doMC registerDoMC
#' @importFrom parallel stopCluster
#' @importFrom data.table rbindlist
#' 
#' @examples
#' 
#' # regression task
#' hyperDT <- data.table(gamma = c(0.3, 0.7), max_depth = c(3, 6), nrounds = c(10, 30))
#' target <- 'Sepal.Width'
#' 
#' cross <- XgbCvGridSearch(iris, y = target, hyperDT = hyperDT, metrics = list('rmse'))
#' 
#' # extract details about best model
#' bestSetting <- cross$log[which.min(test_rmse_mean), setting_id]
#' bestIter <- cross$log[which.min(test_rmse_mean), iter]
#' cross$calls[[bestSetting]]
#' 
#' 
#' # multiclass classification task, use default hyperparameter grid.
#' iris[, 'Species'] <- sapply(as.character(iris[, 'Species']), switch, setosa = 0, versicolor = 1, virginica = 2)
#' cross <- XgbCvGridSearch(iris, y = 'Species', metrics = list('mlogloss'))
#' 
#' @export
#' 
XgbCvGridSearch <- function(x
                            , y
                            , nfold = 5
                            , objective = NULL
                            , hyperDT = NULL
                            , nJobs = 1
                            , seed = NULL
                            , ...){
  
  # extract vectorized y so we can determine if multiclass classification objective.
  if (length(y) == 1) {
    dat <- PeelTargetVar(x
                         , y = y)
    x <- dat$x
    y <- dat$y
    rm('dat')
  } else if (length(y) != nrow(x)) {
    stop('length(y) != nrow(x) and y is not a target variable name nor column index')
  }

  # ---- Handle learning objective.
  # user has *not* explicitly specified objective:
  if (is.null(objective)){
    
    # if user has specified learning objective in hyperDT, warn them if there are multiple different objectives.
    if (!is.null(hyperDT) && ('objective' %in% names(hyperDT))) {
      if (length(unique(hyperDT$objective)) > 1) {
        warning('Multiple objectives were specified. Do not use resulting output with TrainFromSearch')
      }
      
    # if user has not specified a learning objective and it's not in hyperDT, then infer objective.
    } else {
      objective <- DetermineObjective(y)
      
      # inform user of the determined objective.
      message('determined objective function: '
              , objective
              , '\n'
              , '(please specify `objective` directly if this inferred objective is incorrect)')
      flush.console()
    }
    
  # user has explicitly specified objective:
  # Don't allow user to specify multiple learning objectives at once.
  } else {
    if (length(objective) > 1) {
      stop('Please specify one learning objective at a time, otherwise put `objective` in hyperDT.')
    }
    message('fixed objective function: '
            , objective)
    flush.console()
  }

  # if user has supplied an objective but not a parameter grid, get default grid.
  if (is.null(hyperDT)){
    message('No hyperparameters specified... Using default parameter grid.')
    flush.console()
    hyperDT <- GetXgbTreeDefaults(objective)
  }
  
  # finally, ensure objective passed as param to xgb.train/cv.
  if (!'objective' %in% names(hyperDT)){
    hyperDT[, objective := objective]
  }
  
  # drop duplicated grid points, ensure hyperparameter fields aren't factors.
  hyperDT <- unique(hyperDT)
  catCols <- DetectCatVars(hyperDT)
  hyperDT[, eval(catCols) := lapply(.SD, as.character), .SDcols = catCols]

  # Eliminate any invalid xgb keyword arguments.
  # xgbArgs correspond to {all arguments to xgb.train} - {'params'}.
  xgbArgs <- list(...)
  xgbArgs <- xgbArgs[which(names(xgbArgs) %in% setdiff(names(formals(xgboost::xgb.cv)), 'params'))]

  # Ensure that no hyperparameters were specified redundantly as xgb arguments.
  argsParams <- intersect(xgbArgs, names(hyperDT))
  if (length(argsParams) > 0){
    stop('Some Xgboost hyperparameters were also specified as xgb arguments (only specify once):'
         , paste0(argsParams
                  , collapse = ', '))
  }
  
  # formal args to xgboost like `nrounds` and `early_stopping_rounds` are tunable,
  # but are not listed as tree parameters. Ensure they're in hyperDT.
  for (arg in intersect(names(xgbArgs), setdiff(XgbTreeTunables(), XgbTreeParams()))){
    hyperDT[, eval(arg) := xgbArgs[[arg]]]
    xgbArgs[[arg]] <- NULL
  }
  
  # Determine the number of cores to run parallel cross val search on, register parallel backend.
  if(nJobs > 1){
    nJobs <- CheckJobCount(nJobs)
    doMC::registerDoMC(nJobs)
  }
  
  # Set up non-tunable arguments to be supplied to XgbWrapper.
  xgbArgs[['x']] <- x
  xgbArgs[['y']] <- y
  xgbArgs[['cv']] <- TRUE
  xgbArgs[['nfold']] <- nfold
  
  # if multiclass classification problem, attempt to ensure that num_class passed to xgb.cv or xgb.train.
  if (!is.null(objective) && grepl('multi', objective)){
    xgbArgs[['num_class']] <- length(unique(y))
  }
  # Ensure that verbosity is quiet by default.
  if(!'verbose' %in% xgbArgs){
    xgbArgs[['verbose']] <- FALSE
  }
  
  # ---- Run gridsearch over multiple cores.
  # Distribute hyperparameter search over cores.
  cvs <- foreach::foreach(i = seq_along(1:nrow(hyperDT))
                          # , .export = c('xgbArgs', 'hyperDT', 'seed')
                          , .packages = c('fbboost', 'data.table')) %dopar% {
    cvArgs <- list()

    # Set xgboost param configuration for this hyperparameter setting.
    params <- intersect(names(hyperDT), c(XgbTreeParams(), XgbTaskParams()))
    if (length(params) > 0){
      cvArgs[['params']] <- as.list(unlist(hyperDT[i, .SD, .SDcols = params]))
    }

    # Take non-hyperparameter (but tunable) arguments and set them for xgb.cv args.
    for(arg in intersect(names(hyperDT), setdiff(XgbTreeTunables(), XgbTreeParams()))){
      cvArgs[[arg]] <- hyperDT[i, get(arg)]
    }
    
    if (!is.null(seed)){
      set.seed(seed)
    }
    
    # Store call to xgboost, only keeping args/params passed as list (for easy retraining).
    cv <- do.call(XgbWrapper, c(xgbArgs, cvArgs))
    cv$call <- as.list(cv$call)[-1]
    cv$call$data <- NULL
    cv$call$nfold <- NULL
    
    return(list(log = cv$evaluation_log, call = cv$call))
  }
  
  # clean up messages, backend.
  if (nJobs > 1){
    parallel::stopCluster()
  }
  
  # consolidate run logs for easy analysis
  logDT <- rbindlist(lapply(cvs, function(x){x$log})
                     , idcol = 'setting_id')
  
  # display min/max average CV test-set metrics + corresponding setting.
  testMeanCols <- names(logDT)[grepl('^test_.*_mean$', names(logDT))]
  for (col in testMeanCols){
    minValSettingId <- logDT[which.min(get(col)), setting_id]
    minVal <- logDT[which.min(get(col)), get(col)]
    minValIter <- logDT[which.min(get(col)), iter]
    maxValSettingId <- logDT[which.max(get(col)), setting_id]
    maxVal <- logDT[which.max(get(col)), get(col)]
    maxValIter <- logDT[which.max(get(col)), iter]
    
    minMsg <- paste0('Min(', col, ') = ', minVal, ' -- parameter setting ', minValSettingId, ' @ ', minValIter, ' rounds')
    maxMsg <- paste0('Max(', col, ') = ', maxVal, ' -- parameter setting ', maxValSettingId, ' @ ', maxValIter, ' rounds')
    message(minMsg)
    message(maxMsg)
    flush.console()
  }
  
  # format output
  out <- list(log = logDT, calls = lapply(cvs, function(x){x$call}))
  class(out) <- 'xgbcvgrid'

  return(out)
}


#' @name TrainFromSearch
#' @description Train a tree-based xgboost model using a specific parameter combination present within a
#' parameter grid that was evaluated by the \code{\Link{XgbCvGridSearch}} function
#' @param x input data (matrix, data.frame, or data.table). Shape is n x p. Categorical features are automatically one-hot encoded.
#' @param y vector of (numeric) target values or character name or column index of target variable present in x.
#' @param xgbcvgrid list output from \code{\Link{XgbCvGridSearch}} containing named elements `log` and `calls`.
#' @param id integer specific parameter setting (evaluated during grid search, e.g. row number from hyperDT parameter grid)
#' to use for model training. Implies that the xgboost settings within xgbcvgrid$calls[[id]] will be used for training.
#' @param nrounds integer (optional) number of training rounds to use. If not provided, the `nrounds` provided by
#' xgbcvgrid$calls[[id]] will be used. Use if the optimal parameter setting's `nrounds` overfit during cross-validation and
#' a shorter training time would be more optimal.
#' @return fitted xgb.Booster model
#' @seealso \code{\Link{XgbCvGridSearch}}
#' 
#' @example
#' 
#' # gridsearch on multiclass classification task
#' hyperDT <- data.table(gamma = c(0.3, 0.7), max_depth = c(3, 6), nrounds = c(10, 30))
#' iris[, 'Species'] <- sapply(as.character(iris[, 'Species']), switch, setosa = 0, versicolor = 1, virginica = 2)
#' cross <- XgbCvGridSearch(iris, y = 'Species', metrics = list('mlogloss'))
#' 
#' # select best model settings with best number of training rounds
#' best <- cross$log[which.min(test_mlogloss_mean), c(setting_id, iter)]
#' mdl <- TrainFromSearch(iris, y = 'Species', xgbcvgrid = cross, id = best[1], nrounds = best[2])
#' 
#' @export
#' 
TrainFromSearch <- function(x, y, xgbcvgrid, id, nrounds = NULL){
  
  if (!class(xgbcvgrid) == 'xgbcvgrid') {
    stop('xgbcvgrid must be `xgbcvgrid` object output from XgbCvGridSearch.')
  }
  
  # extract desired hyperparam setting's call to xgb.cv.
  call <- xgbcvgrid$calls[[id]]
  
  # allow user to directly specify number of training rounds to use, in case
  # best parameter setting still eventually led to overfitting at original nrounds.
  if (!is.null(nrounds)) {
    call$nrounds <- as.integer(nrounds)
  }
  
  bst <- do.call(XgbWrapper, c(list(x = x, y = y), call))
  
  return(bst)
}
