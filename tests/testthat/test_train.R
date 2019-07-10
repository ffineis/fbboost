# ---- Load reqd packages
library(testthat)
library(data.table)

# ---- Clear environment
rm(list = ls())


# ------------------ #
# ---- OVERHEAD ---- #
# ------------------ #
irisDT <- as.data.table(iris)
multiClassDT <- copy(irisDT)

binaryClassDT <- copy(irisDT)
binaryClassDT[, Species := ifelse(as.character(Species) == 'setosa', 1, 0)]
binaryClassDT[, Petal.Width := factor(as.numeric(Petal.Width > 1.75))]

multiClassDT[, Species := unlist(lapply(as.character(Species), FUN = function(x){
	switch(x
		, setosa = 0
		, versicolor = 1
		, virginica = 2)
	}))]
multiClassDT[, Petal.Width := factor(as.numeric(Petal.Width > 1.75))]


# binary booster params argument, for testing XgbWrapper
binaryClassParams <- list(max_depth = 3
	, eta = 1
	, silent = 1
	, nthread = 2
	, colsample_bytree = 0.8
	, booster = 'gbtree'
	, objective = 'binary:logistic'
	, eval_metric = 'auc')

# multiclass booster params argument, for testing XgbWrapper
multiClassParams <- binaryClassParams
multiClassParams$num_class <- 3
multiClassParams$objective <- 'multi:softmax'
multiClassParams$eval_metric <- 'mlogloss'

# Etc variables/settings
varNames <- setdiff(names(irisDT), 'Species')  # predict Species
nRounds <- 3
nFold <- 3

# --------------- #
# ---- TESTS ---- #
# --------------- #

# ---- CONTEXT: xgb.train wrapper tests
context('XgbWrapper (for model training) testing...')

test_that('XgbWrapper works with vectorized target variable.', {
	bst <- XgbWrapper(x = binaryClassDT[, .SD, .SDcols = varNames]
	                  , y = binaryClassDT[, Species]
	                  , params = binaryClassParams
	                  , nrounds = nRounds)

	expect_is(bst, 'xgb.Booster')
})

test_that('XgbWrapper works with character string target variable.', {
  bst <- XgbWrapper(x = binaryClassDT
                    , y = 'Species'
                    , params = binaryClassParams
                    , nrounds = nRounds)
  
  expect_is(bst, 'xgb.Booster')
})

test_that('XgbWrapper works with multiclass classification.', {
	bst <- XgbWrapper(x = multiClassDT
	                  , y = 'Species'
		                , params = multiClassParams
		                , nrounds = nRounds)

	expect_is(bst, 'xgb.Booster')
})

test_that('XgbWrapper works with continuous regression', {
  bst <- XgbWrapper(x = binaryClassDT
                    , y = 'Sepal.Length'
                    , params = multiClassParams
                    , nrounds = nRounds)
  
  expect_is(bst, 'xgb.Booster')
})

test_that('XgbWrapper accepts matrix, data.frame inputs.', {
  bst1 <- XgbWrapper(x = as.matrix(dplyr::select(binaryClassDT
                                                 , Sepal.Length
                                                 , Sepal.Width
                                                 , Petal.Length))
                     , y = binaryClassDT[, Species]
                     , params = binaryClassParams
                     , nrounds = nRounds)
  
  bst2 <- XgbWrapper(x = as.data.frame(multiClassDT)
                     , y = multiClassDT[, Species]
                     , params = multiClassParams
                     , nrounds = nRounds)
  
  expect_is(bst1, 'xgb.Booster')
  expect_is(bst2, 'xgb.Booster')
})


# ---- CONTEXT: xgb.train wrapper tests for cross validation
context('WrapXgbTrain (for cross validation) testing...')

test_that('WrapXgbTrain works with cv == TRUE by returning an evaluation log list.', {
  cv <- XgbWrapper(x = binaryClassDT[, .SD, .SDcols = varNames]
                   , y = binaryClassDT[, Species]
                   , cv = TRUE
                   , params = binaryClassParams
                   , nrounds = nRounds
                   , nfold = nFold)
  
  expect_is(cv, 'xgb.cv.synchronous')
  expect_equal(dim(cv$evaluation_log), c(3, 5))
})


# ---- CONTEXT: SearchHyperparams cross validation function
context('SearchXgbHyperparams testing...')

test_that('SearchXgbHyperparams runs without user-supplied hyperparameter grid on binary data.', {
  cv <- SearchXgbHyperparams(x = binaryClassDT[, .SD, .SDcols = varNames]
                          , y = binaryClassDT[, Species]
                          , nfold = nFold)
  
  expect_is(cv, 'list')
  expect_true(length(cv) > 0)
  expect_true(any(grepl('test_auc', names(cv[[1]]))))
})

test_that('SearchXgbHyperparams runs without user-supplied hyperparameter grid on multiclass data.', {
  cv <- SearchXgbHyperparams(x = multiClassDT[, .SD, .SDcols = varNames]
                             , y = multiClassDT[, Species]
                             , nfold = nFold)
  
  expect_is(cv, 'list')
  expect_true(length(cv) > 0)
  expect_true(any(grepl('test_mlogloss', names(cv[[1]]))))
})

test_that('SearchXgbHyperparams runs with user-supplied hyperparameter grids.', {
  cv <- SearchXgbHyperparams(x = multiClassDT[, .SD, .SDcols = varNames]
                          , y = multiClassDT[, Species]
                          , hyperDT = data.table(nrounds = c(10, 20))
                          , nfold = nFold)
  
  expect_is(cv, 'list')
  expect_equal(length(cv), 2)
  expect_true(any(grepl('test_rmse', names(cv[[1]])))) # default objective is rmse
})

test_that('SearchXgbHyperparams runs when user provides early_stopping_rounds.', {})


test_that('SearchXgbHyperparams runs when nrounds is fixed or is provided within hyperDT.', {})


test_that('SearchXgbHyperparams fails when hyperDT is invalid.', {
  expect_error(SearchXgbHyperparams(x = multiClassDT[, .SD, .SDcols = varNames]
                                 , y = multiClassDT[, Species]
                                 , hyperDT = data.table()
                                 , nfold = nFold)
               , regexp = '`nrounds` argument to xgb.train/cv must be supplied')
  
  expect_error(SearchHyperparams(x = multiClassDT[, .SD, .SDcols = varNames]
                                 , y = multiClassDT[, Species]
                                 , hyperDT = data.table(cowabunga = 'xxxxx')
                                 , nfold = nFold)
               , regexp = 'The following xgboost arguments are invalid')
})


# ------------------ #
# ---- CLEAN UP ---- #
# ------------------ #
rm(list = ls())
