# ---- Load reqd packages
library(testthat)
library(data.table)
library(xgboost)

# ---- Clear environment
rm(list = ls())


# ------------------ #
# ---- OVERHEAD ---- #
# ------------------ #
irisDT <- as.data.table(iris)
binaryClassDT <- copy(irisDT)

binaryClassDT[, Species := ifelse(as.character(Species) == 'setosa', 1, 0)]

binaryClassParams <- list(max_depth = 2
	, eta = 1
	, silent = 1
	, nthread = 2
	, colsample_bytree = 0.8
	, booster = 'gbtree'
	, objective = 'binary:logistic'
	, eval_metric = 'auc')

varNames <- setdiff(names(irisDT), 'Species')
nRounds <- 3

xgbDat <- xgboost::xgb.DMatrix(as.matrix(binaryClassDT[, .SD, .SDcols = varNames])
	, label = binaryClassDT[, Species])
bst <- xgboost::xgb.train(data = xgbDat
	, params = binaryClassParams
	, nrounds = nRounds)


# --------------- #
# ---- TESTS ---- #
# --------------- #

# ---- CONTEXT: GetTreeTable function testing
context('GetTreeTable testing...')

test_that('GetTreeTable returns data.table with correct number of trees', {
	treeDT <- GetTreeTable(varNames
		, model = bst)

	expect_true('data.table' %in% class(treeDT))
	expect_true(max(treeDT[, Tree]) == bst$niter)
})

test_that('GetTreeTable fails on bad input', {
	expect_error(GetTreeTable(NULL
		, model = bst)
		, 'must be > 0')
	expect_error(GetTreeTable(NULL
		, model = 'not an xgb.Booster')
		, 'must be of class xgb.Booster')
})

# ---- CONTEXT: tree embedding function testing
context('EmbedInTree testing...')

test_that('EmbedInTree returns a matrix with expected dimensions', {
	treeDT <- GetTreeTable(varNames
			, model = bst)

	embedMat <- EmbedInTree(binaryClassDT
		, tree = treeDT[Tree == 2])

	expect_equal(nrow(embedMat), nrow(binaryClassDT))
	expect_equal(ncol(embedMat), max(treeDT[Tree == 2]$Node))
})

test_that('EmbedInTree fails on bad input', {
	treeDT <- GetTreeTable(varNames
			, model = bst)

	expect_error(EmbedInTree(NULL
		, tree = treeDT[Tree == 2])
		, 'length zero')
	expect_error(EmbedInTree(binaryClassDT
		, tree = treeDT)
		, 'one row per node')
})


# ---- CONTEXT: booster embedding function
context('EmbedBooster testing...')

test_that('EmbedInTree returns a matrix with no missing values, with expected dimensions', {
	embedMat <- EmbedBooster(binaryClassDT
		, model = bst)

	expect_equal(nrow(embedMat), nrow(binaryClassDT))
	expect_equal(nrow(na.omit(embedMat)), nrow(embedMat))
})


# ---- Clean up
rm(list = ls())


