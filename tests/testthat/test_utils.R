# ---- Load reqd packages
library(testthat)
library(data.table)

# ---- Clear environment
rm(list = ls())


# ------------------ #
# ---- OVERHEAD ---- #
# ------------------ #
irisDT <- as.data.table(iris)
binaryClassDT <- copy(irisDT)
multiClassDT <- copy(irisDT)


# --------------- #
# ---- TESTS ---- #
# --------------- #
context('Input data validation function testing...')

test_that('DetectCatVars selects character/factor features correctly', {
  dt <- data.table(matrix(rnorm(12)
                          , ncol = 4))
  dt[, eval(c('V3', 'V4')) := lapply(.SD, as.character), .SDcols = c('V3', 'V4')]
  dt[, V3 := factor(1:3)]
  dt[, V4 := c(paste0('row', 1:3))]
  
  expect_true(all(('V3' %in% DetectCatVars(dt)) &
                    ('V4' %in% DetectCatVars(dt)) &
                    !grepl('V1|V2', DetectCatVars(dt))))
})

test_that('PeelTargetVar throws error from invalid column name or index', {
  expect_error(PeelTargetVar(x = binaryClassDT
                             , y = 'not-a-valid-field-name')
               , regexp = 'supplied y "not-a-valid-field-name" is not the name of a column in x')
  
  expect_error(PeelTargetVar(x = binaryClassDT
                             , y = 7)
               , regexp = 'cannot select target column y = 7')
})

test_that('PeelTargetVar throws error from invalid column name', {
  expect_error(PeelTargetVar(x = binaryClassDT
                             , y = 'not-a-valid-field-name')
               , regexp = 'supplied y "not-a-valid-field-name" is not the name of a column in x')
})

test_that('PeelTargetVar works with data.frame, data.table, and matrix inputs', {
  
})

context('Xgboost modeling helper utilities testing...')

test_that('DetermineObjective fails when y is 2-dimensional or degenerate', {
  y <- 1
  expect_error(DetermineObjective(y)
               , regexp = 'only 1 unique value')
  
  expect_error(DetermineObjective(iris)
               , regexp = 'y must be a vector')
})

test_that('DetermineObjective detects correct objectives', {
  expect_equal(DetermineObjective(c(1, 2)), 'binary:logistic')
  expect_equal(DetermineObjective(factor(c(1, 2, 3))), 'multi:softmax')
  expect_equal(DetermineObjective(c(1.2, 4.5)), 'reg:squarederror')
})

test_that('GetXgbTreeDefaults fails for invalid objective or numClass specifications', {
  expect_error(GetXgbTreeDefaults('multi:softmax')
               , regexp = 'must provide numClass')
  expect_error(GetXgbTreeDefaults('multi:softmax'
                                  , numClass = 2)
               , regexp = 'objective is not binary classification')
  expect_error(GetXgbTreeDefaults('not-an-objective')
               , regexp = 'Unknown objective')
})

test_that('GetXgbTreeDefaults returns expected default xgboost arguments', {
  expect_named(GetXgbTreeDefaults('reg:squarederror')
               , expected = c('obj', 'eval_metric', 'params')
               , ignore.order = TRUE)
  expect_named(GetXgbTreeDefaults('binary:logistic'
                                  , numClass = 2)
               , expected = c('obj', 'num_class', 'eval_metric', 'params')
               , ignore.order = TRUE)
  expect_true(is.data.table(GetXgbTreeDefaults('multi:softmax'
                                               , numClass = 4)$params))
})


# ------------------ #
# ---- CLEAN UP ---- #
# ------------------ #
rm(list = ls())


