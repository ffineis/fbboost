#' @title Back out trees from boosted tree model from xgboost package
#' @name GetTreeTable
#' @description Make the output of xgboost::xgb.model.dt.tree more useful
#' @param featureNames names of data as it was sent into xgboost::xgb.train
#' @param model an xgb.Booster model
#' @return a data.table of trees
#' @importFrom xgboost xgb.model.dt.tree
#' @importFrom data.table := rbindlist
#' @seealso ?xgboost::xgb.model.dt.tree
#' 
GetTreeTable <- function(featureNames, model){
  
  # param checking
  if(!('xgb.Booster' %in% class(model))){
    stop('model argument must be of class xgb.Booster')
  }
  if(length(featureNames) < 1){
    stop('length(featureNames) must be > 0')
  }
  
  # Get xgboost representation of tree learners in a booster model
  trees <- tryCatch({
    xgboost::xgb.model.dt.tree(featureNames
                               , model = model
                               , use_int_id = TRUE)
  }, error = function(e){
    stop('xgb.model.dt.tree failed to back out trees using supplied featureNames and model.')
  })
  
  # Remove any trees that may have been duplicated (on the off chance)
  treeList <- lapply(0:max(trees$Tree)
                     , FUN= function(x){trees[Tree == x]})
  treeList <- unique(treeList)

  return(data.table::rbindlist(treeList))
}

#' @title Embed data in a single xgboost ensemble tree.
#' @name EmbedInTree
#' @description Send a datapoint through a decision tree, embedding the data in binary tree-leaf space, that tree
#' being a structure created with the GetTreeTable function.
#' @param x input data (matrix, data.frame, or data.table). Shape is n x p.
#' @param tree processed data.table output from xgboost::xgb.model.dt.tree corresponding to exactly one constituent tree.
#' @return binary embedding matrix. Of shape n x #-terminal-nodes-in-tree
#' @useDynLib fbboost
#' @importFrom Rcpp sourceCpp
#' 
EmbedInTree <- function(x, featureNames, tree){
  
  # param checking
  nObs <- dim(x)[1]
  if(nObs < 1){
    stop('Cannot embed < 1 observations in to tree space!')
  }
  if(!('data.table' %in% class(tree))){
    stop('tree parameter must be a data.table; use output from GetTreeTable.')
  }
  if(length(unique(tree[, Node])) != nrow(tree)){
    stop('Submit one tree at a time. tree parameter must have one row per node.')
  }
  
  # Ensure that the tree is ordered by ascending Node ID.
  tree <- tree[order(Node)]
  
  # browser()
  
  # embed all observations in tree space
  nodes <- SendThroughTree(x
                           , featureNames = as.character(featureNames)
                           , splitFeatures = as.character(tree[, Feature])
                           , splits = as.numeric(tree[, Split])
                           , movements = as.matrix(tree[, .SD, .SDcols = c('Yes', 'No', 'Missing')]))
  
  return(nodes)
}


#' @title Cast node embedding into a full one-hot-encoded ensemble node embedding.
#' @name OheNodeData
#' @description Once data has been embedded into a booster (i.e. each record stored as an integer
#' vector of leaf IDs, one leaf per ensembled tree), cast it into a sparse embedding: n x (\sum_{i=1}^{k}nodes_{tree_{i}})
#' @param nodeDat integer matrix with shape n x |trees|. Row j is the booster-embedding of observation j.
#' @param leafVec integer vector of length |trees|. Represents the total number of terminal nodes in the
#' constituent ensemble tree learners.
#' @return sparse dgCMatrix of one-hot-encoded node embedding data
#' @importFrom Matrix Matrix sparse.model.matrix
#' 
OheNodeData <- function(nodeDat, leafVec){
  
  nTrees <- ncol(nodeDat)
  
  # data quality checks: each tree must have node count.
  stopifnot(length(leafVec) == nTrees)
  
  # data quality checks: no leaf ID can be larger than tree's node count.
  stopifnot(all(apply(nodeDat, 2, max) <= leafVec))
  
  # instantiate sparse output matrix
  oheMat <- Matrix::Matrix(0
                           , nrow = nrow(nodeDat)
                           , ncol = sum(leafVec)
                           , sparse = TRUE)
  
  # one-hot-encode the node embedding, tree by tree.
  colIdx <- 1
  nodeNames <- character(sum(leafVec))
  for (i in 1:nTrees) {
    nLevels <- leafVec[i]
    treeDat <- data.frame(col = factor(nodeDat[, i]
                                       , levels = 1:nLevels))
    
    # fill up overall ohe embedding with constituent sparse ohe embeddings.
    oheMat[, colIdx:(colIdx + nLevels - 1)] <- Matrix::sparse.model.matrix(~ . - 1
                                                                           , data = treeDat
                                                                           , drop.unused.levels = FALSE)
    nodeNames[colIdx:(colIdx + nLevels - 1)] <- paste0(i, '_', 1:nLevels)
    colIdx <- colIdx + nLevels
  }
  
  # keep track of which trees/nodes.
  colnames(oheMat) <- nodeNames
  
  return(oheMat)
}


#' @title Embed data in high-dimensional space with a boosted tree model.
#' @name EmbedBoosterData
#' @description Embed a dataset into a high-dimensional binary space with a boosted tree model.
#' Largely just assembles trees from XGBoost model and sends data through each tree.
#' @param x input data (matrix, data.frame, or data.table). Shape is n x p.
#' @param model an xgb.Booster model
#' @return list of 2: 'data': data embedded into high-dim space, 'treeCuts': vector defining each tree's embedding.
#' @importFrom doMC registerDoMC
#' @importFrom foreach foreach %dopar%
#' 
#' @example 
#' require(data.table)
#' 
#' # binary classification data
#' DT <- as.data.table(iris)
#' DT[, Species := ifelse(Species == 'setosa', 1, 0)]
#' 
#' # training/test split
#' trIdx <- sample(1:150, size = 100)
#' testIdx <- setdiff(1:150, y = trIdx)
#' 
#' # train xgboost model
#' bst <- XgbWrapper(DT[trIdx], y = 'Species', nrounds = 20)
#' 
#' # embed all data within booster
#' dat <- EmbedBoosterData(as.matrix(DT[, -'Species']), model = bst)
#' 
#' # fit elasticnet model on top of embedded data, then get test set preds.
#' enetMdl <- glmnet(dat[trIdx, ], y = DT$Species[trIdx], alpha = 0.6, lambda = 1e-6)
#' enetPreds <- predict(enetMdl, newx = dat[testIdx, ])  # perfect separation.
#' 
#' @export
EmbedBoosterData <- function(x, model, nJobs=1){
  
  # x *must* be a matrix, as SendThroughTree expects x to be a NumericMatrix.
  stopifnot(class(x) == 'matrix')
  
  # extract feature names.
  if ('feature_names' %in% names(model)){
    featureNames <- model$feature_names
  } else {
    featureNames <- names(x)
  }
  
  # Get trees!
  trees <- GetTreeTable(featureNames
                        , model = model)
  treeIds <- sort(unique(trees[, Tree]))
  nTrees <- length(treeIds)
  
  if(nJobs > 1){
    # Create parallel cluster
    nJobs <- CheckJobCount(nJobs)
    doMC::registerDoMC(nJobs)
    
    # Set up progress bar.
    progBar <- txtProgressBar(min = 1
                              , max = nObs
                              , style = 3)
    message('Embedding progress (% of trees):\n')
    flush.console()

    # Embedding: distribute trees over clusters, all data goes through each tree once.
    # store embedding as n x |trees| (one column per tree embedding).
    nodeMat <- foreach::foreach(i = seq_along(treeIds)
                                , .combine = 'cbind') %dopar% {
      nodes <- EmbedInTree(x
                           , featureNames = featureNames
                           , tree = trees[Tree == treeIds[i]])
      setTxtProgressBar(progBar
                        , i)
      return(nodes)
    }
    close(progBar)

  # for non-parallelized runs, set up node storage directly (shape = n x |trees|).
  } else {
    nodeMat <- matrix(nrow = nrow(x)
                      , ncol = nTrees)
    for(i in seq_along(treeIds)){
      nodeMat[, i] <- EmbedInTree(x
                                  , featureNames = featureNames
                                  , tree = trees[Tree == treeIds[i]])
    }
  }
  
  
  # find number of leaves per tree (need for future embedding)
  leafVec <- trees[, .(nLeaves = max(Node)), by = Tree]$nLeaves
  
  
  return(OheNodeData(nodeMat
                     , leafVec = leafVec))
}
