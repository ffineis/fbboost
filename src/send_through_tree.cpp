#include <Rcpp.h>
using namespace Rcpp;


// [[Rcpp::export]]
IntegerVector SendThroughTree(NumericMatrix x,
                              CharacterVector featureNames,
                              CharacterVector splitFeatures,
                              NumericVector splits,
                              IntegerMatrix movements) {
  /* <summary>
   * Send each observation within a data matrix through a decision tree, recording
   * which leaf node each observation fell within
   * </summary>
   * 
   * <param name="x"> data matrix (n x p) </param>
   * <param name="featureNames"> feature (column) names (1 x p) </param>
   * <param name="splitFeatures"> names of features used for splitting, one per node </param>
   * <param name="splits"> feature split values comprising the tree, one per node </param>
   * <param name="movements"> rules for moving down decision tree using splits
   * comprised of 3 columns, one row per node: Yes (left), No (right), Missing (a default direction) </param>
   * 
   * <returns>
   * integer vector of node IDs representing the embedding of a dataset within the decision tree.
   * </returns>
   */
  
  int n = x.nrow();
  IntegerVector out(n);
  
  for (int i = 0; i < n; i++){
    
    bool leaf = false;
    double split, xVal;
    int node = 0;
    
    // various components of tree must be *Vector to use sugar functions.
    CharacterVector splitFeature = CharacterVector::create("0");
    IntegerVector featureNameIdx;

    while (!leaf) {
      // Rcpp::Rcout << "current node index is " << node << std::endl;
      splitFeature[0] = splitFeatures[node];
      
      if (splitFeature[0] == "Leaf"){
        // Rcpp::Rcout << "Leaf found: node = " << node << std::endl;
        leaf = true;
      } else{
        
        // find current split and feature value.
        featureNameIdx = Rcpp::match(splitFeature, featureNames);
        split = splits[node];
        xVal = x(i, featureNameIdx[0] - 1);
        // Rcpp::Rcout << "splitting feature is " << splitFeature[0] << std::endl;
        // Rcpp::Rcout << "split is " << split << std::endl;
        // Rcpp::Rcout << "xVal is " << xVal << std::endl;
        
        if (NumericVector::is_na(xVal)){
          // Rcpp::Rcout << "xVal is missing " << std::endl;
          node = movements(node, 2);
        } else{
          // split condition is always "less than"
          if (xVal < split){
            // Rcpp::Rcout << "Yes: go left " << std::endl;
            node = movements(node, 0);
          } else {
            // Rcpp::Rcout << "No: go right " << std::endl;
            node = movements(node, 1);
          }
        }
      }
    }
    
    // store node ID if node is a leaf.
    out[i] = node;
  }
  
  return out;
}


/*** R
require(data.table)
irisDT <- as.data.table(iris)
x <- dplyr::select(irisDT[1:3], -Species)
x[1, 'Petal.Width'] <- NA_real_
movements <- matrix(c(1, 2, NA_integer_, NA_integer_, NA_integer_
                      , 4, 3, NA_integer_, NA_integer_, NA_integer_
                      , 1, 2, NA_integer_, NA_integer_, NA_integer_)
                    , ncol = 3
                    , byrow = FALSE)

SendThroughTree(as.matrix(x)
                , featureNames = names(x)
                , splitFeatures = c('Petal.Width', 'Sepal.Width', 'Leaf', 'Leaf', 'Leaf')
                , splits = c(0.1, 3.05, NA_real_, NA_real_, NA_real_)
                , movements = movements)
*/
