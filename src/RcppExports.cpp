// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

// EmbedInTree
IntegerVector EmbedInTree(NumericMatrix x, CharacterVector featureNames, CharacterVector splitFeatures, NumericVector splits, IntegerMatrix movements);
RcppExport SEXP _fbboost_EmbedInTree(SEXP xSEXP, SEXP featureNamesSEXP, SEXP splitFeaturesSEXP, SEXP splitsSEXP, SEXP movementsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type x(xSEXP);
    Rcpp::traits::input_parameter< CharacterVector >::type featureNames(featureNamesSEXP);
    Rcpp::traits::input_parameter< CharacterVector >::type splitFeatures(splitFeaturesSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type splits(splitsSEXP);
    Rcpp::traits::input_parameter< IntegerMatrix >::type movements(movementsSEXP);
    rcpp_result_gen = Rcpp::wrap(EmbedInTree(x, featureNames, splitFeatures, splits, movements));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_fbboost_EmbedInTree", (DL_FUNC) &_fbboost_EmbedInTree, 5},
    {NULL, NULL, 0}
};

RcppExport void R_init_fbboost(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
