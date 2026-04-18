// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include <vector>
#include <unordered_map>
#include <string>
#include <sstream>
#include <limits>
#include "hal9001_types.h"
using namespace Rcpp;

struct BasisMeta {
  std::vector<int> cols;
  std::vector<double> cutoffs;
  std::vector<int> orders;
};

struct SupportMask {
  std::vector<unsigned char> active;
  std::vector<double> values;
};

std::string basis_key(const BasisMeta& basis, int upto = -1) {
  if (upto < 0 || upto > static_cast<int>(basis.cols.size())) {
    upto = static_cast<int>(basis.cols.size());
  }

  std::ostringstream oss;
  oss.precision(std::numeric_limits<double>::max_digits10);
  for (int i = 0; i < upto; ++i) {
    oss << basis.cols[i] << ':' << basis.cutoffs[i] << ':' << basis.orders[i] << '|';
  }
  return oss.str();
}

inline double condition_value(double obs, double cutoff, int order) {
  if (!(obs >= cutoff)) {
    return 0.0;
  }
  if (order == 0) {
    return 1.0;
  }
  return std::pow(obs - cutoff, order);
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Functions to enumerate basis functions
//------------------------------------------------------------------------------

// populates a map with unique basis functions based on data in xsub values are
// thresholds, keys are column indices
BasisMap enumerate_basis(const NumericMatrix& X_sub,
                         const NumericVector& cols) {
  BasisMap bmap;

  //find unique basis functions
  int n = X_sub.rows();
  for(int i = 0; i < n; i++) {
    NumericVector cutoffs = X_sub.row(i);
    bmap.insert(std::pair<NumericVector, NumericVector>(cutoffs, cols));
  }
  //erase the lowest(always true) basis function
  //actually, I think we don't want to do this
  //because it might not be true in an OOB prediction set
  //bmap.erase(bmap.begin());
  return(bmap);
}

//------------------------------------------------------------------------------

//' Sort Basis Functions
//'
//' Build a sorted list of unique basis functions based on columns, where each
//' basis function is a list
//'
//' @details Note that sorting of columns is performed such that the basis order
//' equals cols.length() and each basis function is a list(cols, cutoffs).
//'
//' @param X_sub A subset of the columns of X, the original design matrix.
//' @param cols An index of the columns that were reduced to by sub-setting.
//' @param order_map A vector with length the original unsubsetted matrix X which specifies the smoothness of the function in each covariate.
// [[Rcpp::export]]
List make_basis_list(const NumericMatrix& X_sub, const NumericVector& cols, const IntegerVector& order_map){

  BasisMap bmap = enumerate_basis(X_sub, cols);
  List basis_list(bmap.size());
  int index = 0;
  for (BasisMap::iterator it = bmap.begin(); it != bmap.end(); ++it) {
    // List basis(2);
    // basis[0]=it->second;
    // basis[1]=it->first;

    // List basis();
    // basis["cols"]=it->second;
    // basis["cutoffs"]=it->first;
    NumericVector subCols = it->second;

    IntegerVector order (subCols.length());
    for(int i=0; i < subCols.length(); i++){
      order[i] = order_map[subCols[i]-1];
    }

    List basis = List::create(
      Rcpp::Named("cols") = it->second,
      Rcpp::Named("cutoffs") = it->first,
      Rcpp::Named("orders") = order
    );
    basis_list[index++] = basis;
  }
  return(basis_list);
}

//------------------------------------------------------------------------------

//' Compute Values of Basis Functions
//'
//' Computes and returns the indicator value for the basis described by
//' cols and cutoffs for a given row of X
//'
//' @param X The design matrix, containing the original data.
//' @param row_num Numeri for  a row index over which to evaluate.
//' @param cols Numeric for the column indices of the basis function.
//' @param cutoffs Numeric providing thresholds.
//' @param orders Numeric providing smoothness orders
//'
// [[Rcpp::export]]
double meets_basis(const NumericMatrix& X, const int row_num,
                   const IntegerVector& cols, const NumericVector& cutoffs,  const IntegerVector& orders) {
  int p = cols.length();
  double value = 1;


  for (int i = 0; i<p; i++) {
    double obs = X(row_num,cols[i] - 1); // using 1-indexing for basis columns
    int order =  orders[i];
    double cutoff = cutoffs[i];
    if(!(obs >= cutoff)) {
      return(0);
    }
    if(order!=0){
      value = value * pow((obs - cutoff),order);
    }

  }

  return(value);
}



//------------------------------------------------------------------------------

//' Generate Basis Functions
//'
//' Populates a column (indexed by basis_col) of x_basis with basis indicators.
//'
//' @param basis The basis function.
//' @param X The design matrix, containing the original data.
//' @param x_basis The HAL design matrix, containing indicator functions.
//' @param basis_col Numeric indicating which column to populate.
//'
void evaluate_basis_full(const BasisMeta& basis, const NumericMatrix& X, SpMat& x_basis,
                         int basis_col, SupportMask& support) {
  int n = X.rows();
  int p = static_cast<int>(basis.cols.size());

  support.active.assign(n, 0);
  support.values.assign(n, 0.0);

  for (int row_num = 0; row_num < n; row_num++) {
    double value = 1.0;
    bool keep = true;

    for (int i = 0; i < p; i++) {
      double obs = X(row_num, basis.cols[i] - 1);
      double cutoff = basis.cutoffs[i];
      int order = basis.orders[i];

      if (!(obs >= cutoff)) {
        keep = false;
        break;
      }

      if (order != 0) {
        value *= std::pow(obs - cutoff, order);
      }
    }

    if (keep && value != 0.0) {
      x_basis.insert(row_num, basis_col) = value;
      support.active[row_num] = 1;
      support.values[row_num] = value;
    }
  }
}

void evaluate_basis_from_parent(const BasisMeta& basis,
                                const SupportMask& parent_support,
                                const NumericMatrix& X, SpMat& x_basis,
                                int basis_col, SupportMask& support) {
  int n = X.rows();
  int last = static_cast<int>(basis.cols.size()) - 1;

  support.active.assign(n, 0);
  support.values.assign(n, 0.0);

  double cutoff = basis.cutoffs[last];
  int order = basis.orders[last];

  for (int row_num = 0; row_num < n; ++row_num) {
    if (!parent_support.active[row_num]) {
      continue;
    }

    double obs = X(row_num, basis.cols[last] - 1);

    if (!(obs >= cutoff)) {
      continue;
    }

    double value = parent_support.values[row_num];
    if (order != 0) {
      value *= std::pow(obs - cutoff, order);
    }

    if (value != 0.0) {
      x_basis.insert(row_num, basis_col) = value;
      support.active[row_num] = 1;
      support.values[row_num] = value;
    }
  }
}

//------------------------------------------------------------------------------

//' Build HAL Design Matrix
//'
//' Make a HAL design matrix based on original design matrix X and a list of
//' basis functions in argument blist
//'
//' @param X Matrix of covariates containing observed data in the columns.
//' @param blist List of basis functions with which to build HAL design matrix.
//' @param p_reserve Sparse matrix pre-allocation proportion. Default value is 0.5. 
//' If one expects a dense HAL design matrix, it is useful to set p_reserve to a higher value.
//' @export
//'
//' @examples
//' \donttest{
//' gendata <- function(n) {
//'   W1 <- runif(n, -3, 3)
//'   W2 <- rnorm(n)
//'   W3 <- runif(n)
//'   W4 <- rnorm(n)
//'   g0 <- plogis(0.5 * (-0.8 * W1 + 0.39 * W2 + 0.08 * W3 - 0.12 * W4))
//'   A <- rbinom(n, 1, g0)
//'   Q0 <- plogis(0.15 * (2 * A + 2 * A * W1 + 6 * A * W3 * W4 - 3))
//'   Y <- rbinom(n, 1, Q0)
//'   data.frame(A, W1, W2, W3, W4, Y)
//' }
//' set.seed(1234)
//' data <- gendata(100)
//' covars <- setdiff(names(data), "Y")
//' X <- as.matrix(data[, covars, drop = FALSE])
//' basis_list <- enumerate_basis(X)
//' x_basis <- make_design_matrix(X, basis_list)
//' }
//'
//' @return A \code{dgCMatrix} sparse matrix of indicator basis functions
//'  corresponding to the design matrix in a zero-order highly adaptive lasso.
// [[Rcpp::export]]
SpMat make_design_matrix(const NumericMatrix& X, const List& blist, double p_reserve = 0.5) {
  int n = X.rows();
  int basis_p = blist.size();

  SpMat x_basis(n, basis_p);
  x_basis.reserve(p_reserve * n * basis_p);

  std::vector<BasisMeta> basis_meta;
  basis_meta.reserve(basis_p);
  std::unordered_map<std::string, int> basis_index;
  basis_index.reserve(basis_p);

  for (int basis_col = 0; basis_col < basis_p; basis_col++) {
    List basis = blist[basis_col];
    IntegerVector cols_r = as<IntegerVector>(basis["cols"]);
    NumericVector cutoffs_r = as<NumericVector>(basis["cutoffs"]);
    IntegerVector orders_r = as<IntegerVector>(basis["orders"]);

    BasisMeta meta;
    meta.cols = Rcpp::as<std::vector<int> >(cols_r);
    meta.cutoffs = Rcpp::as<std::vector<double> >(cutoffs_r);
    meta.orders = Rcpp::as<std::vector<int> >(orders_r);

    basis_meta.push_back(meta);
    basis_index[basis_key(basis_meta.back())] = basis_col;
  }

  std::vector<SupportMask> supports(basis_p);

  for (int basis_col = 0; basis_col < basis_p; basis_col++) {
    const BasisMeta& meta = basis_meta[basis_col];
    int degree = static_cast<int>(meta.cols.size());

    if (degree > 1) {
      std::string parent_key = basis_key(meta, degree - 1);
      auto parent_it = basis_index.find(parent_key);

      if (parent_it != basis_index.end() && parent_it->second < basis_col) {
        int parent_col = parent_it->second;
        evaluate_basis_from_parent(
          meta,
          supports[parent_col],
          X,
          x_basis,
          basis_col,
          supports[basis_col]
        );
        continue;
      }
    }

    evaluate_basis_full(
      meta,
      X,
      x_basis,
      basis_col,
      supports[basis_col]
    );
  }

  x_basis.makeCompressed();
  return(x_basis);
}

