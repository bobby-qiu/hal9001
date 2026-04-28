#' HAL: The Highly Adaptive Lasso
#'
#' Estimation procedure for HAL, the Highly Adaptive Lasso
#'
#' @details The procedure uses a custom C++ implementation to generate a design
#'  matrix of spline basis functions of covariates and interactions of
#'  covariates. The lasso regression is fit to this design matrix via
#'  \code{\link[glmnet]{cv.glmnet}} or a custom implementation derived from
#'  \pkg{origami}. The maximum dimension of the design matrix is \eqn{n} -by-
#'  \eqn{(n * 2^(d-1))}, where where \eqn{n} is the number of observations and
#'  \eqn{d} is the number of covariates.
#'
#'  For \code{smoothness_orders = 0}, only zero-order splines (piece-wise
#'  constant) are generated, which assume the true regression function has no
#'  smoothness or continuity. When \code{smoothness_orders = 1}, first-order
#'  splines (piece-wise linear) are generated, which assume continuity of the
#'  true regression function. When \code{smoothness_orders = 2}, second-order
#'  splines (piece-wise quadratic and linear terms) are generated, which assume
#'  a the true regression function has a single order of differentiability.
#'
#'  \code{num_knots} argument specifies the number of knot points for each
#'  covariate and for each \code{max_degree}. Fewer knot points can
#'  significantly decrease runtime, but might be overly simplistic. When
#'  considering \code{smoothness_orders = 0}, too few knot points (e.g., < 50)
#'  can significantly reduce performance. When \code{smoothness_orders = 1} or
#'  higher, then fewer knot points (e.g., 10-30) is actually better for
#'  performance. We recommend specifying \code{num_knots} with respect to
#'  \code{smoothness_orders}, and as a vector of length \code{max_degree} with
#'  values decreasing exponentially. This prevents combinatorial explosions in
#'  the number of higher-degree basis functions generated. The default behavior
#'  of \code{num_knots} follows this logic --- for \code{smoothness_orders = 0},
#'  \code{num_knots} is set to \eqn{500 / 2^{j-1}}, and for
#'  \code{smoothness_orders = 1} or higher, \code{num_knots} is set to
#'  \eqn{200 / 2^{j-1}}, where \eqn{j} is the interaction degree. We also
#'  include some other suitable settings for \code{num_knots} below, all of
#'  which are less complex than default \code{num_knots} and will thus result
#'  in a faster runtime:
#'  - Some good settings for little to no cost in performance:
#'    - If \code{smoothness_orders = 0} and \code{max_degree = 3},
#'      \code{num_knots = c(400, 200, 100)}.
#'    - If \code{smoothness_orders = 1+} and \code{max_degree = 3},
#'      \code{num_knots = c(100, 75, 50)}.
#'  - Recommended settings for fairly fast runtime:
#'    - If \code{smoothness_orders = 0} and \code{max_degree = 3},
#'      \code{num_knots = c(200, 100, 50)}.
#'    - If \code{smoothness_orders = 1+} and \code{max_degree = 3},
#'      \code{num_knots = c(50, 25, 15)}.
#'  - Recommended settings for fast runtime:
#'    - If \code{smoothness_orders = 0} and \code{max_degree = 3},
#'      \code{num_knots = c(100, 50, 25)}.
#'    - If \code{smoothness_orders = 1+} and \code{max_degree = 3},
#'      \code{num_knots = c(40, 15, 10)}.
#'  - Recommended settings for very fast runtime:
#'    - If \code{smoothness_orders = 0} and \code{max_degree = 3},
#'      \code{num_knots = c(50, 25, 10)}.
#'    - If \code{smoothness_orders = 1+} and \code{max_degree = 3},
#'      \code{num_knots = c(25, 10, 5)}.
#'
#' @param X An input \code{matrix} with dimensions number of observations -by-
#'  number of covariates that will be used to derive the design matrix of basis
#'  functions.
#' @param Y A \code{numeric} vector of observations of the outcome variable. For
#'  \code{family="mgaussian"}, \code{Y} is a matrix of observations of the
#'  outcome variables.
#' @param formula A character string formula to be used in
#'  \code{\link{formula_hal}}. See its documentation for details.
#' @param X_unpenalized An input \code{matrix} with the same number of rows as
#'  \code{X}, for which no L1 penalization will be performed. Note that
#'  \code{X_unpenalized} is directly appended to the design matrix; no basis
#'  expansion is performed on \code{X_unpenalized}.
#' @param max_degree The highest order of interaction terms for which basis
#'  functions ought to be generated.
#' @param smoothness_orders An \code{integer}, specifying the smoothness of the
#'  basis functions. See details for \code{smoothness_orders} for more
#'  information.
#' @param num_knots An \code{integer} vector of length 1 or \code{max_degree},
#'  specifying the maximum number of knot points (i.e., bins) for any covariate
#'  for generating basis functions. If \code{num_knots} is a unit-length
#'  vector, then the same \code{num_knots} are used for each degree (this is
#'  not recommended). The default settings for \code{num_knots} are
#'  recommended, and these defaults decrease \code{num_knots} with increasing
#'  \code{max_degree} and \code{smoothness_orders}, which prevents (expensive)
#'  combinatorial explosions in the number of higher-degree and higher-order
#'  basis functions generated. This allows the complexity of the optimization
#'  problem to grow scalably. See details of \code{num_knots} more information.
#' @param reduce_basis Am optional \code{numeric} value bounded in the open
#'  unit interval indicating the minimum proportion of 1's in a basis function
#'  column needed for the basis function to be included in the procedure to fit
#'  the lasso. Any basis functions with a lower proportion of 1's than the
#'  cutoff will be removed. Defaults to 1 over the square root of the number of
#'  observations. Only applicable for models fit with zero-order splines, i.e.
#'  \code{smoothness_orders = 0}.
#' @param family A \code{character} or a \code{\link[stats]{family}} object
#'  (supported by \code{\link[glmnet]{glmnet}}) specifying the error/link
#'  family for a generalized linear model. \code{character} options are limited
#'  to "gaussian" for fitting a standard penalized linear model, "binomial" for
#'  penalized logistic regression, "poisson" for penalized Poisson regression,
#'  "cox" for a penalized proportional hazards model, and "mgaussian" for
#'  multivariate penalized linear model. Note that passing in
#'  family objects leads to slower performance relative to passing in a
#'  character family (if supported). For example, one should set
#'  \code{family = "binomial"} instead of \code{family = binomial()} when
#'  calling \code{fit_hal}.
#' @param lambda User-specified sequence of values of the regularization
#'  parameter for the lasso L1 regression. If \code{NULL}, the default sequence
#'  in \code{\link[glmnet]{cv.glmnet}} will be used. The cross-validated
#'  optimal value of this regularization parameter will be selected with
#'  \code{\link[glmnet]{cv.glmnet}}. If \code{fit_control}'s \code{cv_select}
#'  argument is set to \code{FALSE}, then the lasso model will be fit via
#'  \code{\link[glmnet]{glmnet}}, and regularized coefficient values for each
#'  lambda in the input array will be returned.
#' @param id A vector of ID values that is used to generate cross-validation
#'  folds for \code{\link[glmnet]{cv.glmnet}}. This argument is ignored when
#'  \code{fit_control}'s \code{cv_select} argument is \code{FALSE}.
#' @param weights observation weights; defaults to 1 per observation.
#' @param offset a vector of offset values, used in fitting.
#' @param fit_control List of arguments, including the following, and any
#'  others to be passed to \code{\link[glmnet]{cv.glmnet}} or
#'  \code{\link[glmnet]{glmnet}}.
#'  - \code{cv_select}: A \code{logical} specifying if the sequence of
#'    specified \code{lambda} values should be passed to
#'    \code{\link[glmnet]{cv.glmnet}} in order for a single, optimal value of
#'    \code{lambda} to be selected according to cross-validation. When
#'    \code{cv_select = FALSE}, a \code{\link[glmnet]{glmnet}} model will be
#'    used to fit the sequence of (or single) \code{lambda}.
#'  - \code{use_min}: Specify the choice of lambda to be selected by
#'    \code{\link[glmnet]{cv.glmnet}}. When \code{TRUE}, \code{"lambda.min"} is
#'    used; otherwise, \code{"lambda.1se"}. Only used when
#'    \code{cv_select = TRUE}.
#'  - \code{lambda.min.ratio}: A \code{\link[glmnet]{glmnet}} argument
#'    specifying the smallest value for \code{lambda}, as a fraction of
#'    \code{lambda.max}, the (data derived) entry value (i.e. the smallest value
#'    for which all coefficients are zero). We've seen that not setting
#'    \code{lambda.min.ratio} can lead to no \code{lambda} values that fit the
#'    data sufficiently well.
#'  - \code{prediction_bounds}: An optional vector of size two that provides
#'    the lower and upper bounds predictions; not used when
#'    \code{family = "cox"}. When \code{prediction_bounds = "default"}, the
#'    predictions are bounded between \code{min(Y) - sd(Y)} and
#'    \code{max(Y) + sd(Y)} for each outcome (when \code{family = "mgaussian"},
#'    each outcome can have different bounds). Bounding ensures that there is
#'    no extrapolation.
#' @param basis_list The full set of basis functions generated from \code{X}.
#' @param return_lasso A \code{logical} indicating whether or not to return
#'  the \code{\link[glmnet]{glmnet}} fit object of the lasso model.
#' @param return_x_basis A \code{logical} indicating whether or not to return
#'  the matrix of (possibly reduced) basis functions used in \code{fit_hal}.
#' @param yolo A \code{logical} indicating whether to print one of a curated
#'  selection of quotes from the HAL9000 computer, from the critically
#'  acclaimed epic science-fiction film "2001: A Space Odyssey" (1968).
#'
#' @importFrom glmnet cv.glmnet glmnet
#' @importFrom stats coef
#' @importFrom assertthat assert_that
#' @importFrom origami make_folds folds2foldvec
#'
#' @return Object of class \code{hal9001}, containing a list of basis
#'  functions, a copy map, coefficients estimated for basis functions, and
#'  timing results (for assessing computational efficiency).
#'
#' @rdname fit_hal
#'
#' @export
#'
#' @examples
#' n <- 100
#' p <- 3
#' x <- xmat <- matrix(rnorm(n * p), n, p)
#' y_prob <- plogis(3 * sin(x[, 1]) + sin(x[, 2]))
#' y <- rbinom(n = n, size = 1, prob = y_prob)
#' hal_fit <- fit_hal(X = x, Y = y, family = "binomial")
#' preds <- predict(hal_fit, new_data = x)
score_gaussian_screen_candidates <- function(x_basis, y, candidate_penalized_cols) {
  y_centered <- as.numeric(y - mean(y))
  scores <- as.numeric(abs(Matrix::crossprod(x_basis[, candidate_penalized_cols, drop = FALSE], y_centered)))
  norms <- sqrt(as.numeric(Matrix::colSums(x_basis[, candidate_penalized_cols, drop = FALSE]^2)))
  finite <- is.finite(scores) & is.finite(norms) & norms > 0
  scaled_scores <- rep(-Inf, length(scores))
  scaled_scores[finite] <- scores[finite] / norms[finite]
  scaled_scores
}

compute_linear_screen_residual <- function(raw_x, y) {
  if (is.null(raw_x)) {
    return(NULL)
  }

  raw_mat <- tryCatch(as.matrix(raw_x), error = function(...) NULL)
  if (is.null(raw_mat) || !nrow(raw_mat) || !ncol(raw_mat)) {
    return(NULL)
  }

  finite_col <- apply(raw_mat, 2L, function(col) all(is.finite(col)))
  if (!all(finite_col)) {
    raw_mat <- raw_mat[, finite_col, drop = FALSE]
  }
  if (!ncol(raw_mat)) {
    return(NULL)
  }

  fit <- tryCatch(stats::lm.fit(x = cbind(`(Intercept)` = 1, raw_mat), y = y), error = function(...) NULL)
  if (is.null(fit) || is.null(fit$residuals)) {
    return(NULL)
  }

  as.numeric(fit$residuals)
}

screen_basis_for_gaussian_fit <- function(x_basis, y, max_basis, penalty_factor,
                                          unpenalized_covariates = 0L,
                                          candidate_penalized_cols = NULL,
                                          raw_x = NULL,
                                          fit_control = list()) {
  p_total <- ncol(x_basis)
  penalized_cols <- seq_len(max(0L, p_total - unpenalized_covariates))
  unpenalized_cols <- if (unpenalized_covariates > 0L) {
    seq.int(p_total - unpenalized_covariates + 1L, p_total)
  } else {
    integer(0)
  }

  if (is.null(candidate_penalized_cols)) {
    candidate_penalized_cols <- penalized_cols
  } else {
    candidate_penalized_cols <- sort(intersect(as.integer(candidate_penalized_cols), penalized_cols))
  }

  if (length(candidate_penalized_cols) <= max_basis) {
    return(sort(c(candidate_penalized_cols, unpenalized_cols)))
  }

  primary_scores <- score_gaussian_screen_candidates(x_basis, y, candidate_penalized_cols)
  primary_ranked <- candidate_penalized_cols[order(primary_scores, decreasing = TRUE)]

  approx_linear_residual_screen <- fit_control$approx_linear_residual_screen %||% TRUE
  approx_linear_residual_ratio <- fit_control$approx_linear_residual_ratio %||% 0.35
  approx_linear_residual_min_basis <- as.integer(fit_control$approx_linear_residual_min_basis %||% 60L)

  keep_penalized <- primary_ranked[seq_len(max_basis)]

  if (isTRUE(approx_linear_residual_screen) && !is.null(raw_x)) {
    residual_y <- compute_linear_screen_residual(raw_x = raw_x, y = y)
    if (!is.null(residual_y)) {
      residual_scores <- score_gaussian_screen_candidates(x_basis, residual_y, candidate_penalized_cols)
      residual_ranked <- candidate_penalized_cols[order(residual_scores, decreasing = TRUE)]
      residual_budget <- min(
        length(candidate_penalized_cols),
        max_basis,
        max(approx_linear_residual_min_basis, ceiling(max_basis * approx_linear_residual_ratio))
      )
      residual_keep <- residual_ranked[seq_len(residual_budget)]
      keep_penalized <- unique(c(
        primary_ranked[seq_len(max(1L, max_basis - residual_budget))],
        residual_keep
      ))
      if (length(keep_penalized) < max_basis) {
        fill_cols <- setdiff(primary_ranked, keep_penalized)
        keep_penalized <- c(keep_penalized, fill_cols[seq_len(min(length(fill_cols), max_basis - length(keep_penalized)))])
      }
      keep_penalized <- keep_penalized[seq_len(min(length(keep_penalized), max_basis))]
    }
  }

  sort(c(keep_penalized, unpenalized_cols))
}

extract_selected_lambda_type <- function(fit_control) {
  if (isTRUE(fit_control$use_min)) "lambda.min" else "lambda.1se"
}

augment_screened_basis_for_gaussian_fit <- function(x_basis, y, screened_keep_cols,
                                                    stage1_fit, lambda_type,
                                                    refine_max_basis,
                                                    penalty_factor,
                                                    unpenalized_covariates = 0L) {
  p_total <- ncol(x_basis)
  penalized_cols <- seq_len(max(0L, p_total - unpenalized_covariates))
  screened_penalized <- sort(screened_keep_cols[screened_keep_cols <= length(penalized_cols)])
  omitted_penalized <- setdiff(penalized_cols, screened_penalized)

  if (!length(omitted_penalized) || refine_max_basis <= 0L) {
    return(sort(unique(screened_keep_cols)))
  }

  stage1_pred <- as.numeric(stats::predict(
    stage1_fit,
    newx = x_basis[, screened_keep_cols, drop = FALSE],
    s = lambda_type,
    type = "response"
  ))
  residual <- as.numeric(y - stage1_pred)

  refine_keep_cols <- screen_basis_for_gaussian_fit(
    x_basis = x_basis,
    y = residual,
    max_basis = min(as.integer(refine_max_basis), length(omitted_penalized)),
    penalty_factor = penalty_factor,
    unpenalized_covariates = unpenalized_covariates,
    candidate_penalized_cols = omitted_penalized
  )
  refine_penalized <- refine_keep_cols[refine_keep_cols <= length(penalized_cols)]
  sort(unique(c(screened_keep_cols, refine_penalized)))
}

compute_adaptive_refine_target <- function(fit_control,
                                           screen_target,
                                           penalized_count,
                                           stage1_fit,
                                           stage1_keep_cols,
                                           lambda_type,
                                           unpenalized_covariates = 0L) {
  approx_refine_ratio <- fit_control$approx_refine_ratio %||% 0.2
  approx_refine_max_basis <- as.integer(fit_control$approx_refine_max_basis %||% 80L)
  approx_refine_min_basis <- as.integer(fit_control$approx_refine_min_basis %||% 30L)
  approx_refine_hard_max_basis <- as.integer(fit_control$approx_refine_hard_max_basis %||% 160L)
  approx_refine_dense_threshold <- fit_control$approx_refine_dense_threshold %||% 0.18
  approx_refine_dense_multiplier <- fit_control$approx_refine_dense_multiplier %||% 2.0
  approx_refine_compression_threshold <- fit_control$approx_refine_compression_threshold %||% 1.8
  approx_refine_compression_multiplier <- fit_control$approx_refine_compression_multiplier %||% 1.5

  base_target <- max(
    approx_refine_min_basis,
    min(approx_refine_max_basis, ceiling(screen_target * approx_refine_ratio))
  )

  penalized_stage1_cols <- stage1_keep_cols[stage1_keep_cols <= penalized_count]
  stage1_coefs <- as.matrix(stats::coef(stage1_fit, s = lambda_type))
  penalized_coefs <- stage1_coefs[-1, , drop = FALSE]
  n_penalized_coef_rows <- max(0L, nrow(penalized_coefs) - unpenalized_covariates)
  active_penalized <- if (n_penalized_coef_rows > 0L) {
    sum(abs(penalized_coefs[seq_len(n_penalized_coef_rows), , drop = FALSE]) > 0)
  } else {
    0L
  }
  stage1_penalized_count <- max(1L, length(penalized_stage1_cols) - unpenalized_covariates)
  active_density <- active_penalized / stage1_penalized_count
  compression_ratio <- penalized_count / max(1L, screen_target)

  adaptive_multiplier <- 1
  if (is.finite(active_density) && active_density >= approx_refine_dense_threshold) {
    adaptive_multiplier <- max(adaptive_multiplier, approx_refine_dense_multiplier)
  }
  if (is.finite(compression_ratio) && compression_ratio >= approx_refine_compression_threshold) {
    adaptive_multiplier <- max(adaptive_multiplier, approx_refine_compression_multiplier)
  }

  min(
    penalized_count,
    approx_refine_hard_max_basis,
    max(base_target, ceiling(base_target * adaptive_multiplier))
  )
}

refit_augmented_basis_at_selected_lambda <- function(fit_control, x_basis,
                                                     penalty_factor, lambda_star,
                                                     family, Y, offset, weights) {
  fit_control$x <- x_basis
  fit_control$y <- Y
  fit_control$standardize <- FALSE
  fit_control$family <- family
  fit_control$lambda <- lambda_star
  fit_control$penalty.factor <- penalty_factor
  fit_control$offset <- offset
  fit_control$weights <- weights
  fit_control$cv_select <- NULL
  fit_control$use_min <- NULL
  do.call(glmnet::glmnet, fit_control)
}

compute_adaptive_gaussian_screen_target <- function(
  x_basis,
  basis_list,
  fit_control,
  feature_count,
  unpenalized_covariates = 0L
) {
  penalized_count <- length(basis_list)
  n_obs <- nrow(x_basis)

  approx_screen_ratio <- fit_control$approx_screen_ratio %||% 0.2
  approx_screen_max_basis <- as.integer(fit_control$approx_screen_max_basis %||% 600L)
  approx_screen_min_basis <- as.integer(fit_control$approx_screen_min_basis %||% 250L)
  approx_screen_n_multiplier <- fit_control$approx_screen_n_multiplier %||% 0.35
  approx_screen_geom_multiplier <- fit_control$approx_screen_geom_multiplier %||% 0.6

  adaptive_target <- max(
    approx_screen_min_basis,
    ceiling(penalized_count * approx_screen_ratio),
    ceiling(n_obs * approx_screen_n_multiplier),
    ceiling(sqrt(n_obs * penalized_count) * approx_screen_geom_multiplier)
  )

  approx_lowdim_p_threshold <- as.integer(fit_control$approx_lowdim_p_threshold %||% 8L)
  approx_lowdim_basis_per_feature_threshold <- fit_control$approx_lowdim_basis_per_feature_threshold %||% 180
  approx_lowdim_screen_ratio <- fit_control$approx_lowdim_screen_ratio %||% 0.45
  approx_lowdim_screen_n_multiplier <- fit_control$approx_lowdim_screen_n_multiplier %||% 0.8
  approx_lowdim_screen_geom_multiplier <- fit_control$approx_lowdim_screen_geom_multiplier %||% 1.1
  approx_lowdim_screen_max_basis <- as.integer(fit_control$approx_lowdim_screen_max_basis %||% 1200L)

  basis_per_feature <- penalized_count / max(1L, feature_count)
  structure_aware_mode <- feature_count <= approx_lowdim_p_threshold &&
    is.finite(basis_per_feature) &&
    basis_per_feature >= approx_lowdim_basis_per_feature_threshold

  if (structure_aware_mode) {
    structure_aware_target <- max(
      approx_screen_min_basis,
      ceiling(penalized_count * approx_lowdim_screen_ratio),
      ceiling(n_obs * approx_lowdim_screen_n_multiplier),
      ceiling(sqrt(n_obs * penalized_count) * approx_lowdim_screen_geom_multiplier)
    )
    adaptive_target <- max(adaptive_target, structure_aware_target)
    adaptive_target <- min(approx_lowdim_screen_max_basis, adaptive_target)
  } else {
    adaptive_target <- min(approx_screen_max_basis, adaptive_target)
  }
  adaptive_target <- min(penalized_count, adaptive_target)

  auto_min_basis <- as.integer(fit_control$approx_auto_min_basis %||% 600L)
  auto_min_ratio <- fit_control$approx_auto_min_ratio %||% 0.9
  auto_gap_ratio <- fit_control$approx_auto_gap_ratio %||% 1.2
  auto_min_excess <- as.integer(fit_control$approx_auto_min_excess %||% 100L)
  auto_trigger_basis <- max(
    auto_min_basis,
    ceiling(n_obs * auto_min_ratio),
    ceiling(adaptive_target * auto_gap_ratio),
    adaptive_target + auto_min_excess
  )

  list(
    use = penalized_count >= auto_trigger_basis,
    target = adaptive_target,
    trigger_basis = auto_trigger_basis,
    n_obs = n_obs,
    penalized_count = penalized_count,
    basis_per_feature = basis_per_feature,
    structure_aware_mode = structure_aware_mode
  )
}

should_use_approx_gaussian_backend <- function(fam, fit_control, lambda, weights,
                                               offset, x_basis, basis_list,
                                               feature_count) {
  approx_mode <- fit_control$approx_backend
  if (is.null(approx_mode)) {
    approx_mode <- "auto"
  }

  if (identical(approx_mode, FALSE) || identical(approx_mode, "off")) {
    return(FALSE)
  }

  eligible <- identical(fam, "gaussian") &&
    isTRUE(fit_control$cv_select) &&
    is.null(lambda) &&
    is.null(weights) &&
    is.null(offset) &&
    !is.null(x_basis) &&
    !is.null(basis_list)

  if (!eligible) {
    return(FALSE)
  }

  if (identical(approx_mode, TRUE) || identical(approx_mode, "on") || identical(approx_mode, "force")) {
    return(TRUE)
  }

  adaptive_screen <- compute_adaptive_gaussian_screen_target(
    x_basis = x_basis,
    basis_list = basis_list,
    fit_control = fit_control,
    feature_count = feature_count
  )

  approx_selective_regime <- fit_control$approx_selective_regime %||% TRUE
  approx_selective_min_n <- as.integer(fit_control$approx_selective_min_n %||% 350L)
  approx_selective_max_p <- as.integer(fit_control$approx_selective_max_p %||% 8L)
  approx_selective_min_basis <- as.integer(fit_control$approx_selective_min_basis %||% 650L)

  if (isTRUE(approx_selective_regime)) {
    selective_ok <- adaptive_screen$n_obs >= approx_selective_min_n &&
      feature_count <= approx_selective_max_p &&
      adaptive_screen$penalized_count >= approx_selective_min_basis
    if (!selective_ok) {
      return(FALSE)
    }
  }

  adaptive_screen$use
}

`%||%` <- function(x, y) if (is.null(x)) y else x

fit_hal <- function(X,
                    Y,
                    formula = NULL,
                    X_unpenalized = NULL,
                    max_degree = ifelse(ncol(X) >= 20, 2, 3),
                    smoothness_orders = 1,
                    num_knots = num_knots_generator(
                      max_degree = max_degree,
                      smoothness_orders = smoothness_orders,
                      base_num_knots_0 = 20,
                      base_num_knots_1 = 10
                    ),
                    reduce_basis = NULL,
                    family = c("gaussian", "binomial", "poisson", "cox",
                               "mgaussian"),
                    lambda = NULL,
                    id = NULL,
                    weights = NULL,
                    offset = NULL,
                    fit_control = list(
                      cv_select = TRUE,
                      use_min = TRUE,
                      lambda.min.ratio = 1e-4,
                      prediction_bounds = "default",
                      approx_backend = "auto",
                      approx_auto_min_basis = 600L,
                      approx_auto_min_ratio = 0.9,
                      approx_auto_gap_ratio = 1.2,
                      approx_auto_min_excess = 100L,
                      approx_screen_ratio = 0.2,
                      approx_screen_n_multiplier = 0.35,
                      approx_screen_geom_multiplier = 0.6,
                      approx_screen_max_basis = 600L,
                      approx_screen_min_basis = 250L,
                      approx_lowdim_p_threshold = 8L,
                      approx_lowdim_basis_per_feature_threshold = 180,
                      approx_lowdim_screen_ratio = 0.45,
                      approx_lowdim_screen_n_multiplier = 0.8,
                      approx_lowdim_screen_geom_multiplier = 1.1,
                      approx_lowdim_screen_max_basis = 1200L,
                      approx_refine_ratio = 0.2,
                      approx_refine_max_basis = 80L,
                      approx_refine_min_basis = 30L,
                      approx_refine_hard_max_basis = 160L,
                      approx_refine_dense_threshold = 0.18,
                      approx_refine_dense_multiplier = 2.0,
                      approx_refine_compression_threshold = 1.8,
                      approx_refine_compression_multiplier = 1.5,
                      approx_linear_residual_screen = TRUE,
                      approx_linear_residual_ratio = 0.35,
                      approx_linear_residual_min_basis = 60L,
                      approx_selective_regime = TRUE,
                      approx_selective_min_n = 350L,
                      approx_selective_max_p = 8L,
                      approx_selective_min_basis = 650L
                    ),
                    basis_list = NULL,
                    return_lasso = TRUE,
                    return_x_basis = FALSE,
                    yolo = FALSE) {
  if (!inherits(family, "family")) {
    family <- match.arg(family)
  }
  fam <- ifelse(inherits(family, "family"), family$family, family)

  # errors when a supplied control list is missing arguments
  defaults <- list(
    cv_select = TRUE,
    use_min = TRUE,
    lambda.min.ratio = 1e-4,
    prediction_bounds = "default",
    approx_backend = "auto",
    approx_auto_min_basis = 600L,
    approx_auto_min_ratio = 0.9,
    approx_auto_gap_ratio = 1.2,
    approx_auto_min_excess = 100L,
    approx_screen_ratio = 0.2,
    approx_screen_n_multiplier = 0.35,
    approx_screen_geom_multiplier = 0.6,
    approx_screen_max_basis = 600L,
    approx_screen_min_basis = 250L,
    approx_lowdim_p_threshold = 8L,
    approx_lowdim_basis_per_feature_threshold = 180,
    approx_lowdim_screen_ratio = 0.45,
    approx_lowdim_screen_n_multiplier = 0.8,
    approx_lowdim_screen_geom_multiplier = 1.1,
    approx_lowdim_screen_max_basis = 1200L,
    approx_refine_ratio = 0.2,
    approx_refine_max_basis = 80L,
    approx_refine_min_basis = 30L,
    approx_refine_hard_max_basis = 160L,
    approx_refine_dense_threshold = 0.18,
    approx_refine_dense_multiplier = 2.0,
    approx_refine_compression_threshold = 1.8,
    approx_refine_compression_multiplier = 1.5,
    approx_linear_residual_screen = TRUE,
    approx_linear_residual_ratio = 0.35,
    approx_linear_residual_min_basis = 60L,
    approx_selective_regime = TRUE,
    approx_selective_min_n = 350L,
    approx_selective_max_p = 8L,
    approx_selective_min_basis = 650L
  )
  if (any(!names(defaults) %in% names(fit_control))) {
    fit_control <- c(
      defaults[!names(defaults) %in% names(fit_control)], fit_control
    )
  }
  # check fit_control names (exluding defaults) are glmnet/cv.glmnet formals
  glmnet_formals <- unique(c(
    names(formals(glmnet::cv.glmnet)),
    names(formals(glmnet::glmnet)),
    names(formals(glmnet::relax.glmnet)) # extra allowed args to glmnet
  ))
  control_names <- names(fit_control[!names(fit_control) %in% names(defaults)])
  if (any(!control_names %in% glmnet_formals)) {
    bad_args <- control_names[(!control_names %in% glmnet_formals)]
    warning(sprintf(
      "Some fit_control arguments are neither default nor glmnet/cv.glmnet arguments: %s; \nThey will be removed from fit_control",
      paste0(bad_args, collapse = ", ")
    ))
    fit_control <- fit_control[!names(fit_control) %in% bad_args]
  }

  if (!is.matrix(X)) X <- as.matrix(X)

  # check for missingness and ensure dimensionality matches
  assertthat::assert_that(
    all(!is.na(X)),
    msg = "NA detected in `X`, missingness in `X` is not supported"
  )
  assertthat::assert_that(
    all(!is.na(Y)),
    msg = "NA detected in `Y`, missingness in `Y` is not supported"
  )

  n_Y <- ifelse(is.matrix(Y), nrow(Y), length(Y))
  assertthat::assert_that(
    nrow(X) == n_Y,
    msg = "Number of rows in `X` and `Y` must be equal"
  )

  if (!is.null(X_unpenalized)) {
    assertthat::assert_that(
      all(!is.na(X_unpenalized)),
      msg = paste(
        "NA detected in `X_unpenalized`, missingness",
        "in `X_unpenalized` is not supported."
      )
    )
    assertthat::assert_that(
      nrow(X) == nrow(X_unpenalized),
      msg = paste(
        "Number of rows in `X` and `X_unpenalized`,",
        "and length of `Y` must be equal."
      )
    )
  }

  if (!is.character(fit_control$prediction_bounds)) {
    if (fam == "mgaussian") {
      assertthat::assert_that(
        is.list(fit_control$prediction_bounds) &
          length(fit_control$prediction_bounds) == ncol(Y),
        msg = "prediction_bounds must be 'default' or list of numeric (lower, upper) bounds for each outcome"
      )
    } else {
      assertthat::assert_that(
        is.numeric(fit_control$prediction_bounds) &
          length(fit_control$prediction_bounds) == 2,
        msg = "prediction_bounds must be 'default' or numeric (lower, upper) bounds"
      )
    }
  }



  if (!is.null(formula)) {
    # formula <- formula_hal(
    #   formula = formula, X = X, smoothness_orders = smoothness_orders,
    #   num_knots = num_knots, exclusive_dot = formula_control$exclusive_dot,
    #   custom_group = formula_control$custom_group
    # )

    if (!inherits(formula, "formula_hal")) {
      formula <- formula_hal(
        formula,
        X = X, smoothness_orders = smoothness_orders,
        num_knots = num_knots
      )
    }
    basis_list <- formula$basis_list
    fit_control$upper.limits <- formula$upper.limits
    fit_control$lower.limits <- formula$lower.limits
    penalty_factor <- formula$penalty_factors
  } else {
    penalty_factor <- NULL
  }

  # FUN! Quotes from HAL 9000, the robot from the film "2001: A Space Odyssey"
  if (yolo) hal9000()

  # Generate fold_ids that respect id
  if (is.null(fit_control$foldid)) {
    if (is.null(fit_control$nfolds)) fit_control$nfolds <- 10
    folds <- origami::make_folds(
      n = n_Y, V = fit_control$nfolds, cluster_ids = id
    )
    fit_control$foldid <- origami::folds2foldvec(folds)
  }

  # bookkeeping: get start time of enumerate basis procedure
  time_start <- proc.time()

  # enumerate basis functions for making HAL design matrix
  if (is.null(basis_list)) {
    basis_list <- enumerate_basis(
      X,
      max_degree = max_degree,
      smoothness_orders = smoothness_orders,
      num_knots = num_knots,
      include_lower_order = FALSE,
      include_zero_order = FALSE
    )
  }

  # bookkeeping: get end time of enumerate basis procedure
  time_enumerate_basis <- proc.time()

  # make design matrix for HAL from basis functions
  x_basis <- make_design_matrix(X, basis_list)

  # bookkeeping: get end time of design matrix procedure
  time_design_matrix <- proc.time()

  # NOTE: keep only basis functions with some (or higher) proportion of 1's
  if (all(smoothness_orders == 0)) {
    if (is.null(reduce_basis)) {
      reduce_basis <- 1 / sqrt(n_Y)
    }
    reduced_basis_map <- make_reduced_basis_map(x_basis, reduce_basis)
    x_basis <- x_basis[, reduced_basis_map]
    basis_list <- basis_list[reduced_basis_map]
  } else {
    if (!is.null(reduce_basis)) {
      warning("Dropping reduce_basis; only applies if smoothness_orders = 0")
    }
  }

  time_reduce_basis <- proc.time()

  # catalog and eliminate duplicates
  # Lars's change: copy_map is not needed but to preserve functionality (e.g.,
  # summary), pass in a trivial copy_map.
  if (all(smoothness_orders == 0)) {
    copy_map <- make_copy_map(x_basis)
    unique_columns <- as.numeric(names(copy_map))
    x_basis <- x_basis[, unique_columns]
    basis_list <- basis_list[unique_columns]
  }
  copy_map <- seq_along(basis_list)
  names(copy_map) <- seq_along(basis_list)

  # bookkeeping: get end time of duplicate removal procedure
  time_rm_duplicates <- proc.time()

  # generate a vector of col names
  if (!is.null(colnames(X))) {
    X_colnames <- colnames(X)
  } else {
    X_colnames <- paste0("x", 1:ncol(X))
  }

  # the HAL basis are subject to L1 penalty
  if (is.null(penalty_factor)) {
    penalty_factor <- rep(1, ncol(x_basis))
  }

  unpenalized_covariates <- ifelse(
    test = is.null(X_unpenalized),
    yes = 0,
    no = {
      assertthat::assert_that(is.matrix(X_unpenalized))
      assertthat::assert_that(nrow(X_unpenalized) == nrow(x_basis))
      ncol(X_unpenalized)
    }
  )
  if (unpenalized_covariates > 0) {
    x_basis <- cbind(x_basis, X_unpenalized)
    penalty_factor <- c(penalty_factor, rep(0, ncol(X_unpenalized)))
  }

  # NOTE: workaround for "Cox model not implemented for sparse x in glmnet"
  #       casting to a regular (dense) matrix has a large memory cost :(
  # General families throws warnings if you pass in sparse matrix and does not
  # seem to lead to speed benefit.
  # I'm guessing glmnet internally converts to matrix.
  # if (inherits(family, "family") || family == "cox") {
  #   x_basis <- as.matrix(x_basis)
  # }

  if (fam == "cox") {
    x_basis <- as.matrix(x_basis)
  }

  # bookkeeping: get start time of lasso
  time_start_lasso <- proc.time()

  # fit lasso regression
  # If the standardize argument is passed to glmnet through "...", simply
  # note that it will be discarded and set to FALSE.
  if ("standardize" %in% names(fit_control)) {
    message(
      "Argument `standardize` to `glmnet` detected, overriding to `FALSE`."
    )
  }

  # just use the standard implementation available in glmnet
  approx_fit_meta <- list(
    used = FALSE,
    original_basis_count = ncol(x_basis),
    screened_basis_count = ncol(x_basis),
    refine_basis_count = 0L,
    final_basis_count = ncol(x_basis),
    auto_trigger_basis = NA_integer_,
    screen_target = ncol(x_basis)
  )
  hal_lasso <- NULL
  lambda_star <- NULL
  coefs <- NULL

  if (should_use_approx_gaussian_backend(
    fam = fam,
    fit_control = fit_control,
    lambda = lambda,
    weights = weights,
    offset = offset,
    x_basis = x_basis,
    basis_list = basis_list,
    feature_count = ncol(X)
  )) {
    penalized_count <- length(basis_list)
    adaptive_screen <- compute_adaptive_gaussian_screen_target(
      x_basis = x_basis,
      basis_list = basis_list,
      fit_control = fit_control,
      feature_count = ncol(X),
      unpenalized_covariates = unpenalized_covariates
    )
    screen_target <- adaptive_screen$target

    if (screen_target < penalized_count) {
      stage1_keep_cols <- screen_basis_for_gaussian_fit(
        x_basis = x_basis,
        y = Y,
        max_basis = screen_target,
        penalty_factor = penalty_factor,
        unpenalized_covariates = unpenalized_covariates,
        raw_x = X,
        fit_control = fit_control
      )

      stage1_fit_control <- fit_control
      stage1_fit_control$x <- x_basis[, stage1_keep_cols, drop = FALSE]
      stage1_fit_control$y <- Y
      stage1_fit_control$standardize <- FALSE
      stage1_fit_control$family <- family
      stage1_fit_control$lambda <- lambda
      stage1_fit_control$penalty.factor <- penalty_factor[stage1_keep_cols]
      stage1_fit_control$offset <- offset
      stage1_fit_control$weights <- weights
      stage1_fit_control$approx_backend <- NULL
      stage1_fit_control$approx_auto_min_basis <- NULL
      stage1_fit_control$approx_auto_min_ratio <- NULL
      stage1_fit_control$approx_auto_gap_ratio <- NULL
      stage1_fit_control$approx_auto_min_excess <- NULL
      stage1_fit_control$approx_screen_ratio <- NULL
      stage1_fit_control$approx_screen_n_multiplier <- NULL
      stage1_fit_control$approx_screen_geom_multiplier <- NULL
      stage1_fit_control$approx_screen_max_basis <- NULL
      stage1_fit_control$approx_screen_min_basis <- NULL
      stage1_fit_control$approx_lowdim_p_threshold <- NULL
      stage1_fit_control$approx_lowdim_basis_per_feature_threshold <- NULL
      stage1_fit_control$approx_lowdim_screen_ratio <- NULL
      stage1_fit_control$approx_lowdim_screen_n_multiplier <- NULL
      stage1_fit_control$approx_lowdim_screen_geom_multiplier <- NULL
      stage1_fit_control$approx_lowdim_screen_max_basis <- NULL
      stage1_fit_control$approx_refine_ratio <- NULL
      stage1_fit_control$approx_refine_max_basis <- NULL
      stage1_fit_control$approx_refine_min_basis <- NULL
      stage1_fit_control$approx_refine_hard_max_basis <- NULL
      stage1_fit_control$approx_refine_dense_threshold <- NULL
      stage1_fit_control$approx_refine_dense_multiplier <- NULL
      stage1_fit_control$approx_refine_compression_threshold <- NULL
      stage1_fit_control$approx_refine_compression_multiplier <- NULL
      stage1_fit_control$approx_linear_residual_screen <- NULL
      stage1_fit_control$approx_linear_residual_ratio <- NULL
      stage1_fit_control$approx_linear_residual_min_basis <- NULL
      stage1_fit_control$approx_selective_regime <- NULL
      stage1_fit_control$approx_selective_min_n <- NULL
      stage1_fit_control$approx_selective_max_p <- NULL
      stage1_fit_control$approx_selective_min_basis <- NULL
      stage1_fit <- do.call(glmnet::cv.glmnet, stage1_fit_control)
      lambda_type <- extract_selected_lambda_type(fit_control)
      lambda_star <- if (identical(lambda_type, "lambda.min")) stage1_fit$lambda.min else stage1_fit$lambda.1se

      refine_target <- compute_adaptive_refine_target(
        fit_control = fit_control,
        screen_target = screen_target,
        penalized_count = penalized_count,
        stage1_fit = stage1_fit,
        stage1_keep_cols = stage1_keep_cols,
        lambda_type = lambda_type,
        unpenalized_covariates = unpenalized_covariates
      )

      keep_cols <- augment_screened_basis_for_gaussian_fit(
        x_basis = x_basis,
        y = Y,
        screened_keep_cols = stage1_keep_cols,
        stage1_fit = stage1_fit,
        lambda_type = lambda_type,
        refine_max_basis = refine_target,
        penalty_factor = penalty_factor,
        unpenalized_covariates = unpenalized_covariates
      )
      refine_penalized_count <- length(setdiff(
        keep_cols[keep_cols <= penalized_count],
        stage1_keep_cols[stage1_keep_cols <= penalized_count]
      ))

      x_basis <- x_basis[, keep_cols, drop = FALSE]
      penalty_factor <- penalty_factor[keep_cols]
      penalized_keep <- keep_cols[keep_cols <= length(basis_list)]
      basis_list <- basis_list[penalized_keep]
      copy_map <- seq_along(basis_list)
      names(copy_map) <- seq_along(basis_list)
      hal_lasso <- refit_augmented_basis_at_selected_lambda(
        fit_control = fit_control,
        x_basis = x_basis,
        penalty_factor = penalty_factor,
        lambda_star = lambda_star,
        family = family,
        Y = Y,
        offset = offset,
        weights = weights
      )
      coefs <- stats::coef(hal_lasso)
      approx_fit_meta <- list(
        used = TRUE,
        original_basis_count = penalized_count,
        screened_basis_count = length(stage1_keep_cols[stage1_keep_cols <= penalized_count]),
        refine_basis_count = refine_penalized_count,
        final_basis_count = length(penalized_keep),
        auto_trigger_basis = adaptive_screen$trigger_basis,
        screen_target = screen_target
      )
    } else {
      approx_fit_meta <- list(
        used = FALSE,
        original_basis_count = penalized_count,
        screened_basis_count = penalized_count,
        refine_basis_count = 0L,
        final_basis_count = penalized_count,
        auto_trigger_basis = adaptive_screen$trigger_basis,
        screen_target = penalized_count
      )
    }
  }

  fit_control$x <- x_basis
  fit_control$y <- Y
  fit_control$standardize <- FALSE
  fit_control$family <- family
  fit_control$lambda <- lambda
  fit_control$penalty.factor <- penalty_factor
  fit_control$offset <- offset
  fit_control$weights <- weights

  fit_control$approx_backend <- NULL
  fit_control$approx_auto_min_basis <- NULL
  fit_control$approx_auto_min_ratio <- NULL
  fit_control$approx_auto_gap_ratio <- NULL
  fit_control$approx_auto_min_excess <- NULL
  fit_control$approx_screen_ratio <- NULL
  fit_control$approx_screen_n_multiplier <- NULL
  fit_control$approx_screen_geom_multiplier <- NULL
  fit_control$approx_screen_max_basis <- NULL
  fit_control$approx_screen_min_basis <- NULL
  fit_control$approx_lowdim_p_threshold <- NULL
  fit_control$approx_lowdim_basis_per_feature_threshold <- NULL
  fit_control$approx_lowdim_screen_ratio <- NULL
  fit_control$approx_lowdim_screen_n_multiplier <- NULL
  fit_control$approx_lowdim_screen_geom_multiplier <- NULL
  fit_control$approx_lowdim_screen_max_basis <- NULL
  fit_control$approx_refine_ratio <- NULL
  fit_control$approx_refine_max_basis <- NULL
  fit_control$approx_refine_min_basis <- NULL
  fit_control$approx_refine_hard_max_basis <- NULL
  fit_control$approx_refine_dense_threshold <- NULL
  fit_control$approx_refine_dense_multiplier <- NULL
  fit_control$approx_refine_compression_threshold <- NULL
  fit_control$approx_refine_compression_multiplier <- NULL
  fit_control$approx_linear_residual_screen <- NULL
  fit_control$approx_linear_residual_ratio <- NULL
  fit_control$approx_linear_residual_min_basis <- NULL
  fit_control$approx_selective_regime <- NULL
  fit_control$approx_selective_min_n <- NULL
  fit_control$approx_selective_max_p <- NULL
  fit_control$approx_selective_min_basis <- NULL

  if (is.null(hal_lasso)) {
    if (!fit_control$cv_select) {
      hal_lasso <- do.call(glmnet::glmnet, fit_control)
      lambda_star <- hal_lasso$lambda
      coefs <- stats::coef(hal_lasso)
    } else {
      hal_lasso <- do.call(glmnet::cv.glmnet, fit_control)
      if (fit_control$use_min) {
        lambda_type <- "lambda.min"
        lambda_star <- hal_lasso$lambda.min
      } else {
        lambda_type <- "lambda.1se"
        lambda_star <- hal_lasso$lambda.1se
      }
      coefs <- stats::coef(hal_lasso, lambda_type)
    }
  }

  # bookkeeping: get time for computation of the lasso regression
  time_lasso <- proc.time()

  # bookkeeping: get time for the whole procedure
  time_final <- proc.time()

  # bookkeeping: construct table for viewing procedure times
  times <- rbind(
    enumerate_basis = time_enumerate_basis - time_start,
    design_matrix = time_design_matrix - time_enumerate_basis,
    reduce_basis = time_reduce_basis - time_design_matrix,
    remove_duplicates = time_rm_duplicates - time_reduce_basis,
    lasso = time_lasso - time_start_lasso,
    total = time_final - time_start
  )

  # Bounds for prediction on new data (to prevent extrapolation for linear HAL)
  if (is.character(fit_control$prediction_bounds) &&
    fit_control$prediction_bounds == "default") {
    if (fam == "mgaussian") {
      fit_control$prediction_bounds <- lapply(seq(ncol(Y)), function(i) {
        c(min(Y[, i]) - 2 * stats::sd(Y[, i]), max(Y[, i]) + 2 * stats::sd(Y[, i]))
      })
    } else if (fam == "cox") {
      fit_control$prediction_bounds <- NULL
    } else {
      fit_control$prediction_bounds <- c(
        min(Y) - 2 * stats::sd(Y), max(Y) + 2 * stats::sd(Y)
      )
    }
  }

  # construct output object via lazy S3 list
  fit <- list(
    x_basis =
      if (return_x_basis) {
        x_basis
      } else {
        NULL
      },
    basis_list = basis_list,
    X_colnames = X_colnames,
    copy_map = copy_map,
    coefs = as.matrix(coefs),
    times = times,
    lambda_star = lambda_star,
    reduce_basis = reduce_basis,
    family = family,
    lasso_fit =
      if (return_lasso) {
        hal_lasso
      } else {
        NULL
      },
    unpenalized_covariates = unpenalized_covariates,
    prediction_bounds = fit_control$prediction_bounds,
    approx_fit = approx_fit_meta
  )
  class(fit) <- "hal9001"
  return(fit)
}

###############################################################################

#' A default generator for the \code{num_knots} argument for each degree of
#' interactions and the smoothness orders.
#'
#' @param max_degree interaction degree.
#' @param smoothness_orders see \code{\link{fit_hal}}.
#' @param base_num_knots_0 The base number of knots for zeroth-order smoothness
#'  basis functions. The number of knots by degree interaction decays as
#'  `base_num_knots_0/2^(d-1)` where `d` is the interaction degree of the basis
#'  function.
#' @param base_num_knots_1 The base number of knots for 1 or greater order
#'  smoothness basis functions. The number of knots by degree interaction
#'  decays as `base_num_knots_1/2^(d-1)` where `d` is the interaction degree of
#'  the basis function.
#'
#' @keywords internal
num_knots_generator <- function(max_degree, smoothness_orders, base_num_knots_0 = 500,
                                base_num_knots_1 = 200) {
  if (all(smoothness_orders > 0)) {
    return(sapply(seq_len(max_degree), function(d) {
      round(base_num_knots_1 / 2^(d - 1))
    }))
  } else {
    return(sapply(seq_len(max_degree), function(d) {
      round(base_num_knots_0 / 2^(d - 1))
    }))
  }
}
