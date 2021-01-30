#ifndef STAN_VARIATIONAL_ADVI_HPP
#define STAN_VARIATIONAL_ADVI_HPP

#include <stan/math.hpp>
#include <stan/analyze/mcmc/autocovariance.hpp>
#include <stan/analyze/mcmc/compute_effective_sample_size.hpp>
#include <stan/analyze/mcmc/compute_potential_scale_reduction.hpp>
#include <stan/analyze/mcmc/estimate_gpd_params.hpp>
#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <stan/io/dump.hpp>
#include <stan/services/error_codes.hpp>
#include <stan/variational/print_progress.hpp>
#include <stan/variational/families/normal_fullrank.hpp>
#include <stan/variational/families/normal_meanfield.hpp>
#include <boost/circular_buffer.hpp>
#include <boost/lexical_cast.hpp>
#include <algorithm>
#include <chrono>
#include <limits>
#include <numeric>
#include <ostream>
#include <queue>
#include <string>
#include <vector>
#include <cmath>

namespace stan {

namespace variational {

/**
 * Automatic Differentiation Variational Inference
 *
 * Implements "black box" variational inference using stochastic gradient
 * ascent to maximize the Evidence Lower Bound for a given model
 * and variational family.
 *
 * @tparam Model class of model
 * @tparam Q class of variational distribution
 * @tparam BaseRNG class of random number generator
 */
template <class Model, class Q, class BaseRNG>
class advi {
 public:
  /**
   * Constructor
   *
   * @param[in] m stan model
   * @param[in] cont_params initialization of continuous parameters
   * @param[in,out] rng random number generator
   * @param[in] n_monte_carlo_grad number of samples for gradient computation
   * @param[in] n_monte_carlo_elbo number of samples for ELBO computation
   * @param[in] n_posterior_samples number of samples to draw from posterior
   * @throw std::runtime_error if n_monte_carlo_grad is not positive
   * @throw std::runtime_error if n_monte_carlo_elbo is not positive
   * @throw std::runtime_error if n_posterior_samples is not positive
   */
  advi(Model& m, Eigen::VectorXd& cont_params, BaseRNG& rng,
       int n_monte_carlo_grad, int n_monte_carlo_elbo,
       int n_posterior_samples)
      : model_(m),
        cont_params_(cont_params),
        rng_(rng),
        n_monte_carlo_grad_(n_monte_carlo_grad),
        n_monte_carlo_elbo_(n_monte_carlo_elbo),
        n_posterior_samples_(n_posterior_samples) {
    static const char* function = "stan::variational::advi";
    math::check_positive(function,
                         "Number of Monte Carlo samples for gradients",
                         n_monte_carlo_grad_);
    math::check_positive(function, "Number of Monte Carlo samples for ELBO",
                         n_monte_carlo_elbo_);
    math::check_positive(function, "Number of posterior samples for output",
                         n_posterior_samples_);
  }

  /**
   * Calculates the Evidence Lower BOund (ELBO) by sampling from
   * the variational distribution and then evaluating the log joint,
   * adjusted by the entropy term of the variational distribution.
   *
   * @param[in] variational variational approximation at which to evaluate
   * the ELBO.
   * @param logger logger for messages
   * @return the evidence lower bound.
   * @throw std::domain_error If, after n_monte_carlo_elbo_ number of draws
   * from the variational distribution all give non-finite log joint
   * evaluations. This means that the model is severely ill conditioned or
   * that the variational distribution has somehow collapsed.
   */
  double calc_ELBO(const Q& variational, callbacks::logger& logger) const {
    static const char* function = "stan::variational::advi::calc_ELBO";

    double elbo = 0.0;
    int dim = variational.dimension();
    Eigen::VectorXd zeta(dim);

    int n_dropped_evaluations = 0;
    for (int i = 0; i < n_monte_carlo_elbo_;) {
      variational.sample(rng_, zeta);
      try {
        std::stringstream ss;
        double log_prob = model_.template log_prob<false, true>(zeta, &ss);
        if (ss.str().length() > 0)
          logger.info(ss);
        stan::math::check_finite(function, "log_prob", log_prob);
        elbo += log_prob;
        ++i;
      } catch (const std::domain_error& e) {
        ++n_dropped_evaluations;
        if (n_dropped_evaluations >= n_monte_carlo_elbo_) {
          const char* name = "The number of dropped evaluations";
          const char* msg1 = "has reached its maximum amount (";
          const char* msg2
              = "). Your model may be either severely "
                "ill-conditioned or misspecified.";
          stan::math::throw_domain_error(function, name, n_monte_carlo_elbo_,
                                         msg1, msg2);
        }
      }
    }
    elbo /= n_monte_carlo_elbo_;
    elbo += variational.entropy();
    return elbo;
  }

  /**
   * Calculates the "black box" gradient of the ELBO.
   *
   * @param[in] variational variational approximation at which to evaluate
   * the ELBO.
   * @param[out] elbo_grad gradient of ELBO with respect to variational
   * approximation.
   * @param logger logger for messages
   */
  void calc_ELBO_grad(const Q& variational, Q& elbo_grad,
                      callbacks::logger& logger) const {
    static const char* function = "stan::variational::advi::calc_ELBO_grad";

    stan::math::check_size_match(
        function, "Dimension of elbo_grad", elbo_grad.dimension(),
        "Dimension of variational q", variational.dimension());
    stan::math::check_size_match(
        function, "Dimension of variational q", variational.dimension(),
        "Dimension of variables in model", cont_params_.size());

    variational.calc_grad(elbo_grad, model_, cont_params_, n_monte_carlo_grad_,
                          rng_, logger);
  }

  /**
   * Heuristic grid search to adapt eta to the scale of the problem.
   *
   * @param[in] variational initial variational distribution.
   * @param[in] adapt_iterations number of iterations to spend doing stochastic
   * gradient ascent at each proposed eta value.
   * @param[in,out] logger logger for messages
   * @return adapted (tuned) value of eta via heuristic grid search
   * @throw std::domain_error If either (a) the initial ELBO cannot be
   * computed at the initial variational distribution, (b) all step-size
   * proposals in eta_sequence fail.
   */
  double adapt_eta(Q& variational, int adapt_iterations,
                   callbacks::logger& logger) const {
    static const char* function = "stan::variational::advi::adapt_eta";

    stan::math::check_positive(function, "Number of adaptation iterations",
                               adapt_iterations);

    logger.info("Begin eta adaptation.");

    // Sequence of eta values to try during adaptation
    const int eta_sequence_size = 5;
    double eta_sequence[eta_sequence_size] = {100, 10, 1, 0.1, 0.01};

    // Initialize ELBO tracking variables
    double elbo = -std::numeric_limits<double>::max();
    double elbo_best = -std::numeric_limits<double>::max();
    double elbo_init;
    try {
      elbo_init = calc_ELBO(variational, logger);
    } catch (const std::domain_error& e) {
      const char* name
          = "Cannot compute ELBO using the initial "
            "variational distribution.";
      const char* msg1
          = "Your model may be either "
            "severely ill-conditioned or misspecified.";
      stan::math::throw_domain_error(function, name, "", msg1);
    }

    // Variational family to store gradients
    Q elbo_grad = Q(model_.num_params_r());

    // Adaptive step-size sequence
    Q history_grad_squared = Q(model_.num_params_r());
    double tau = 1.0;
    double pre_factor = 0.9;
    double post_factor = 0.1;

    double eta_best = 0.0;
    double eta;
    double eta_scaled;

    bool do_more_tuning = true;
    int eta_sequence_index = 0;
    while (do_more_tuning) {
      // Try next eta
      eta = eta_sequence[eta_sequence_index];

      int print_progress_m;
      for (int iter_tune = 1; iter_tune <= adapt_iterations; ++iter_tune) {
        print_progress_m = eta_sequence_index * adapt_iterations + iter_tune;
        variational ::print_progress(print_progress_m, 0,
                                     adapt_iterations * eta_sequence_size,
                                     adapt_iterations, true, "", "", logger);

        // (ROBUST) Compute gradient of ELBO. It's OK if it diverges.
        // We'll try a smaller eta.
        try {
          calc_ELBO_grad(variational, elbo_grad, logger);
        } catch (const std::domain_error& e) {
          elbo_grad.set_to_zero();
        }

        // Update step-size
        if (iter_tune == 1) {
          history_grad_squared += elbo_grad.square();
        } else {
          history_grad_squared = pre_factor * history_grad_squared
                                 + post_factor * elbo_grad.square();
        }
        eta_scaled = eta / sqrt(static_cast<double>(iter_tune));
        // Stochastic gradient update
        variational
            += eta_scaled * elbo_grad / (tau + history_grad_squared.sqrt());
      }

      // (ROBUST) Compute ELBO. It's OK if it has diverged.
      try {
        elbo = calc_ELBO(variational, logger);
      } catch (const std::domain_error& e) {
        elbo = -std::numeric_limits<double>::max();
      }

      // Check if:
      // (1) ELBO at current eta is worse than the best ELBO
      // (2) the best ELBO hasn't gotten worse than the initial ELBO
      if (elbo < elbo_best && elbo_best > elbo_init) {
        std::stringstream ss;
        ss << "Success!"
           << " Found best value [eta = " << eta_best << "]";
        if (eta_sequence_index < eta_sequence_size - 1)
          ss << (" earlier than expected.");
        else
          ss << ".";
        logger.info(ss);
        logger.info("");
        do_more_tuning = false;
      } else {
        if (eta_sequence_index < eta_sequence_size - 1) {
          // Reset
          elbo_best = elbo;
          eta_best = eta;
        } else {
          // No more eta values to try, so use current eta if it
          // didn't diverge or fail if it did diverge
          if (elbo > elbo_init) {
            std::stringstream ss;
            ss << "Success!"
               << " Found best value [eta = " << eta_best << "].";
            logger.info(ss);
            logger.info("");
            eta_best = eta;
            do_more_tuning = false;
          } else {
            const char* name = "All proposed step-sizes";
            const char* msg1
                = "failed. Your model may be either "
                  "severely ill-conditioned or misspecified.";
            stan::math::throw_domain_error(function, name, "", msg1);
          }
        }
        // Reset
        history_grad_squared.set_to_zero();
      }
      ++eta_sequence_index;
      variational = Q(cont_params_);
    }
    return eta_best;
  }

  /**
   * Runs stochastic gradient ascent with an adaptive stepsize sequence.
   *
   * @param[in,out] variational initial variational distribution
   * @param[in] eta stepsize scaling parameter
   * @param[in] tol_rel_obj relative tolerance parameter for convergence
   * @param[in] max_iterations max number of iterations to run algorithm
   * @param[in,out] logger logger for messages
   * @param[in,out] diagnostic_writer writer for diagnostic information
   * @throw std::domain_error If the ELBO or its gradient is ever
   * non-finite, at any iteration
   */
  /*void stochastic_gradient_ascent(Q& variational, double eta,
                                  double tol_rel_obj, int max_iterations,
                                  callbacks::logger& logger,
                                  callbacks::writer& diagnostic_writer) const {
    static const char* function
        = "stan::variational::advi::stochastic_gradient_ascent";

    stan::math::check_positive(function, "Eta stepsize", eta);
    stan::math::check_positive(
        function, "Relative objective function tolerance", tol_rel_obj);
    stan::math::check_positive(function, "Maximum iterations", max_iterations);

    // Gradient parameters
    Q elbo_grad = Q(model_.num_params_r());

    // Stepsize sequence parameters
    Q history_grad_squared = Q(model_.num_params_r());
    double tau = 1.0;
    double pre_factor = 0.9;
    double post_factor = 0.1;
    double eta_scaled;

    // Initialize ELBO and convergence tracking variables
    double elbo(0.0);
    double elbo_best = -std::numeric_limits<double>::max();
    double elbo_prev = -std::numeric_limits<double>::max();
    double delta_elbo = std::numeric_limits<double>::max();
    double delta_elbo_ave = std::numeric_limits<double>::max();
    double delta_elbo_med = std::numeric_limits<double>::max();

    // Heuristic to estimate how far to look back in rolling window
    int cb_size
        = static_cast<int>(std::max(0.1 * max_iterations / eval_elbo_, 2.0));
    boost::circular_buffer<double> elbo_diff(cb_size);

    logger.info("Begin stochastic gradient ascent.");
    logger.info(
        "  iter"
        "             ELBO"
        "   delta_ELBO_mean"
        "   delta_ELBO_med"
        "   notes ");

    // Timing variables
    auto start = std::chrono::steady_clock::now();

    // Main loop
    bool do_more_iterations = true;
    for (int iter_counter = 1; do_more_iterations; ++iter_counter) {
      // Compute gradient using Monte Carlo integration
      calc_ELBO_grad(variational, elbo_grad, logger);

      // Update step-size
      if (iter_counter == 1) {
        history_grad_squared += elbo_grad.square();
      } else {
        history_grad_squared = pre_factor * history_grad_squared
                               + post_factor * elbo_grad.square();
      }
      eta_scaled = eta / sqrt(static_cast<double>(iter_counter));

      // Stochastic gradient update
      variational
          += eta_scaled * elbo_grad / (tau + history_grad_squared.sqrt());

      // Check for convergence every "eval_elbo_"th iteration
      if (iter_counter % eval_elbo_ == 0) {
        elbo_prev = elbo;
        elbo = calc_ELBO(variational, logger);
        if (elbo > elbo_best)
          elbo_best = elbo;
        delta_elbo = rel_difference(elbo, elbo_prev);
        elbo_diff.push_back(delta_elbo);
        delta_elbo_ave
            = std::accumulate(elbo_diff.begin(), elbo_diff.end(), 0.0)
              / static_cast<double>(elbo_diff.size());
        delta_elbo_med = circ_buff_median(elbo_diff);
        std::stringstream ss;
        ss << "  " << std::setw(4) << iter_counter << "  " << std::setw(15)
           << std::fixed << std::setprecision(3) << elbo << "  "
           << std::setw(16) << std::fixed << std::setprecision(3)
           << delta_elbo_ave << "  " << std::setw(15) << std::fixed
           << std::setprecision(3) << delta_elbo_med;
        auto end = std::chrono::steady_clock::now();
        double delta_t
            = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                  .count()
              / 1000.0;
        std::vector<double> print_vector;
        print_vector.clear();
        print_vector.push_back(iter_counter);
        print_vector.push_back(delta_t);
        print_vector.push_back(elbo);
        diagnostic_writer(print_vector);

        if (delta_elbo_ave < tol_rel_obj) {
          ss << "   MEAN ELBO CONVERGED";
          do_more_iterations = false;
        }

        if (delta_elbo_med < tol_rel_obj) {
          ss << "   MEDIAN ELBO CONVERGED";
          do_more_iterations = false;
        }

        if (iter_counter > 10 * eval_elbo_) {
          if (delta_elbo_med > 0.5 || delta_elbo_ave > 0.5) {
            ss << "   MAY BE DIVERGING... INSPECT ELBO";
          }
        }

        logger.info(ss);

        if (do_more_iterations == false
            && rel_difference(elbo, elbo_best) > 0.05) {
          logger.info(
              "Informational Message: The ELBO at a previous "
              "iteration is larger than the ELBO upon "
              "convergence!");
          logger.info(
              "This variational approximation may not "
              "have converged to a good optimum.");
        }
      }

      if (iter_counter == max_iterations) {
        logger.info(
            "Informational Message: The maximum number of "
            "iterations is reached! The algorithm may not have "
            "converged.");
        logger.info(
            "This variational approximation is not "
            "guaranteed to be meaningful.");
        do_more_iterations = false;
      }
    }
    }*/

  /**
   * Runs ADVI and writes to output.
   *
   * @param[in] eta eta parameter of stepsize sequence
   * @param[in] adapt_engaged boolean flag for eta adaptation
   * @param[in] adapt_iterations number of iterations for eta adaptation
   * @param[in] tol_rel_obj relative tolerance parameter for convergence
   * @param[in] max_iterations max number of iterations to run algorithm
   * @param[in,out] logger logger for messages
   * @param[in,out] parameter_writer writer for parameters
   *   (typically to file)
   * @param[in,out] diagnostic_writer writer for diagnostic information
   */
  int run(double eta, bool adapt_engaged, int adapt_iterations,
          int max_iterations, int eval_window, double rhat_cut,
	  double mcse_cut, double ess_cut, int num_chains,
	  callbacks::logger& logger,
          callbacks::writer& parameter_writer,
          callbacks::writer& diagnostic_writer) const {
    diagnostic_writer("iter,time_in_seconds,ELBO");

    // Initialize variational approximation
    Q variational = Q(cont_params_);

    if (adapt_engaged) {
      eta = adapt_eta(variational, adapt_iterations, logger);
      parameter_writer("Stepsize adaptation complete.");
      std::stringstream ss;
      ss << "eta = " << eta;
      parameter_writer(ss.str());
    }

    run_rvi(variational, eta,
	    max_iterations, eval_window,
	    rhat_cut, mcse_cut, ess_cut, 
	    num_chains, adapt_iterations, 
	    logger);

    //stochastic_gradient_ascent(variational, eta, tol_rel_obj, max_iterations,
    //                           logger, diagnostic_writer);

    // Write posterior mean of variational approximations.
    cont_params_ = variational.mean();
    std::vector<double> cont_vector(cont_params_.size());
    for (int i = 0; i < cont_params_.size(); ++i)
      cont_vector.at(i) = cont_params_(i);
    std::vector<int> disc_vector;
    std::vector<double> values;

    std::stringstream msg;
    model_.write_array(rng_, cont_vector, disc_vector, values, true, true,
                       &msg);
    if (msg.str().length() > 0)
      logger.info(msg);

    // The first row of lp_, log_p, and log_g.
    values.insert(values.begin(), {0, 0, 0});
    parameter_writer(values);

    // Draw more from posterior and write on subsequent lines
    logger.info("");
    std::stringstream ss;
    ss << "Drawing a sample of size " << n_posterior_samples_
       << " from the approximate posterior... ";
    logger.info(ss);
    double log_p = 0;
    double log_g = 0;
    // Draw posterior sample. log_g is the log normal densities.
    for (int n = 0; n < n_posterior_samples_; ++n) {
      variational.sample_log_g(rng_, cont_params_, log_g);
      for (int i = 0; i < cont_params_.size(); ++i) {
        cont_vector.at(i) = cont_params_(i);
      }
      std::stringstream msg2;
      model_.write_array(rng_, cont_vector, disc_vector, values, true, true,
                         &msg2);
      //  log_p: Log probability in the unconstrained space
      log_p = model_.template log_prob<false, true>(cont_params_, &msg2);
      if (msg2.str().length() > 0)
        logger.info(msg2);
      // Write lp__, log_p, and log_g.
      values.insert(values.begin(), {0, log_p, log_g});
      parameter_writer(values);
    }
    logger.info("COMPLETED.");
    return stan::services::error_codes::OK;
  }

  /**
  * RVI Diagnostics: Calculates log importance weights
  *
  * @param[in] variational_obj variational family object
  * @param[in, out] weight_vector An Eigen
  * dynamic vector of weights, sorted in descending order
  */
  void lr(const Q& variational_obj, Eigen::VectorXd& weight_vector) 
          const {
    // Need to check the vector is empty
    weight_vector.resize(n_posterior_samples_);
    double log_p, log_g;
    std::stringstream msg2;
    Eigen::VectorXd draws(variational_obj.dimension());
    // Draw posterior sample. log_g is the log normal densities.
    for (int n = 0; n < n_posterior_samples_; ++n) {
      variational_obj.sample_log_g(rng_, draws, log_g);
      //  log_p: Log probability in the unconstrained space
      log_p = model_.template log_prob<false, true>(draws, &msg2);
      weight_vector(n) = log_p - log_g;
    }
    // sort descending order
    std::sort(weight_vector.data(), weight_vector.data() + weight_vector.size(),
              std::greater<double>());
  }

  /**
   * RVI Diagnostics
   * A sample function to show how compute_potential_scale_reduction() in
   * stan/analyze/mcmc/compute_potential_scale_reduction.hpp would be used to
   * calculate Rhat values
   * 
   * @param[in] samples An Eigen::VectorXd containing individual samples
   * @return Calculated Rhat
   */
  static double calculate_sample_rhat(const Eigen::VectorXd& samples){
    std::vector<const double*> sample_ptr_arr = {samples.data()};
    std::vector<size_t> size_arr = {(size_t)samples.size()};
    return stan::analyze::compute_potential_scale_reduction(sample_ptr_arr,
                                                            size_arr);
  }
 
  /**
   * RVI Diagnostics
   * Estimate the Effective Sample Size and Monte Carlo Standard Error of posterior samples where
   * MCSE = sqrt( var(parmas) / ess)
   * @param[in] samples An Eigen::VectorXd containing posterior samples @TODO rewrite from here
   * @param[in] ess If specified, will be used as effective sample size instead
   * of calling compute_effective_sample_size()
   * 
   * @return Calculated MCSE
   */

  static double ESS_MCSE(double &ess, double &mcse,
                        const std::vector<const double*> draws,
                        const std::vector<size_t> sizes) {
    int num_chains = sizes.size();
    size_t num_draws = sizes[0];
    for (int chain = 1; chain < num_chains; ++chain) {
      num_draws = std::min(num_draws, sizes[chain]);
    }

    if (num_draws < 4) {
      ess = std::numeric_limits<double>::quiet_NaN();
      mcse = std::numeric_limits<double>::quiet_NaN();
      return std::numeric_limits<double>::quiet_NaN();
    }

    // check if chains are constant; all equal to first draw's value
    bool are_all_const = false;
    Eigen::VectorXd init_draw = Eigen::VectorXd::Zero(num_chains);

    for (int chain_idx = 0; chain_idx < num_chains; chain_idx++) {
      Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1>> draw(
              draws[chain_idx], sizes[chain_idx]);

      for (int n = 0; n < num_draws; n++) {
        if (!std::isfinite(draw(n))) {
          ess = std::numeric_limits<double>::quiet_NaN();
          mcse = std::numeric_limits<double>::quiet_NaN();
          return std::numeric_limits<double>::quiet_NaN();
        }
      }

      init_draw(chain_idx) = draw(0);

      if (draw.isApproxToConstant(draw(0))) {
        are_all_const |= true;
      }
    }

    if (are_all_const) {
      // If all chains are constant then return NaN
      // if they all equal the same constant value
      if (init_draw.isApproxToConstant(init_draw(0))) {
        ess = std::numeric_limits<double>::quiet_NaN();;
        mcse = std::numeric_limits<double>::quiet_NaN();
        return std::numeric_limits<double>::quiet_NaN();
      }
    }

    Eigen::Matrix<Eigen::VectorXd, Eigen::Dynamic, 1> acov(num_chains);
    Eigen::VectorXd chain_mean(num_chains);
    Eigen::VectorXd chain_sq_mean(num_chains);
    Eigen::VectorXd chain_var(num_chains);
    for (int chain = 0; chain < num_chains; ++chain) {
      Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1>> draw(
              draws[chain], sizes[chain]);
      stan::analyze::autocovariance<double>(draw, acov(chain));
      chain_mean(chain) = draw.mean();
      chain_sq_mean(chain) = draw.array().square().mean();
      chain_var(chain) = acov(chain)(0) * num_draws / (num_draws - 1);
    }

    double mean_var = chain_var.mean();
    double var_plus = mean_var * (num_draws - 1) / num_draws;
    if (num_chains > 1)
      var_plus += math::variance(chain_mean);
    Eigen::VectorXd rho_hat_s(num_draws);
    rho_hat_s.setZero();
    Eigen::VectorXd acov_s(num_chains);
    for (int chain = 0; chain < num_chains; ++chain)
      acov_s(chain) = acov(chain)(1);
    double rho_hat_even = 1.0;
    rho_hat_s(0) = rho_hat_even;
    double rho_hat_odd = 1 - (mean_var - acov_s.mean()) / var_plus;
    rho_hat_s(1) = rho_hat_odd;

    // Convert raw autocovariance estimators into Geyer's initial
    // positive sequence. Loop only until num_draws - 4 to
    // leave the last pair of autocorrelations as a bias term that
    // reduces variance in the case of antithetical chains.
    size_t s = 1;
    while (s < (num_draws - 4) && (rho_hat_even + rho_hat_odd) > 0) {
      for (int chain = 0; chain < num_chains; ++chain)
        acov_s(chain) = acov(chain)(s + 1);
      rho_hat_even = 1 - (mean_var - acov_s.mean()) / var_plus;
      for (int chain = 0; chain < num_chains; ++chain)
        acov_s(chain) = acov(chain)(s + 2);
      rho_hat_odd = 1 - (mean_var - acov_s.mean()) / var_plus;
      if ((rho_hat_even + rho_hat_odd) >= 0) {
        rho_hat_s(s + 1) = rho_hat_even;
        rho_hat_s(s + 2) = rho_hat_odd;
      }
      s += 2;
    }

    int max_s = s;
    // this is used in the improved estimate, which reduces variance
    // in antithetic case -- see tau_hat below
    if (rho_hat_even > 0)
      rho_hat_s(max_s + 1) = rho_hat_even;

    // Convert Geyer's initial positive sequence into an initial
    // monotone sequence
    for (int s = 1; s <= max_s - 3; s += 2) {
      if (rho_hat_s(s + 1) + rho_hat_s(s + 2) > rho_hat_s(s - 1) + rho_hat_s(s)) {
        rho_hat_s(s + 1) = (rho_hat_s(s - 1) + rho_hat_s(s)) / 2;
        rho_hat_s(s + 2) = rho_hat_s(s + 1);
      }
    }

    double num_total_draws = num_chains * num_draws;
    // Geyer's truncated estimator for the asymptotic variance
    // Improved estimate reduces variance in antithetic case
    double tau_hat = -1 + 2 * rho_hat_s.head(max_s).sum() + rho_hat_s(max_s + 1);
    double ess_val = num_total_draws / tau_hat;
    ess = ess_val;
    mcse = std::sqrt((chain_sq_mean.mean() - chain_mean.mean() * chain_mean.mean())/ess_val);
    return 0;
  }

  /**
   * Run Robust Variational Inference.
   * A. Dhaka et al., 2020
   * 
   * @param[in] max_runs Max number of VI iterations
   * @param[in] rhat_cut Threshold for Rhat diagnostics
   * @param[in] mcse_cut Threshold for MCSE diagnostics
   * @param[in] ess_cut Threshold for effective sample size diagnostics
   * @param[in] num_chains Number of VI chains to run simultaneously
   * @param[in] logger logger
   */
  void run_rvi(Q& variational, const double eta,
	       const int max_runs, const int eval_window, const double rhat_cut, 
               const double mcse_cut, const double ess_cut, 
               const int num_chains, const int adapt_iterations, 
               callbacks::logger& logger) const {

    std::stringstream ss;

    double khat, ess, mcse, max_rhat, rhat, eta_scaled;
    int T0 = max_runs - 1;

    const int dim = variational.dimension();
    const int n_approx_params = variational.num_approx_params();


    std::vector<Q> variational_obj_vec;
    std::vector<Q> elbo_grad_vec;
    std::vector<Q> elbo_grad_square_vec;

    // for each chain, save variational parameter values on matrix
    // of dim (n_params, n_iters)
    typedef Eigen::Matrix<double, Eigen::Dynamic, 
                          Eigen::Dynamic, Eigen::RowMajor> histMat;
    std::vector<histMat> hist_vector;
    hist_vector.reserve(num_chains);
    variational_obj_vec.reserve(num_chains);
    elbo_grad_vec.reserve(num_chains);
    elbo_grad_square_vec.reserve(num_chains);

    for(int i = 0; i < num_chains; i++){
      hist_vector.push_back(histMat(n_approx_params, max_runs));
      variational_obj_vec.push_back(Q(cont_params_));
      elbo_grad_vec.push_back(Q(dim));
      elbo_grad_square_vec.push_back(Q(dim));
    }

    for (int n_iter = 0; n_iter < max_runs; n_iter++){
      eta_scaled = eta / sqrt(static_cast<double>(n_iter + 1));
      for (int n_chain = 0; n_chain < num_chains; n_chain++){
        calc_ELBO_grad(variational_obj_vec[n_chain], elbo_grad_vec[n_chain], logger);
        if (n_iter == 0) {
          elbo_grad_square_vec[n_chain] += elbo_grad_vec[n_chain].square();
        }
        else {
          elbo_grad_square_vec[n_chain] = 0.9 * elbo_grad_square_vec[n_chain]
                                          + 0.1 * elbo_grad_vec[n_chain].square();
        }
        variational_obj_vec[n_chain] += eta_scaled * elbo_grad_vec[n_chain] / (1.0 + elbo_grad_square_vec[n_chain].sqrt());

        hist_vector[n_chain].col(n_iter) = variational_obj_vec[n_chain].return_approx_params();
      }

      if ((n_iter % eval_window == 0 && n_iter > 0) || n_iter == max_runs - 1){
        max_rhat = std::numeric_limits<double>::lowest();
        for(int k = 0; k < n_approx_params; k++) {
          std::vector<const double*> hist_ptrs;
          std::vector<size_t> chain_length;
          if(num_chains == 1){
            // use split rhat
            chain_length.assign(2, static_cast<size_t>(n_iter/2));
            hist_ptrs.push_back(hist_vector[0].row(k).data());
            hist_ptrs.push_back(hist_ptrs[0] + chain_length[0]);
          }
          else{
            for(int i = 0; i < num_chains; i++){
              chain_length.push_back(static_cast<size_t>(n_iter));
              hist_ptrs.push_back(hist_vector[i].row(k).data());
            }
          }
          rhat = stan::analyze::compute_potential_scale_reduction(hist_ptrs, chain_length);
          //rhat = 2.0;
          max_rhat = std::max<double>(max_rhat, rhat);
        }

        if (max_rhat < rhat_cut) {
          T0 = n_iter;
          ss << "Preliminary iterations terminated by rhat condition at iteration # " << T0 <<
          " with max reported Rhat value of " << max_rhat << "\n";
          break;
        }
      }
    }

    bool khat_failed = false;
    for(int k = 0; k < num_chains; k++){
      Eigen::VectorXd lw_vec(n_posterior_samples_);
      lr(variational_obj_vec[k], lw_vec);
      double sigma, max_lw;
      int n_tail;
      if(n_posterior_samples_ < 255) {
        n_tail = int(lw_vec.size() * 0.2);
      }
      else{
        n_tail = 3 * sqrt(lw_vec.size()); // if more than 255 samples 3 * sqrt(lw_vec.size())
      }
      max_lw = lw_vec.maxCoeff();
      lw_vec = lw_vec.array() - max_lw;
      lw_vec = lw_vec.array().exp() - std::exp(lw_vec(n_tail));
      lw_vec = lw_vec.head(n_tail);
      stan::analyze::gpdfit(lw_vec, khat, sigma);

      ss << "Chain " << k << " khat: " << khat << "\n";
      if(khat > 1.0) { 
        khat_failed = true;
        break;
      }
    }
    if (khat_failed || max_rhat > rhat_cut) {
      logger.warn("Optimization may have not converged");
      ss << " max_rhat: " << max_rhat << " rhat_cut: " << rhat_cut << "\n";
      // avarage parameters from the last eval_window param iterations
      for(int i = 0; i < num_chains; i++){
        variational_obj_vec[i].set_approx_params(
          hist_vector[i].block(0, T0 - eval_window + 1, n_approx_params, eval_window).rowwise().mean());
      }
    }
    else {
      ss << "Start secondary iteration at step #: " << T0 << "\n";
      for(int n_post_iter = T0; n_post_iter < max_runs; n_post_iter++){
        eta_scaled = eta / sqrt(static_cast<double>(n_post_iter + 1));
        for (int n_chain = 0; n_chain < num_chains; n_chain++){
          calc_ELBO_grad(variational_obj_vec[n_chain], elbo_grad_vec[n_chain], logger);
          elbo_grad_square_vec[n_chain] = 0.9 * elbo_grad_square_vec[n_chain]
                                          + 0.1 * elbo_grad_vec[n_chain].square();

          variational_obj_vec[n_chain] += eta_scaled * elbo_grad_vec[n_chain] / 
                                          elbo_grad_square_vec[n_chain].sqrt();

          hist_vector[n_chain].col(n_post_iter) = variational_obj_vec[n_chain].return_approx_params();
        }
        if ((n_post_iter - T0) % eval_window == 0 && (n_post_iter - T0) > 0) {
          ss << "Secondary iteration # " << n_post_iter << "\n";
          double min_ess = std::numeric_limits<double>::infinity(), max_mcse = std::numeric_limits<double>::lowest();
          for(int k = 0; k < n_approx_params; k++){
            std::vector<const double*> hist_ptrs;
            std::vector<size_t> chain_length;
            if(num_chains == 1){
              // write split rhat again here
              chain_length.assign(2, static_cast<size_t>((n_post_iter - T0 + 1) / 2));
              hist_ptrs.push_back(hist_vector[0].row(k).data() + T0);
              hist_ptrs.push_back(hist_ptrs[0] + chain_length[0]); 
            }
            else {
              for(int i = 0; i < num_chains; i++){
                chain_length.push_back(static_cast<size_t>(n_post_iter - T0 + 1));
                hist_ptrs.push_back(hist_vector[i].row(k).data() + T0);
              }
            }
            double ess, mcse;
            ESS_MCSE(ess, mcse, hist_ptrs, chain_length);
            ss << "ESS: " << ess << " MCSE: " << mcse << "\n";
            min_ess = std::min<double>(min_ess, ess);
            max_mcse = std::max<double>(max_mcse, mcse);
          }
          if(max_mcse < mcse_cut && min_ess > ess_cut){
            ss << "Secondary iteration break condition reached at iteration # "
                       << n_post_iter << "\n";
            ss << "min ESS: " << min_ess << " max MCSE: " << max_mcse << "\n";
            for(int i = 0; i < num_chains; i++){
                variational_obj_vec[i].set_approx_params(
                hist_vector[i].block(0, T0, n_approx_params, n_post_iter - T0 + 1).rowwise().mean());
            }
            break;
          }
        }
      }
    }
    variational.set_to_zero();
    for(int i = 0; i < num_chains; i++){
      variational += 1.0 / num_chains * variational_obj_vec[i];
    }
    ss << "Finished optimization" << "\n";
    cont_params_ = variational.mean();
    std::vector<double> cont_vector(cont_params_.size());
    for (int i = 0; i < cont_params_.size(); ++i)
      cont_vector.at(i) = cont_params_(i);
    std::vector<int> disc_vector;
    std::vector<double> values;

    std::stringstream msg;
    model_.write_array(rng_, cont_vector, disc_vector, values, true, true,
                       &msg);
    if (msg.str().length() > 0)
    for(int i = 0; i < num_chains; i++){
      ss << "Chain " << i << "mean :" << variational_obj_vec[i].mean();
    }
    ss << "----\nQ variational:\n" << variational.mean() << "\n----\n";
    ss << "Num of Model params: " << dim << "\n";
    ss << "Num of Approx params: " << n_approx_params << "\n";
    logger.info(ss);
  }

  /**
   * Compute the median of a circular buffer.
   *
   * @param[in] cb circular buffer with some number of values in it.
   * @return median of values in circular buffer.
   */
  double circ_buff_median(const boost::circular_buffer<double>& cb) const {
    // FIXME: naive implementation; creates a copy as a vector
    std::vector<double> v;
    for (boost::circular_buffer<double>::const_iterator i = cb.begin();
         i != cb.end(); ++i) {
      v.push_back(*i);
    }

    size_t n = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + n, v.end());
    return v[n];
  }

  /**
   * Compute the relative difference between two double values.
   *
   * @param[in] prev previous value
   * @param[in] curr current value
   * @return  absolutely value of relative difference
   */
  double rel_difference(double prev, double curr) const {
    return std::fabs((curr - prev) / prev);
  }

 protected:
  Model& model_;
  Eigen::VectorXd& cont_params_;
  BaseRNG& rng_;
  int n_monte_carlo_grad_;
  int n_monte_carlo_elbo_;
  int n_posterior_samples_;
};
}  // namespace variational
}  // namespace stan
#endif
