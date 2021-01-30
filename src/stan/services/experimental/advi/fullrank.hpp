#ifndef STAN_SERVICES_EXPERIMENTAL_ADVI_FULLRANK_HPP
#define STAN_SERVICES_EXPERIMENTAL_ADVI_FULLRANK_HPP

#include <stan/callbacks/interrupt.hpp>
#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/services/util/experimental_message.hpp>
#include <stan/services/util/initialize.hpp>
#include <stan/services/util/create_rng.hpp>
#include <stan/io/var_context.hpp>
#include <stan/variational/advi.hpp>
#include <boost/random/additive_combine.hpp>
#include <string>
#include <vector>

namespace stan {
namespace services {
namespace experimental {
namespace advi {

/**
 * Runs full rank ADVI.
 *
 * @tparam Model A model implementation
 * @param[in] model Input model to test (with data already instantiated)
 * @param[in] init var context for initialization
 * @param[in] random_seed random seed for the random number generator
 * @param[in] chain chain id to advance the random number generator
 * @param[in] init_radius radius to initialize
 * @param[in] grad_samples number of samples for Monte Carlo estimate
 *   of gradients
 * @param[in] elbo_samples number of samples for Monte Carlo estimate
 *   of ELBO
 * @param[in] max_iterations maximum number of iterations
 * @param[in] eta stepsize scaling parameter for variational inference
 * @param[in] eval_window Interval to calculate termination conditions
 * @param[in] window_size Proportion of eval_window samples to calculate
 *   Rhat for termination condition
 * @param[in] rhat_cut Rhat termination criteria
 * @param[in] mcse_cut MCSE termination criteria
 * @param[in] ess_cut effective sample size termination criteria
 * @param[in] num_chains Number of VI chains to run
 * @param[in] adapt_engaged adaptation engaged?
 * @param[in] adapt_iterations number of iterations for eta adaptation
 * @param[in] output_samples number of posterior samples to draw and
 *   save
 * @param[in,out] interrupt callback to be called every iteration
 * @param[in,out] logger Logger for messages
 * @param[in,out] init_writer Writer callback for unconstrained inits
 * @param[in,out] parameter_writer output for parameter values
 * @param[in,out] diagnostic_writer output for diagnostic values
 * @return error_codes::OK if successful
 */
template <class Model>
int fullrank(Model& model, const stan::io::var_context& init,
              unsigned int random_seed, unsigned int chain, double init_radius,
              int grad_samples, int elbo_samples, int max_iterations,
              double eta, int eval_window, double window_size, double rhat_cut, 
              double mcse_cut, double ess_cut, int num_chains, 
              bool adapt_engaged, int adapt_iterations, int output_samples,
              callbacks::interrupt& interrupt, callbacks::logger& logger,
              callbacks::writer& init_writer,
              callbacks::writer& parameter_writer,
              callbacks::writer& diagnostic_writer) {
  util::experimental_message(logger);

  boost::ecuyer1988 rng = util::create_rng(random_seed, chain);

  std::vector<int> disc_vector;
  std::vector<double> cont_vector = util::initialize(
      model, init, rng, init_radius, true, logger, init_writer);

  std::vector<std::string> names;
  names.push_back("lp__");
  names.push_back("log_p__");
  names.push_back("log_g__");
  model.constrained_param_names(names, true, true);
  parameter_writer(names);

  Eigen::VectorXd cont_params
      = Eigen::Map<Eigen::VectorXd>(&cont_vector[0], cont_vector.size(), 1);

  stan::variational::advi<Model, stan::variational::normal_fullrank,
                          boost::ecuyer1988>
      cmd_advi(model, cont_params, rng, grad_samples, elbo_samples,
               output_samples);
  cmd_advi.run(eta, adapt_engaged, adapt_iterations,
	       max_iterations, eval_window, window_size, rhat_cut, mcse_cut,
	       ess_cut, num_chains, logger, parameter_writer, diagnostic_writer);

  return 0;
}
}  // namespace advi
}  // namespace experimental
}  // namespace services
}  // namespace stan
#endif
