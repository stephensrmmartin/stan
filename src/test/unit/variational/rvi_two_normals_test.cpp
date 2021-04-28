#include <test/test-models/good/variational/two_normals.hpp>
#include <stan/variational/advi.hpp>
#include <stan/callbacks/stream_logger.hpp>
#include <gtest/gtest.h>
#include <test/unit/util.hpp>
#include <vector>
#include <string>
#include <boost/random/additive_combine.hpp>  // L'Ecuyer RNG
#include <stan/analyze/mcmc/compute_potential_scale_reduction.hpp>

typedef boost::ecuyer1988 rng_t;
typedef two_normals_model_namespace::two_normals_model Model;

TEST(rvi_test, two_normals_meanfield) {
  // Create mock data_var_context
  static const std::string DATA = "";
  std::stringstream data_stream(DATA);
  stan::io::dump dummy_context(data_stream);

  // Instantiate model
  Model my_model(dummy_context);

  // RNG
  rng_t base_rng(1337);

  // Other params
  stan::callbacks::stream_logger logger(std::cout, std::cout, std::cout,
                                        std::cout, std::cout);
  stan::callbacks::stream_writer stdout_writer(std::cout);

  // Dummy input
  Eigen::VectorXd cont_params = Eigen::VectorXd::Zero(2);
  cont_params(0) = 0.0;
  cont_params(1) = 0.0;

  // ADVI
  int n_monte_carlo_grad = 10;
  int n_monte_carlo_elbo = 100;
  int eval_elbo = 10;
  int n_posterior_samples = 100;
  
  stan::variational::advi<Model, stan::variational::normal_meanfield, rng_t>
    test_advi(my_model, cont_params, base_rng, n_monte_carlo_grad,
	      n_monte_carlo_elbo, n_posterior_samples);

  int eval_window = 100;
  double window_size = 0.5;
  test_advi.run(1.0, true, 250, 10000,
  eval_window, window_size, 1.1, 0.02, 20, 4, logger,
	stdout_writer, stdout_writer);
}
