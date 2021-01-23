#include <test/test-models/good/variational/std_normal.hpp>
#include <stan/variational/advi.hpp>
#include <stan/callbacks/stream_logger.hpp>
#include <gtest/gtest.h>
#include <test/unit/util.hpp>
#include <vector>
#include <string>
#include <boost/random/additive_combine.hpp>  // L'Ecuyer RNG

typedef boost::ecuyer1988 rng_t;
typedef std_normal_model_namespace::std_normal_model Model;

TEST(rvi_test, std_normal_meanfield) {
  // Create mock data_var_context
  static const std::string DATA = "";
  std::stringstream data_stream(DATA);
  stan::io::dump dummy_context(data_stream);

  // Instantiate model
  Model my_model(dummy_context);

  // RNG
  rng_t base_rng(0);

  // Other params
  stan::callbacks::stream_logger logger(std::cout, std::cout, std::cout,
                                        std::cout, std::cout);
  stan::callbacks::stream_writer stdout_writer(std::cout);

  // Dummy input
  Eigen::VectorXd cont_params = Eigen::VectorXd::Zero(1);
  cont_params(0) = -0.75;

  // ADVI
  int n_monte_carlo_grad = 10;
  int n_monte_carlo_elbo = 100;
  int eval_elbo = 10;
  int n_posterior_samples = 100;
  stan::variational::advi<Model, stan::variational::normal_meanfield, rng_t>
    test_advi(my_model, cont_params, base_rng, n_monte_carlo_grad,
	      n_monte_carlo_elbo, n_posterior_samples);
  
  test_advi.run(1.0, true, 250, 1000,
		50, 1.1, 20, 10, 4, logger,
		stdout_writer, stdout_writer); 
}
