#include <stan/variational/advi.hpp>
#include <stan/variational/families/normal_meanfield.hpp>
#include <stan/callbacks/stream_logger.hpp>
#include <gtest/gtest.h>
#include <test/unit/util.hpp>
#include <string>
#include <test/test-models/good/model/valid.hpp>
#include <boost/random/additive_combine.hpp>  // L'Ecuyer RNG
#include <stan/mcmc/chains.hpp>
#include <stan/io/stan_csv_reader.hpp>
#include <gtest/gtest.h>
#include <fstream>
#include <sstream>

typedef boost::ecuyer1988 rng_t;

class RVI_SUBMODULE_TEST : public testing::Test {
public:
    void SetUp() {
      blocker1_stream.open("src/test/unit/mcmc/test_csv_files/blocker.1.csv");
      blocker2_stream.open("src/test/unit/mcmc/test_csv_files/blocker.2.csv");
    }

    void TearDown() {
      blocker1_stream.close();
      blocker2_stream.close();
    }
    std::ifstream blocker1_stream, blocker2_stream;
};

TEST_F(RVI_SUBMODULE_TEST, ess_mcse) {
/*std::stringstream out;
stan::io::stan_csv blocker1
        = stan::io::stan_csv_reader::parse(blocker1_stream, &out);
stan::io::stan_csv blocker2
        = stan::io::stan_csv_reader::parse(blocker2_stream, &out);
EXPECT_EQ("", out.str());

stan::mcmc::chains<> chains(blocker1);
chains.add(blocker2);
Eigen::Matrix<Eigen::VectorXd, Eigen::Dynamic, 1> samples(
        chains.num_chains());
std::vector<const double*> draws(chains.num_chains());
std::vector<size_t> sizes(chains.num_chains());

for (int index = 4; index < chains.num_params(); index++) {
  for (int chain = 0; chain < chains.num_chains(); ++chain) {
    samples(chain) = chains.samples(chain, index);
    draws[chain] = &samples(chain)(0);
    sizes[chain] = samples(chain).size();
  }
}

Eigen::VectorXd x(6);
x << 1, 2, 3, 4, 5, 6;
std::fstream data_stream(std::string("").c_str(), std::fstream::in);
stan::io::dump data_var_context(data_stream);
data_stream.close();

std::stringstream output;
stan_model model(data_var_context, 0, static_cast<std::stringstream*>(0));

Eigen::VectorXd cont_params = Eigen::VectorXd::Zero(2);
cont_params(0) = 0.75;
cont_params(1) = 0.75;
rng_t base_rng;
base_rng.seed(3021828106u);

double ess, mcse;
int window = 5;
stan::variational::advi<stan_model, stan::variational::normal_meanfield, rng_t>
  test_advi(model, cont_params, base_rng, 10, 10, 100, 1);
  int num_chains = draws.size();
  std::vector<size_t> windows(num_chains, window);
  test_advi.ESS_MCSE(ess, mcse, draws, windows); // J chains of W parameters stored
*/

  stan::variational::normal_meanfield mf(4);
  Eigen::VectorXd params(8);
  params << 1, 2, 3, 4, 5, 6, 7, 8;
  mf.set_approx_params(params);
  std::cout << mf.return_approx_params() << "\n";
  Eigen::VectorXd rhs(8);
  rhs.setOnes(8);
  mf += 1;
  std::cout << mf.mu() << "\n";
}
