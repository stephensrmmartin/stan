#include <stan/variational/advi.hpp>
#include <stan/callbacks/stream_logger.hpp>
#include <gtest/gtest.h>
#include <test/unit/util.hpp>
#include <string>
#include <test/test-models/good/model/valid.hpp>
#include <boost/random/additive_combine.hpp>  // L'Ecuyer RNG

typedef boost::ecuyer1988 rng_t;

TEST(RVI_SUBMODULE_TEST, gpdfit){
    // check values are equal to loo's gpdfit()
    double advi_k, advi_sigma;
    Eigen::VectorXd x(6);
    x << 1, 2, 3, 4, 5, 6;
    double loo_wip_k = 0.0275548, loo_wip_sigma = 5.616346; // wip = true
    double loo_nowip_k = -0.7598539, loo_nowip_sigma = 5.616346; // wip = false

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

    stan::variational::advi<stan_model, stan::variational::normal_meanfield, rng_t>
        test_advi(model, cont_params, base_rng, 10,
                    10, 100, 1);

    test_advi.gpdfit(x, advi_k, advi_sigma, true);
    EXPECT_FLOAT_EQ(loo_wip_k, advi_k);
    EXPECT_FLOAT_EQ(loo_wip_sigma, advi_sigma);

    test_advi.gpdfit(x, advi_k, advi_sigma, false);
    EXPECT_FLOAT_EQ(loo_nowip_k, advi_k);
    EXPECT_FLOAT_EQ(loo_nowip_sigma, advi_sigma);

    std::cout << test_advi.calculate_sample_standard_error(x) << std::endl;
}