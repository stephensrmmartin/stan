#ifndef STAN_ANALYZE_ESTIMATE_GPD_PARAMS_HPP
#define STAN_ANALYZE_ESTIMATE_GPD_PARAMS_HPP

#include <stan/math.hpp>
#include <stan/analyze/mcmc/autocovariance.hpp>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>


namespace stan{
namespace analyze{
  /**
   * RVI Diagnostics: Approximate parameters k and sigma for pareto-k diagnostics
   * from a given sample vector x
   * Zhang, Stephens (2009)
   * 
   * @param[in] x Vector of values from which parameters will be estimated
   * @param[in, out] k_ret Variable to return the estimated k value.
   * @param[in, out] sigma Variable to return the estimated sigma value.
   * @param[in] wip Boolean indicating whether to use a weakly informed prior
   * @param[in] min_grid_pts The minimum number of grid points used in the fit
   * algorithm. The actual count used is `min_grid_pts + floor(sqrt(length(x)))`
   * @param[in] sort_x If `true` (the default), the first step in the fitting
   * algorithm is to sort the elements of `x`. If `x` is already sorted in
   * ascending order then `sort_x` can be set to `false` to skip the initial
   * sorting step.
   */
  void gpdfit(Eigen::VectorXd& x, double& k, double& sigma, 
              const bool wip = true, const int min_grid_pts = 30, 
              bool sort_x = true){
    //See section 4 of Zhang and Stephens (2009) #TODO translate r to c

    // R function definitions
    auto lx = [&](const Eigen::VectorXd a, const Eigen::VectorXd x,
            Eigen::VectorXd& ret_mat) mutable {
      Eigen::VectorXd tmp_mat(x.size());
      ret_mat.resize(a.size());
      for(int i = 0; i < a.size(); i++){
        tmp_mat = x * -a(i);
        ret_mat(i) = tmp_mat.array().log1p().mean();
        ret_mat(i) = std::log(-a(i) / ret_mat(i)) - ret_mat(i) - 1;
      }
      ret_mat = ret_mat.matrix();
    };

    auto adjust_k_wip = [&](double k_, double n_) mutable {
      int a = 10, n_plus_a = n_ + a;
      return k_ * n_ / n_plus_a + a * 0.5 / n_plus_a;
    };
    // end function definitions

    if (sort_x) {
      std::sort(x.data(), x.data() + x.size());
    }

    const int N = x.size(), prior = 3, M = min_grid_pts + 
                                           std::floor(std::sqrt(N));
    Eigen::VectorXd jj(M);
    for(int i = 0; i < M; i++){
      jj(i) = i + 1;  // seq_len(M) (cpp indexing)
    }

    Eigen::VectorXd theta(M);
    double xstar = x(static_cast<int>(std::floor(N / 4 + 0.5)));

    theta = 1 / x(N-1) + (1 - (M / (jj.array() - 0.5)).sqrt()) / prior / xstar;
    theta = theta.matrix();

    Eigen::VectorXd l_theta(theta.size());
    lx(theta, x, l_theta);
    l_theta *= N;
    
    Eigen::VectorXd w_theta = Eigen::VectorXd::Ones(l_theta.size());
    for(int i = 0; i < w_theta.size(); i++){
      w_theta(i) = 1 / (l_theta.array() - l_theta(jj(i) - 1)).exp().sum();
      // subtract 1 from jj(i) to convert to cpp index
    }
    double theta_hat = (theta.array() * w_theta.array()).sum();
    k = (-theta_hat * x.array()).log1p().mean();
    sigma = -k / theta_hat;

    if(wip){
      k = adjust_k_wip(k, N);
    }

    if(std::isnan(k)){
      k = std::numeric_limits<double>::infinity();
    }
  }
}
}

#endif
