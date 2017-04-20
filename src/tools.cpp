#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using std::cout;

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
    VectorXd rmse(4);
    rmse << 0,0,0,0;

    if (estimations.size() == 0) {
        cout << "the estimation vector size should not be zero\n";
        return rmse;
    }
    if (estimations.size() != ground_truth.size()) {
        cout << "the estimation vector size should equal ground truth vector size\n";
        return rmse;
    }

    //accumulate squared residuals
    for(size_t i = 0; i < estimations.size(); ++i){
        VectorXd c = (estimations[i] - ground_truth[i]).array().square();
        rmse = rmse + c;
    }
    rmse = rmse / estimations.size();
    rmse = rmse.array().sqrt();

    return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
    MatrixXd Hj(3,4);
    //recover state parameters
    auto px = x_state(0);
    auto py = x_state(1);
    auto vx = x_state(2);
    auto vy = x_state(3);

    //pre-compute a set of terms to avoid repeated calculation
    auto c1 = px*px + py*py;
    auto c2 = sqrt(c1);
    auto c3 = (c1*c2);

    //check division by zero
    if(fabs(c1) < 0.0001){
        cout << "CalculateJacobian () - Error - Division by Zero\n";
        return Hj;
    }

    //compute the Jacobian matrix
    Hj << (px/c2), (py/c2), 0, 0,
            -(py/c1), (px/c1), 0, 0,
            py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

    return Hj;
}
