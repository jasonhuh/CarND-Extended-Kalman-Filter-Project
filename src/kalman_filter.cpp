#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

// Predict the state
void KalmanFilter::Predict() {
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd y = z - H_ * x_;
  UpdateCommon(y);
}

// Update the state by using Extended Kalman Filter equations
void KalmanFilter::UpdateEKF(const VectorXd &z) {
  auto ro = sqrt(pow(x_[0], 2) + pow(x_[1], 2));
  auto phi = 0.0, ro_dot = 0.0;
  if (fabs(x_[0]) > 0.001) {
    phi = atan2(x_[1], x_[0]);
  }
  if (fabs(ro) > 0.001) {
    ro_dot = (x_[0] * x_[2] + x_[1] * x_[3]) / ro;
  }

  VectorXd hx(3);
  hx << ro, phi, ro_dot;

  // prediction error
  VectorXd y = z - hx;
  UpdateCommon(y);
}

// Update the state
void KalmanFilter::UpdateCommon(const VectorXd& y) {

  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si; // Kalman gain

  //new estimate
  x_ = x_ + (K * y);
  auto x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}
