#include "FusionEKF.h"
#include "Eigen/Dense"
#include <iostream>
#include <memory>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;
  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
        0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;

  //measurement matrix - laser
  H_laser_ << 1, 0, 0, 0,
          0, 1, 0, 0;

  //initialize Jacobian matrix
  Hj_ << 0, 0, 0, 0,
          0, 0, 0, 0,
          0, 0, 0, 0;

  noise_ax_ = 9.0;
  noise_ay_ = 9.0;

  InitializeKalmanFilter();
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      auto ro = measurement_pack.raw_measurements_[0];
      auto phi = measurement_pack.raw_measurements_[1];
      auto ro_dot = measurement_pack.raw_measurements_[2];
      ekf_->x_ << ro * cos(phi), ro * sin(phi), ro_dot * cos(phi), ro_dot * sin(phi);
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      ekf_->x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0.0, 0.0;
    }
    previous_timestamp_ = measurement_pack.timestamp_;
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  //compute the time elapsed between the current and previous measurements
  auto dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
  previous_timestamp_ = measurement_pack.timestamp_;

  auto dt_2 = dt * dt;
  auto dt_3 = dt_2 * dt;
  auto dt_4 = dt_3 * dt;

  //Modify the F matrix so that the time is integrated
  ekf_->F_(0, 2) = dt;
  ekf_->F_(1, 3) = dt;

  //set the process covariance matrix Q
  ekf_->Q_ <<  dt_4/4*noise_ax_, 0, dt_3/2*noise_ax_, 0,
          0, dt_4/4*noise_ay_, 0, dt_3/2*noise_ay_,
          dt_3/2*noise_ax_, 0, dt_2*noise_ax_, 0,
          0, dt_3/2*noise_ay_, 0, dt_2*noise_ay_;

  ekf_->Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) { // Radar updates
    ekf_->H_ = Tools::CalculateJacobian(ekf_->x_);
    ekf_->R_ = R_radar_;
    ekf_->UpdateEKF(measurement_pack.raw_measurements_);
  } else { // Laser updates
    ekf_->H_ = H_laser_;
    ekf_->R_ = R_laser_;
    ekf_->Update(measurement_pack.raw_measurements_);
  }
}

void FusionEKF::InitializeKalmanFilter() {
  // x state
  VectorXd x = VectorXd(4);
  x << 1, 1, 1, 1;

  // state covariance matrix P
  MatrixXd P = MatrixXd(4, 4);
  P << 1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1000, 0,
          0, 0, 0, 1000;

  // the initial transition matrix F_
  MatrixXd F = MatrixXd(4, 4);
  F << 1, 0, 1, 0,
          0, 1, 0, 1,
          0, 0, 1, 0,
          0, 0, 0, 1;

  // measurement matrix
  MatrixXd H = MatrixXd(2, 4);
  H << 1, 0, 0, 0,
          0, 1, 0, 0;

  // measurement covariance
  MatrixXd R = MatrixXd(2, 2);
  R << 0.0225, 0,
          0, 0.0225;

  MatrixXd Q = MatrixXd(4, 4);
  Q << 0, 0, 0, 0,
          0, 0, 0, 0,
          0, 0, 0, 0,
          0, 0, 0, 0;

  ekf_ = unique_ptr<KalmanFilter>(new KalmanFilter());
  ekf_->Init(x, P, F, H, R, Q);
}