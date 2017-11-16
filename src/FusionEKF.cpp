#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

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
  
  H_laser_ << 1, 0, 0, 0,
  0, 1, 0, 0;
  
  ekf_.F_ = MatrixXd::Zero(4,4);
  ekf_.P_ = MatrixXd::Zero(4,4);
  
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
    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 0, 10;
    
    ekf_.F_ = MatrixXd(4,4);
    ekf_.F_ << 1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1;
    ekf_.Q_ = MatrixXd::Zero(4,4);
    ekf_.P_ << 1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 10, 0,
    0, 0, 0, 10;
    
    previous_timestamp_ = measurement_pack.timestamp_;
    
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
       Convert radar from polar to cartesian coordinates and initialize state.
       */
      double rho = measurement_pack.raw_measurements_[0];
      double phi = measurement_pack.raw_measurements_[1];
      double rhodot = measurement_pack.raw_measurements_[2];
      ekf_.x_[0] = rho*cos(phi);
      ekf_.x_[1] = rho*sin(phi);
      ekf_.x_[2] = 0;
      ekf_.x_[3] = 0;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
       Initialize state.
       */
      double px = measurement_pack.raw_measurements_[0];
      double py = measurement_pack.raw_measurements_[1];
      ekf_.x_[0] = px;
      ekf_.x_[1] = py;
      ekf_.x_[2] = 0;
      ekf_.x_[3] = 0;
    }
    
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }
  
  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
  
  double dt = (measurement_pack.timestamp_ - previous_timestamp_)/1000000.0; // delta_t in seconds
  previous_timestamp_ = measurement_pack.timestamp_;
  
  double dt_2 = dt*dt;
  double dt_3 = dt_2*dt;
  double dt_4 = dt_3*dt;
  double noise_ax = 9;
  double noise_ay = 9;
  
  ekf_.F_(0,2) = dt;
  ekf_.F_(1,3) = dt;
  
  ekf_.Q_ << dt_4*noise_ax/4, 0, dt_3*noise_ax/2, 0,
  0, dt_4*noise_ay/4, 0, dt_3*noise_ay/2,
  dt_3*noise_ax/2, 0, dt_2*noise_ax, 0,
  0, dt_3*noise_ay/2, 0, dt_2*noise_ax;
  
  ekf_.Predict();
  
  /*****************************************************************************
   *  Update
   ****************************************************************************/
  
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  }
  else {
    // Laser updates
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }
  
  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
