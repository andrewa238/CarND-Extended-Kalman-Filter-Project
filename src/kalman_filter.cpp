#include <iostream>
#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
    x_ = x_in;
    P_ = P_in;
    F_ = F_in;
    H_ = H_in;
    R_ = R_in;
    Q_ = Q_in;
}

void KalmanFilter::Predict() {
  //predict the state
  x_ = F_*x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_*P_*Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  //update the state by using Kalman Filter equations
  VectorXd y = z - H_*x_;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_*P_*Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = P_*Ht*Si;
  MatrixXd I = MatrixXd::Identity(4,4);
    
  x_ = x_ + (K*y);
  P_ = (I - K*H_)*P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {

  //update the state by using Extended Kalman Filter equations

  double px = x_(0);
  double py = x_(1);
  double vx = x_(2);
  double vy = x_(3);
    
  double rho = sqrt(px*px + py*py);
  if(abs(rho) < 0.0001){
      rho = 0.0001;
  }
  double phi = atan2(py,px);
  double rhodot = (px*vx + py*vy)/rho;
    
  VectorXd z_pred = VectorXd(3);
  z_pred << rho, phi, rhodot;
    
  VectorXd y = z - z_pred;
  // make sure the prediction phi is in range -PI ~ PI
  y[1] = atan2(sin(y[1]),cos(y[1]));
    
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_*P_*Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = P_*Ht*Si;
  MatrixXd I = MatrixXd::Identity(4,4);
    
  x_ = x_ + (K*y);
  P_ = (I - K*H_)*P_;
}
