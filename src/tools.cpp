#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;
  
  // Check for division by zero
  if(estimations.size() == 0){
    cout << "Error: no estimations provided!" << endl;
    return rmse;
  }
  // Check for equal vector sizes
  if(estimations.size() != ground_truth.size()){
    cout << "Error: Different sizes for estimations/ground_truth vectors!" << endl;
    return rmse;
  }
  
  // Calc residuals, square and sum up in rmse
  for(int i=0; i < estimations.size(); ++i){
    VectorXd diff = VectorXd::Zero(estimations.size());
    diff = estimations[i] - ground_truth[i];
    VectorXd residuals = diff.array()*diff.array();
    rmse += residuals;
  }
  
  //calculate the mean
  rmse = rmse/estimations.size();
  rmse = rmse.array().sqrt();
  
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  
  MatrixXd Hj(3,4);
  Hj << 0, 0, 0, 0,
  0, 0, 0, 0,
  0, 0, 0, 0;
  
  // State parameters
  double px = x_state(0);
  double py = x_state(1);
  double vx = x_state(2);
  double vy = x_state(3);
  
  // Some values that get used more than once
  double pxpy = px*px+py*py;
  double pxpy2 = pow(pxpy,0.5);
  double pxpy32 = pow(pxpy,1.5);
  double vxpy = vx*py - vy*px;
  double vypx = vy*px - vx*py;
  
  if(fabs(pxpy < 0.0001)){
    cout << "Division by zero" << endl;
    return Hj;
  }
  
  Hj << (px/pxpy2), (py/pxpy2), 0, 0,
  -(py/pxpy), (px/pxpy), 0, 0,
  py*vxpy/pxpy32, px*vypx/pxpy32, px/pxpy2, py/pxpy2;
  return Hj;
}
