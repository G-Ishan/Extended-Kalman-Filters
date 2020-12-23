#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {

  // Calculate the RMSE

  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  /**
  * Sanity checks:
  * Estimation vector should have values
  * Estimation vector length should be same as ground truth
  */

  if (estimations.size == 0) {
    std::cout << "Estimation vector has no values." << endl;
    return rmse;
  }

  if (estimations.size != ground_truth.size) {
    std::cout << "Size of Estimation vector and Ground Truth do not match." << endl;
    return rmse;
  }

  // Squared residuals
  for (unsigned int i = 0; i < estimations.size(); ++i){

    VectorXd residual = estimations[i] - ground_truth[i];

    //coefficient-wise multiplication
    residual = residual.array()*residual.array();
    rmse += residual;
  }

  rmse = rmse / estimations.size();
  rmse = rmse.array().sqrt();

  return rmse;

}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {

   // Calculate Jacobian

   MatrixXd Hj(3, 4);
   // Recover state parameters
   float px = x_state(0);
   float py = x_state(1);
   float vx = x_state(2);
   float vy = x_state(3);

   // Terms to be used in later calculations
   float c1 = px * px + py * py;
   float c2 = sqrt(c1);
   float c3 = c1 * c2;

   // Check division by zero
   if (fabs(c1) < 0.0001){
      std::cout << "CalculateJacobian () - Error - Division by Zero" << std::endl;
      return Hj;
   }

   // Compute the Jacobian matrix
   Hj << (px / c2), (py / c2), 0, 0,
       -(py / c1), (px / c1), 0, 0,
       py*(vx*py - vy*px) / c3, px*(px*vy - py*vx) / c3, px / c2, py / c2;

   return Hj;

}
