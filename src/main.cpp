#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <stdlib.h>
#include "Eigen/Dense"
#include "FusionEKF.h"
#include "ground_truth_package.h"
#include "measurement_package.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

void check_arguments(int argc, char* argv[]) {
  string usage_instructions = "Usage instructions: ";
  usage_instructions += argv[0];
  usage_instructions += " path/to/input.txt output.txt";

  auto has_valid_args = false;

  // make sure the user has provided input and output files
  if (argc == 1) {
    cerr << usage_instructions << endl;
  } else if (argc == 2) {
    cerr << "Please include an output file.\n" << usage_instructions << endl;
  } else if (argc == 3) {
    has_valid_args = true;
  } else if (argc > 3) {
    cerr << "Too many arguments.\n" << usage_instructions << endl;
  }

  if (!has_valid_args) {
    exit(EXIT_FAILURE);
  }
}

void LoadData(vector<MeasurementPackage>& measurement_pack_list,
              vector<GroundTruthPackage>& gt_pack_list,
              const string in_file_name) {

  ifstream in_file_(in_file_name.c_str(), ifstream::in);

  if (!in_file_.is_open()) {
    cerr << "Cannot open input file: " << in_file_name << endl;
    exit(EXIT_FAILURE);
  }

  string line;

  // prep the measurement packages (each line represents a measurement at a
  // timestamp)
  while (getline(in_file_, line)) {

    string sensor_type;
    MeasurementPackage meas_package;
    GroundTruthPackage gt_package;
    istringstream iss(line);
    long long timestamp;

    // reads first element from the current line
    iss >> sensor_type;
    if (sensor_type == "L") { // LASER MEASUREMENT
      // read measurements at this timestamp
      meas_package.sensor_type_ = MeasurementPackage::LASER;
      meas_package.raw_measurements_ = VectorXd(2);
      float x, y;
      iss >> x >> y >> timestamp;
      meas_package.raw_measurements_ << x, y;
      meas_package.timestamp_ = timestamp;
      measurement_pack_list.push_back(meas_package);
    } else if (sensor_type == "R") { // RADAR MEASUREMENT
      // read measurements at this timestamp
      meas_package.sensor_type_ = MeasurementPackage::RADAR;
      meas_package.raw_measurements_ = VectorXd(3);
      float ro, phi, ro_dot;
      iss >> ro >> phi >> ro_dot >> timestamp;
      meas_package.raw_measurements_ << ro, phi, ro_dot;
      meas_package.timestamp_ = timestamp;
      measurement_pack_list.push_back(meas_package);
    }

    // read ground truth data to compare later
    float x_gt, y_gt, vx_gt, vy_gt;
    iss >> x_gt >> y_gt >> vx_gt >> vy_gt;
    gt_package.gt_values_ = VectorXd(4);
    gt_package.gt_values_ << x_gt, y_gt, vx_gt, vy_gt;
    gt_pack_list.push_back(gt_package);
  }

  // Close file
  if (in_file_.is_open()) {
    in_file_.close();
  }
}

void ProcessData(const vector<MeasurementPackage>& measurement_pack_list,
                 const vector<GroundTruthPackage>& gt_pack_list,
                 const string out_file_name) {

  // Create a Fusion EKF instance
  unique_ptr<FusionEKF> fusionEKF(new FusionEKF);

  // used to compute the RMSE later
  vector<VectorXd> estimations, ground_truth;

  ofstream out_file_(out_file_name.c_str(), ofstream::out);

  if (!out_file_.is_open()) {
    cerr << "Cannot open output file: " << out_file_name << endl;
    exit(EXIT_FAILURE);
  }

  //Call the EKF-based fusion
  size_t N = measurement_pack_list.size();
  for (size_t k = 0; k < N; ++k) {
    // start filtering from the second frame (the speed is unknown in the first
    // frame)
    fusionEKF->ProcessMeasurement(measurement_pack_list[k]);

    cout << "x_ = " << fusionEKF->ekf_->x_ << '\n';
    cout << "P_ = " << fusionEKF->ekf_->P_ << '\n';

    // output the estimation
    out_file_ << fusionEKF->ekf_->x_(0) << "\t";
    out_file_ << fusionEKF->ekf_->x_(1) << "\t";
    out_file_ << fusionEKF->ekf_->x_(2) << "\t";
    out_file_ << fusionEKF->ekf_->x_(3) << "\t";

    // output the measurements
    if (measurement_pack_list[k].sensor_type_ == MeasurementPackage::LASER) {
      // output the estimation
      out_file_ << measurement_pack_list[k].raw_measurements_(0) << "\t";
      out_file_ << measurement_pack_list[k].raw_measurements_(1) << "\t";
    } else if (measurement_pack_list[k].sensor_type_ == MeasurementPackage::RADAR) {
      // output the estimation in the cartesian coordinates
      float ro = measurement_pack_list[k].raw_measurements_(0);
      float phi = measurement_pack_list[k].raw_measurements_(1);
      out_file_ << ro * cos(phi) << "\t"; // p1_meas
      out_file_ << ro * sin(phi) << "\t"; // ps_meas
    }

    // output the ground truth packages
    out_file_ << gt_pack_list[k].gt_values_(0) << "\t";
    out_file_ << gt_pack_list[k].gt_values_(1) << "\t";
    out_file_ << gt_pack_list[k].gt_values_(2) << "\t";
    out_file_ << gt_pack_list[k].gt_values_(3) << "\n";

    estimations.push_back(fusionEKF->ekf_->x_);
    ground_truth.push_back(gt_pack_list[k].gt_values_);
  }

  // compute the accuracy (RMSE)
  cout << "Accuracy - RMSE:" << '\n' << Tools::CalculateRMSE(estimations, ground_truth) << '\n';

  // close files
  if (out_file_.is_open()) {
    out_file_.close();
  }
}

int main(int argc, char* argv[]) {

  check_arguments(argc, argv);
  string in_file_name = argv[1];
  string out_file_name = argv[2];

  vector<MeasurementPackage> measurement_pack_list;
  vector<GroundTruthPackage> gt_pack_list;
  LoadData(measurement_pack_list, gt_pack_list, in_file_name);
  ProcessData(measurement_pack_list, gt_pack_list, out_file_name);
  return EXIT_SUCCESS;
}
