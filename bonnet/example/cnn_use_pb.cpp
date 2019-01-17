/* Copyright 2017 Andres Milioto. All Rights Reserved.
 *
 *  This file is part of Bonnet.
 *
 *  Bonnet is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  Bonnet is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with Bonnet. If not, see <http://www.gnu.org/licenses/>.
 */

// opencv stuff
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// c++ stuff
#include <chrono>
#include <iomanip>  // for setfill
#include <iostream>
#include <string>

// net stuff
#include <bonnet.hpp>

// boost
#include <boost/program_options.hpp>
namespace po = boost::program_options;
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

int main(int argc, const char *argv[]) {
  // define options
  std::vector<std::string> images;
  std::string log = "/tmp/net_predict_log_cpp/";
  std::string path;
  bool verbose = false;
  std::string device = "/gpu:0";
  std::string backend = "tf";

  // Parse options
  try {
    po::options_description desc{"Options"};
    desc.add_options()("help,h", "Help screen")(
        "image,i", po::value<std::vector<std::string>>(&images)->multitoken(),
        "Images to infer. No Default")(
        "log,l", po::value<std::string>(),
        "Directory to log output of predictions.")(
        "path,p", po::value<std::string>(),
        "Directory to get the frozen pb model. No default")(
        "verbose,v", po::bool_switch(),
        "Verbose mode. Calculates profile (time to run)")(
        "dev,d", po::value<std::string>(),
        "Device to run on. Example (/gpu:0 or /cpu:0)")(
        "backend,b", po::value<std::string>(),
        "Backend. Tensorflow and TensorRT for now. Later MovidiusNCSDK.");

    po::variables_map vm;
    po::store(parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    std::cout << std::setfill('=') << std::setw(80) << "" << std::endl;

    if (vm.count("help")) {
      std::cout << desc << std::endl;
      return 0;
    }

    if (!vm["image"].empty()) {
      for (const auto &image : images) {
        std::cout << "image: " << image << std::endl;
      }
    } else {
      std::cerr << "No images! See --help (-h) for help. Exiting" << std::endl;
      return 1;
    }

    if (vm.count("log")) {
      log = vm["log"].as<std::string>() + "/";  // make sure path is valid
      std::cout << "log: " << log << std::endl;
    } else {
      std::cout << "log: " << log << ". Using default!" << std::endl;
    }
    if (vm.count("path")) {
      path = vm["path"].as<std::string>() + "/";  // make sure path is valid
      std::cout << "path: " << path << std::endl;
    } else {
      std::cerr << "No path! See --help (-h) for help. Exiting" << std::endl;
      return 1;
    }
    if (vm.count("verbose")) {
      verbose = vm["verbose"].as<bool>();
      std::cout << "verbose: " << verbose << std::endl;
    } else {
      std::cout << "verbose: " << verbose << ". Using default!" << std::endl;
    }
    if (vm.count("dev")) {
      device = vm["dev"].as<std::string>();
      std::cout << "device: " << device << std::endl;
    } else {
      std::cout << "device: " << device << ". Using default!" << std::endl;
    }
    if (vm.count("backend")) {
      backend = vm["backend"].as<std::string>();
      std::cout << "backend: " << backend << std::endl;
    } else {
      std::cout << "backend: " << backend << ". Using default!" << std::endl;
    }

    std::cout << std::setfill('=') << std::setw(80) << "" << std::endl;
  } catch (const po::error &ex) {
    std::cerr << ex.what() << std::endl;
    return 1;
  }

  // create the log folder
  try {
    if (fs::exists(log)) {
      fs::remove_all(log);
    }
    fs::create_directory(log);
  } catch (fs::filesystem_error const &e) {
    std::cerr << "Failed to create the log directory: " << log << std::endl;
    std::cerr << e.what();
    return 1;
  }
  std::cout << "Successfully created log directory: " << log << std::endl;

  // network stuff
  cv::Mat mask_argmax;
  bonnet::retCode status;
  std::unique_ptr<bonnet::Bonnet> net;

  // initialize network
  try {
    net = std::unique_ptr<bonnet::Bonnet>(
        new bonnet::Bonnet(path, backend, device, verbose));
  } catch (const std::invalid_argument &e) {
    std::cerr << "Unable to create network. " << std::endl
              << e.what() << std::endl;
    return 1;
  } catch (const std::runtime_error &e) {
    std::cerr << "Unable to init. network. " << std::endl
              << e.what() << std::endl;
    return 1;
  }

  // predict each image
  for (auto image : images) {
    std::cout << std::setfill('=') << std::setw(80) << "" << std::endl;
    std::cout << "Predicting image: " << image << std::endl;

    // Open an image
    cv::Mat cv_img = cv::imread(image);
    // Check for invalid input
    if (!cv_img.data) {
      std::cerr << "Could not open or find the image" << std::endl;
      return 1;
    }

    // predict
    status = net->infer(cv_img, mask_argmax, verbose);
    if (status != bonnet::CNN_OK) {
      std::cerr << "Failed to run CNN." << std::endl;
      return 1;
    }

    // convert xentropy mask to colors using dictionary
    cv::Mat mask_bgr(mask_argmax.rows, mask_argmax.cols, CV_8UC3);
    status = net->color(mask_argmax, mask_bgr, verbose);
    if (status != bonnet::CNN_OK) {
      std::cerr << "Failed to color result of CNN." << std::endl;
      return 1;
    }

    // save image to log directory
    std::string image_log_path =
        log + "/" + fs::path(image).stem().string() + ".png";
    std::cout << "Saving this image to " << image_log_path << std::endl;
    cv::imwrite(image_log_path, mask_bgr);

    // print the output
    if (verbose) {
      cv::namedWindow("Display window",
                      cv::WINDOW_AUTOSIZE);    // Create a window for display.
      cv::imshow("Display window", mask_bgr);  // Show our image inside
      cv::waitKey(0);  // Wait for a keystroke in the window}
    }
    std::cout << std::setfill('=') << std::setw(80) << "" << std::endl;
  }

  return 0;
}
