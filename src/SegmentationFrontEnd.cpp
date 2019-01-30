#include "FUSES/SegmentationFrontEnd.h"

namespace CRFsegmentation {

// load parameters from a yaml file
void loadParameters(SPParam& spParam, SegParam& segParam,
    const std::string& paramFile) {
  SegmentationFrontEndParams fp;
  fp.parseYAML(paramFile);

  spParam.nr_sp = fp.num_sp_;
  spParam.sp_iterations = fp.sp_iterations_;
  spParam.ratio = fp.ratio_;
  spParam.min_element_size = fp.min_element_size_;
  spParam.nr_levels = fp.nr_levels_;
  spParam.prior = fp.prior_;
  spParam.histogram_bins = fp.histogram_bins_;

  segParam.lambda1 = fp.lambda1_;
  segParam.lambda2 = fp.lambda2_;
  segParam.beta = fp.beta_;
};

/* ------------------------------------------------------------------------- */
// constructor: get binary and unary factors from a csv file
SegmentationFrontEnd::SegmentationFrontEnd(const std::string& filename,
    int imgId, int verbose) : imgId_(imgId){
  std::ifstream file(filename);
  std::string line;
  std::vector<std::string> fields;
  if (!file.good()) {
    std::cout <<  "Could not read file " << filename << std::endl;
    throw std::runtime_error("File not found.");
  }

  if(verbose > 1) {
    std::cout << "\n$$$$$$$$$$$$$$$$$$$$$ SegmentationFrontEnd: reading from "
        "csv file $$$$$$$$$$$$$$$$$$$$$" << std::endl;
  }

  // Read the number of nodes
  auto timeStart = myUtils::Timer::tic();
  getline(file, line, ',');
  crf_.nrNodes = stoi(line);

  // Read the number of unary terms
  getline(file, line);
  int nrUnaries = stoi(line);

  // Read the unary factors
  UnaryFactor unaryFactor;
  for(size_t i=0; i<nrUnaries; i++) {
    getline(file, line);
    // extract node index and label
    boost::split(fields, line, boost::is_any_of(","));
    unaryFactor.node = stoi(fields[0]);
    unaryFactor.label = stoi(fields[1]);
    // store unary factor in the vector array
    crf_.unaryFactors.push_back(unaryFactor);
    // update the number of classes
    if(crf_.nrClasses < unaryFactor.label+1){
      crf_.nrClasses = unaryFactor.label+1;
    }
  }

  // Read the binary factors
  BinaryFactor binaryFactor;
  while (getline(file, line)) {
    boost::split(fields, line, boost::is_any_of(","));
    // extract node indices and weight
    binaryFactor.firstNode = stod(fields[0]);
    binaryFactor.secondNode = stod(fields[1]);
    binaryFactor.weight = stod(fields[2]);
    // store binary factor in the vector array
    crf_.binaryFactors.push_back(binaryFactor);
  }
  // Close the File
  file.close();

  if(verbose > 1) {
    std::cout << crf_.unaryFactors.size() << " out of " << crf_.nrNodes
        << " superpixel nodes are classified." << std::endl;
    std::cout << "crf_.nrClasses = " << crf_.nrClasses << std::endl;
    std::cout << crf_.binaryFactors.size() << " binary factors are computed."
        << std::endl;
  }
  if(verbose > 0) {
    std::cout << "Done reading CRF from file. Time elapsed = "
        << myUtils::Timer::toc(timeStart) << "s.\n" << std::endl;
  }
}

// constructor: get binary factors from images
SegmentationFrontEnd::SegmentationFrontEnd(const cv::Mat& image,
    const SegParam& segParam, const SPParam& spParam, int spAlgorithm, int imgId,
    int verbose) : imgId_(imgId) {
  if(verbose > 1) {
    std::cout << "\n$$$$$$$$$$$$$$$$$$$$$ SegmentationFrontEnd: taking input "
        "image $$$$$$$$$$$$$$$$$$$$$" << std::endl;
  }

  // load image
  auto spStart = myUtils::Timer::tic();
  cv::Mat inputCopy, mask, converted, labels;
  image.copyTo(inputCopy);

  // create superpixels
  createSuperpixels(inputCopy, labels, mask, spParam, spAlgorithm, verbose);

  // display output
  if(verbose > 1) {
    std::cout << crf_.nrNodes << " superpixels created." << std::endl;
  }
  if(verbose > 0) {
    std::cout << "Done creating superpixels. Time elapsed = "
        << myUtils::Timer::toc(spStart) << "s.\n" << std::endl;
  }
  if(verbose > 2) {
    inputCopy.setTo(cv::Scalar(0, 0, 255), mask);
    cv::imshow("Segmented", inputCopy);
    cv::waitKey(0);
  }

  // create superpixel array and connectivity matrix
  auto spDataStart = myUtils::Timer::tic();
  superPixels_.resize(crf_.nrNodes);
  gtsam::Matrix connectMatrix = gtsam::Matrix::Zero(crf_.nrNodes,
      crf_.nrNodes);

  for(size_t i=0; i<labels.rows; i++) {
    for(size_t j=0; j<labels.cols; j++){
      int spIndex = labels.at<int>(i,j);    // superpixel index

      // add color and pixel location to the superPixels_ vector
      cv::Vec3b color = image.at<cv::Vec3b>(i,j);
      std::pair<int, int> pixel(i,j);
      superPixels_[spIndex].pixels.push_back(pixel);
      superPixels_[spIndex].colors.push_back(color);
      superPixels_[spIndex].nrPixels += 1;

      // update connectMatrix
      if(i<labels.rows-1) { // compare with bottom pixel
        int bottomIndex = labels.at<int>(i+1,j);
        if(bottomIndex != spIndex) {
          connectMatrix(spIndex, bottomIndex) = 1;
          connectMatrix(bottomIndex, spIndex) = 1;
        }
      }
      if(j<labels.cols-1) { // compare with right pixel
        int rightIndex = labels.at<int>(i,j+1);
        if(rightIndex != spIndex) {
          connectMatrix(spIndex, rightIndex) = 1;
          connectMatrix(rightIndex, spIndex) = 1;
        }
      }
    }
  }
  if(verbose > 1) {
    std::cout << "Number of neighboring pairs: " << connectMatrix.sum()/2
        << std::endl;
    std::cout << superPixels_.size() << " elements in superPixels vector"
        << std::endl;
  }

  // Compute average color and center for each superpixel
  for(SuperPixel& SP : superPixels_) {
    for(size_t i=0; i<SP.nrPixels; ++i) {
      // avgColor is initially set to Vec3d(0,0,0) by default
      cv::Vec3b& color = SP.colors[i];
      SP.avgColor.val[0] += double(color.val[0]) / SP.nrPixels;   // B
      SP.avgColor.val[1] += double(color.val[1]) / SP.nrPixels;   // G
      SP.avgColor.val[2] += double(color.val[2]) / SP.nrPixels;   // R
      // center is initially set to pair<0,0> by default
      std::pair<int, int>& pixel = SP.pixels[i];
      SP.center.first += double(pixel.first) / SP.nrPixels;       // row
      SP.center.second += double(pixel.second) / SP.nrPixels;     // col
    }
  }
  if(verbose > 0) {
    std::cout << "Done computing superPixels_ data member. Time elapsed = "
        << myUtils::Timer::toc(spDataStart) << "s.\n" << std::endl;
  }

  // compute binary factors
  getBinaryFactors(connectMatrix, segParam, verbose);
}

// constructor: given a segmented image, get binary factors from images
SegmentationFrontEnd::SegmentationFrontEnd(const cv::Mat& image,
    const SegParam& segParam, const cv::Mat& labels, int nrNodes, int imgId,
    int verbose) : imgId_(imgId) {
  if(verbose > 1) {
    std::cout << "\n$$$$$$$$$$$$$$$$$$$$$ SegmentationFrontEnd: taking input "
        "image $$$$$$$$$$$$$$$$$$$$$" << std::endl;
  }

  // check image and label dimensions
  crf_.nrNodes = nrNodes;
  if (image.rows != labels.rows || image.cols != labels.cols) {
    std::cout <<  "Image and label size mismatch" << std::endl;
    throw std::runtime_error("SegmentationFrontEnd cannot be initialized.");
  }

  // display output
  if(verbose > 1) {
    std::cout << nrNodes << " superpixels from input labels." << std::endl;
  }

  // create superpixel array and connectivity matrix
  auto spDataStart = myUtils::Timer::tic();
  superPixels_.resize(crf_.nrNodes);
  gtsam::Matrix connectMatrix = gtsam::Matrix::Zero(crf_.nrNodes,
      crf_.nrNodes);
  cv::Mat colorDiffMat(cv::Size(crf_.nrNodes, crf_.nrNodes), CV_32FC3);

  for(size_t i=0; i<labels.rows; i++) {
    for(size_t j=0; j<labels.cols; j++){
      int spIndex = labels.at<int>(i,j);    // superpixel index

      // add color and pixel location to the superPixels_ vector
      cv::Vec3b color = image.at<cv::Vec3b>(i,j);
      std::pair<int, int> pixel(i,j);
      superPixels_[spIndex].pixels.push_back(pixel);
      superPixels_[spIndex].colors.push_back(color);
      superPixels_[spIndex].nrPixels += 1;

      // update connectMatrix
      if(i<labels.rows-1) { // compare with bottom pixel
        int bottomIndex = labels.at<int>(i+1,j);
        if(bottomIndex != spIndex) {
          connectMatrix(spIndex, bottomIndex) = 1;
          connectMatrix(bottomIndex, spIndex) = 1;
        }
      }
      if(j<labels.cols-1) { // compare with right pixel
        int rightIndex = labels.at<int>(i,j+1);
        if(rightIndex != spIndex) {
          connectMatrix(spIndex, rightIndex) = 1;
          connectMatrix(rightIndex, spIndex) = 1;
        }
      }
    }
  }
  // Compute average color and center for each superpixel
  for(SuperPixel& SP : superPixels_) {
    for(size_t i=0; i<SP.nrPixels; ++i) {
      // avgColor is initially set to Vec3d(0,0,0) by default
      cv::Vec3b& color = SP.colors[i];
      SP.avgColor.val[0] += double(color.val[0]) / SP.nrPixels;   // B
      SP.avgColor.val[1] += double(color.val[1]) / SP.nrPixels;   // G
      SP.avgColor.val[2] += double(color.val[2]) / SP.nrPixels;   // R
      // center is initially set to pair<0,0> by default
      std::pair<int, int>& pixel = SP.pixels[i];
      SP.center.first += double(pixel.first) / SP.nrPixels;       // row
      SP.center.second += double(pixel.second) / SP.nrPixels;     // col
    }
  }

  if(verbose > 1) {
    std::cout << "Number of neighboring pairs: " << connectMatrix.sum()/2
        << std::endl;
    std::cout << superPixels_.size() << " elements in superPixels vector"
        << std::endl;
  }

  if(verbose > 0) {
    std::cout << "Done computing superPixels_ data member. Time elapsed = "
        << myUtils::Timer::toc(spDataStart) << "s.\n" << std::endl;
  }

  // compute binary factors
  getBinaryFactors(connectMatrix, segParam, verbose);

  //double radius = 2 * sqrt(labels.rows * labels.cols / crf_.nrNodes);
  //getBinaryFactors(connectMatrix, radius, segParam, verbose);
}

/* ------------------------------------------------------------------------- */
// create super pixels using specified parameter and algorithm
// the segmentation will be stored in labels
// the contours will be stored in mask if verbose > 2
void SegmentationFrontEnd::createSuperpixels(cv::Mat& image, cv::Mat& labels,
    cv::Mat& mask, const SPParam& spParam, int spAlgorithm, int verbose) {
  // display parameters
  if(verbose > 2) {
    std::cout << "superpixel parameters:" << std::endl;
    std::cout << "   num_sp = " << spParam.nr_sp << std::endl;
    std::cout << "   sp_iterations = " << spParam.sp_iterations << std::endl;

    switch(spAlgorithm) {
    case LSC: {
      std::cout << "   ratio = " << spParam.ratio << std::endl;
      std::cout << "   min_element_size = " << spParam.min_element_size
          << std::endl;
      break;
    }
    case SEEDS: {
      std::cout << "   nr_levels = " << spParam.nr_levels << std::endl;
      std::cout << "   prior = " << spParam.prior << std::endl;
      std::cout << "   histogram_bins = " << spParam.histogram_bins
          << std::endl;
      break;
    }
    case SLIC: {
      std::cout << "   min_element_size = " << spParam.min_element_size
          << std::endl;
      break;
    }
    case SLICO: {
      std::cout << "   min_element_size = " << spParam.min_element_size
          << std::endl;
      break;
    }
    case MSLIC: {
      std::cout << "   min_element_size = " << spParam.min_element_size
          << std::endl;
      break;
    }
    }
  }

  // run superpixel algorithm
  cv::Mat converted; // converted image from BGR to Lab
  switch(spAlgorithm) {
  case LSC: {
    if(verbose > 1) {
      std::cout << "Using Superpixel LSC algorithm" << std::endl;
    }
    int region = sqrt((image.cols)*(image.rows)/spParam.nr_sp);
    cv::cvtColor(image, converted, cv::COLOR_BGR2Lab);
    cv::Ptr<cv::ximgproc::SuperpixelLSC> lsc = cv::ximgproc::
        createSuperpixelLSC(converted, region, spParam.ratio);
    lsc->iterate(spParam.sp_iterations);
    lsc->enforceLabelConnectivity(spParam.min_element_size);
    crf_.nrNodes = lsc->getNumberOfSuperpixels();
    lsc->getLabels(labels);

    if(verbose > 2) {
      lsc->getLabelContourMask(mask, true);
    }
    break;
  }
  case SEEDS: {
    if(verbose > 1) {
      std::cout << "Using Superpixel SEEDS algorithm" << std::endl;
    }
    cv::cvtColor(image, converted, cv::COLOR_BGR2HSV);
    cv::Ptr<cv::ximgproc::SuperpixelSEEDS> seeds = cv::ximgproc::
        createSuperpixelSEEDS(converted.cols, converted.rows,
            converted.channels(), spParam.nr_sp, spParam.nr_levels,
            spParam.prior, spParam.histogram_bins);
    seeds->iterate(converted, spParam.sp_iterations);
    crf_.nrNodes = seeds->getNumberOfSuperpixels();
    seeds->getLabels(labels);

    if(verbose > 2) {
      seeds->getLabelContourMask(mask, true);
    }
    break;
  }
  case SLIC: {
    if(verbose > 1) {
      std::cout << "Using Superpixel SLIC algorithm" << std::endl;
    }
    cv::cvtColor(image, converted, cv::COLOR_BGR2Lab);
    int region_size = (image.cols)*(image.rows)/(spParam.nr_sp*20);
    cv::Ptr<cv::ximgproc::SuperpixelSLIC> slic = cv::ximgproc::
        createSuperpixelSLIC(converted, cv::ximgproc::SLIC, region_size);
    slic->iterate(spParam.sp_iterations);
    slic->enforceLabelConnectivity(spParam.min_element_size);
    crf_.nrNodes = slic->getNumberOfSuperpixels();
    slic->getLabels(labels);

    if(verbose > 2) {
      slic->getLabelContourMask(mask, true);
    }
    break;
  }
  case SLICO: {
    if(verbose > 1) {
      std::cout << "Using Superpixel SLICO algorithm" << std::endl;
    }
    cv::cvtColor(image, converted, cv::COLOR_BGR2Lab);
    int region_size = (image.cols)*(image.rows)/(spParam.nr_sp*20);
    cv::Ptr<cv::ximgproc::SuperpixelSLIC> slic = cv::ximgproc::
        createSuperpixelSLIC(converted, cv::ximgproc::SLICO, region_size);
    slic->iterate(spParam.sp_iterations);
    slic->enforceLabelConnectivity(spParam.min_element_size);
    crf_.nrNodes = slic->getNumberOfSuperpixels();
    slic->getLabels(labels);

    if(verbose > 2) {
      slic->getLabelContourMask(mask, true);
    }
    break;
  }
  case MSLIC: {
    if(verbose > 1) {
      std::cout << "Using Superpixel MSLIC algorithm" << std::endl;
    }
    cv::cvtColor(image, converted, cv::COLOR_BGR2Lab);
    int region_size = (image.cols)*(image.rows)/(spParam.nr_sp*20);
    cv::Ptr<cv::ximgproc::SuperpixelSLIC> slic = cv::ximgproc::
        createSuperpixelSLIC(converted, cv::ximgproc::MSLIC, region_size);
    slic->iterate(spParam.sp_iterations);
    slic->enforceLabelConnectivity(spParam.min_element_size);
    crf_.nrNodes = slic->getNumberOfSuperpixels();
    slic->getLabels(labels);

    if(verbose > 2) {
      slic->getLabelContourMask(mask, true);
    }
    break;
  }
  }
}

/* ------------------------------------------------------------------------- */
// compute binary factors using connectivity matrix
// and update colorVariance data member
// Note: CRF only stores half of the binary terms to save time and memory
// (i.e. if edge(i,j) is stored, then edge(j,i) will not be stored but
// should have the same weight)
void SegmentationFrontEnd::getBinaryFactors(const gtsam::Matrix& connectMatrix,
    const SegParam& segParam, int verbose) {
  if (crf_.nrNodes == 0) {
    std::cout << "0 superpixel node" << std::endl;
    throw std::runtime_error("getBinaryFactors (SegmentationFrontEnd): "
        "cannot compute binary factors without super pixel nodes");
  }

  if(verbose > 1) {
    std::cout << "\n$$$$$$$$$ Computing Binary Factors from Input Image $$$$$"
        "$$$$" << std::endl;
  }

  if(verbose > 2) {
    std::cout << "segmentation parameters:" << std::endl;
    std::cout << "   lambda1 = " << segParam.lambda1 << std::endl;
    std::cout << "   lambda2 = " << segParam.lambda2 << std::endl;
    std::cout << "   beta = " << segParam.beta << std::endl;
  }

  auto bfStart = myUtils::Timer::tic();
  crf_.binaryFactors.clear();
  colorVariance = 0;

  for(int i=0; i<crf_.nrNodes; ++i) {
    for(int j=0; j<i; ++j) {
      if(connectMatrix(i, j) > 0.1) { // use very small number to get rid of 0
        BinaryFactor binaryFactor;
        binaryFactor.firstNode = j;
        binaryFactor.secondNode = i;
        double colorDiff = pow(   // colorDiff = || color_i - color_j ||^2
            norm(superPixels_[i].avgColor - superPixels_[j].avgColor), 2.0);
        binaryFactor.weight = (segParam.lambda1 +
            segParam.lambda2 * exp(-colorDiff * segParam.beta));
        colorVariance += colorDiff;
        crf_.binaryFactors.push_back(binaryFactor);
      }
    }
  }
  colorVariance = colorVariance / crf_.binaryFactors.size();

  if(verbose > 1) {
    std::cout << crf_.binaryFactors.size() << " binary factors are computed."
        << std::endl;
    std::cout << "The variance of color difference between neighboring "
        "superpixels is " << colorVariance << std::endl;
  }

  if(verbose > 0) {
    std::cout << "Done computing binary factors. Time elapsed = "
        << myUtils::Timer::toc(bfStart) << "s.\n" << std::endl;
  }
}

// in addition to neighboring SP, this add connections with other *close* SPs
void SegmentationFrontEnd::getBinaryFactors(const gtsam::Matrix& connectMatrix,
    double radius, const SegParam& segParam, int verbose) {
  if (crf_.nrNodes == 0) {
    std::cout << "0 superpixel node" << std::endl;
    throw std::runtime_error("getBinaryFactors (SegmentationFrontEnd): "
        "cannot compute binary factors without super pixel nodes");
  }

  if(verbose > 1) {
    std::cout << "\n$$$$$$$$$ Computing Binary Factors from Input Image $$$$$"
        "$$$$" << std::endl;
  }

  if(verbose > 2) {
    std::cout << "segmentation parameters:" << std::endl;
    std::cout << "   lambda1 = " << segParam.lambda1 << std::endl;
    std::cout << "   lambda2 = " << segParam.lambda2 << std::endl;
    std::cout << "   beta = " << segParam.beta << std::endl;
  }

  auto bfStart = myUtils::Timer::tic();
  crf_.binaryFactors.clear();
  colorVariance = 0;

  for(int i=0; i<crf_.nrNodes; ++i) {
    for(int j=0; j<i; ++j) {
      if(connectMatrix(i, j) > 0.1) { // use very small number to get rid of 0
        BinaryFactor binaryFactor;
        binaryFactor.firstNode = j;
        binaryFactor.secondNode = i;
        double colorDiff = pow(   // colorDiff = || color_i - color_j ||^2
            norm(superPixels_[i].avgColor - superPixels_[j].avgColor), 2.0);
        binaryFactor.weight = (segParam.lambda1 +
            segParam.lambda2 * exp(-colorDiff * segParam.beta));
        colorVariance += colorDiff;
        crf_.binaryFactors.push_back(binaryFactor);
      } else { // check if two SP is within specified radius
        double dist = sqrt(pow(superPixels_[i].center.first
            - superPixels_[j].center.first, 2.0)
            + pow(superPixels_[i].center.second
                - superPixels_[j].center.second, 2.0));
        if(dist < radius) {
          BinaryFactor binaryFactor;
          binaryFactor.firstNode = j;
          binaryFactor.secondNode = i;
          double colorDiff = pow(   // colorDiff = || color_i - color_j ||^2
              norm(superPixels_[i].avgColor - superPixels_[j].avgColor), 2.0);
          double scale = (radius-dist) / radius;
          binaryFactor.weight = scale * (segParam.lambda1 +
              segParam.lambda2 * exp(-colorDiff * segParam.beta));
          colorVariance += colorDiff;
          crf_.binaryFactors.push_back(binaryFactor);
        }
      }
    }
  }
  colorVariance = colorVariance / crf_.binaryFactors.size();

  if(verbose > 1) {
    std::cout << crf_.binaryFactors.size() << " binary factors are computed."
        << std::endl;
    std::cout << "The variance of color difference between neighboring "
        "superpixels is " << colorVariance << std::endl;
  }

  if(verbose > 0) {
    std::cout << "Done computing binary factors. Time elapsed = "
        << myUtils::Timer::toc(bfStart) << "s.\n" << std::endl;
  }
}

// this is used when a colorDiffMat is given so that the average SP color in
// the data member is ignored
void SegmentationFrontEnd::getBinaryFactors(const gtsam::Matrix& connectMatrix,
    const cv::Mat& colorDiffMat, const SegParam& segParam, int verbose) {
  if (crf_.nrNodes == 0) {
    std::cout << "0 superpixel node" << std::endl;
    throw std::runtime_error("getBinaryFactors (SegmentationFrontEnd): "
        "cannot compute binary factors without super pixel nodes");
  }

  if(verbose > 1) {
    std::cout << "\n$$$$$$$$$ Computing Binary Factors from Input Image $$$$$"
        "$$$$" << std::endl;
  }

  if(verbose > 2) {
    std::cout << "segmentation parameters:" << std::endl;
    std::cout << "   lambda1 = " << segParam.lambda1 << std::endl;
    std::cout << "   lambda2 = " << segParam.lambda2 << std::endl;
    std::cout << "   beta = " << segParam.beta << std::endl;
  }

  auto bfStart = myUtils::Timer::tic();
  crf_.binaryFactors.clear();
  colorVariance = 0;

  for(int i=0; i<crf_.nrNodes; ++i) {
    for(int j=0; j<i; ++j) {
      if(connectMatrix(i, j) > 0.9) { // use very small number to get rid of 0
        BinaryFactor binaryFactor;
        binaryFactor.firstNode = j;
        binaryFactor.secondNode = i;
        double colorDiff = pow(norm(colorDiffMat.at<cv::Vec3d>(i, j)/
            connectMatrix(i, j)), 2.0);
        binaryFactor.weight = (segParam.lambda1 +
            segParam.lambda2 * exp(-colorDiff * segParam.beta));
        colorVariance += colorDiff;
        crf_.binaryFactors.push_back(binaryFactor);
      }
    }
  }
  colorVariance = colorVariance / crf_.binaryFactors.size();

  if(verbose > 1) {
    std::cout << crf_.binaryFactors.size() << " binary factors are computed."
        << std::endl;
    std::cout << "The variance of color difference between neighboring "
        "superpixels is " << colorVariance << std::endl;
  }

  if(verbose > 0) {
    std::cout << "Done computing binary factors. Time elapsed = "
        << myUtils::Timer::toc(bfStart) << "s.\n" << std::endl;
  }
}

/* ------------------------------------------------------------------------- */
// read beta from average color of each superpixel for this image
// Note: assume zero mean and beta = (2*variance)^-1
void SegmentationFrontEnd::correctBinary(const std::vector<size_t>& labels,
    double attractiveWeight, double repulsiveWeight) {
  for(auto& binaryFactor : crf_.binaryFactors) {
    size_t label1 = labels[binaryFactor.firstNode];
    size_t label2 = labels[binaryFactor.secondNode];
    if(abs(19 - label1) < 1 || abs(19 - label2) < 1) { // ignore class 19
      binaryFactor.weight = 0.0;
    } else if(abs(label1 - label2) < 1)  { // same label, use attractiveWeight
      binaryFactor.weight = attractiveWeight;
    } else {                           // different label, use repulsiveWeight
      binaryFactor.weight = repulsiveWeight;
    }
  }
}

/* ------------------------------------------------------------------------- */
// read beta from average color of each superpixel for this image
// Note: assume zero mean and beta = (2*variance)^-1
double SegmentationFrontEnd::getBeta(const gtsam::Matrix& connectMatrix) const {
  double n = connectMatrix.sum()/2;
  double var = 0;

  for(size_t i=0; i<crf_.nrNodes; ++i) {
    for(size_t j=0; j<i; ++j) {
      if(connectMatrix(i, j) == 1) {
        double colorDiff = pow(norm(superPixels_[i].avgColor -
            superPixels_[j].avgColor), 2.0);
        var += colorDiff/n;
      }
    }
  }

  return 1.0/(2*var);
}

/* ------------------------------------------------------------------------- */
// read beta from average color of each superpixel for this image
// Note: assume zero mean and beta = (2*variance)^-1
double SegmentationFrontEnd::getBeta(const gtsam::Matrix& connectMatrix,
    const cv::Mat& colorDiffMat) const {
  double n = 0;
  double var = 0;

  for(size_t i=0; i<crf_.nrNodes; ++i) {
    for(size_t j=0; j<i; ++j) {
      if(connectMatrix(i, j) > 0.9) {
        double colorDiff = pow(norm(colorDiffMat.at<cv::Vec3d>(i, j)/
            connectMatrix(i, j)), 2.0);
        var += colorDiff;
        n++;
      }
    }
  }
  var = var / n;

  return 1.0/(2*var);
}

/* ------------------------------------------------------------------------- */
// read ground-truth labels from label image
// (input image stores label ID as uint8_t)
// label ID less than 0 is assumed to be in the last class (i.e. nrClasses-1)
void SegmentationFrontEnd::getGTLabelsFromID(const cv::Mat& labels,
    size_t nrClasses, const std::vector<size_t>& labelVec, int verbose) {
  if (crf_.nrNodes == 0) {
    std::cout << "0 superpixel node" << std::endl;
    throw std::runtime_error("getGTUnaryFactors (SegmentationFrontEnd): "
        "cannot find labels without super pixel nodes");
  }
  spLabels_.clear();

  if(verbose > 1) {
    std::cout << "The ground-truth ID image should contain " << nrClasses
        << " classes." << std::endl;
  }
  if(verbose > 2) {
    cv::imshow("Ground truth instance ID", labels);
    cv::waitKey(0);
  }

  for(auto const& superPixel : superPixels_) {
    // store the number of pixels belonging to each class for a superpixel
    std::vector<int> labelCount;
    for(int i=0; i<nrClasses; ++i) { // initialize labelCount
      labelCount.push_back(0);
    }
    for(auto const& pixel: superPixel.pixels) {
      uint8_t gray = labels.at<uint8_t>(pixel.first, pixel.second);
      size_t label;
      if(gray < 0) {
        label = nrClasses-1;
      } else {
        label = labelVec[gray];
      }
      labelCount[label] += 1;
    }
    std::vector<int>::const_iterator it = std::max_element(labelCount.begin(),
        labelCount.end());
    int label = it - labelCount.begin();
    spLabels_.push_back(label);
  }

  if(verbose > 2) {
    cv::imshow("Image label", labels);
    cv::waitKey(0);
  }

}

/* ------------------------------------------------------------------------- */
// get unary factors from bonnet mask
void SegmentationFrontEnd::getUnaryFactorsFromBonnet(const cv::Mat& bonnetLabel,
    size_t dstCols, size_t dstRows, bool constUnary, int verbose) {
  if (crf_.nrNodes == 0) {
    std::cout << "0 superpixel node" << std::endl;
    throw std::runtime_error("getUnaryFactorsFromBonnet "
        "(SegmentationFrontEnd): cannot find labels without super pixel "
        "nodes");
  }

  if(verbose > 1) {
    std::cout << "\n$$$$$$$$$ Computing Unary Factors from Bonnet $$$$$"
        "$$$$" << std::endl;
  }
  auto start = myUtils::Timer::tic();

  cv::Mat inputLabel;
  cv::resize(bonnetLabel, inputLabel, cv::Size(dstCols, dstRows), 0, 0,
      cv::INTER_NEAREST);

  crf_.unaryFactors.clear(); // clear crf_.unaryFactors vector
  crf_.nrClasses = 20;       // bonnet has 20 classes for cityscape dataset

  // for each sp, compute the number of pixels belonging to each class
  std::vector<std::vector<size_t>> totalLabelCount;

  for(auto const& superPixel : superPixels_) {
    // store the number of pixels belonging to each class for a superpixel
    std::vector<size_t> labelCount;
    for(int i=0; i<crf_.nrClasses; ++i) { // initialize labelCount
      labelCount.push_back(0);
    }
    for(auto const& pixel: superPixel.pixels) {
      int label = static_cast<int>
      (inputLabel.at<uint8_t>(pixel.first, pixel.second));
      labelCount[label] += 1;
    }
    totalLabelCount.push_back(labelCount);// add labelCount to totalLabelCount
  }

  // set unary terms
  for(int i=0; i<crf_.nrNodes; ++i) {
    addUnaryFactorsFromLabelCount(constUnary, i, totalLabelCount[i]);
  }

  if(verbose > 1) {
    std::cout << crf_.unaryFactors.size() << " out of " << crf_.nrNodes
        << " superpixel nodes are classified." << std::endl;
  }

  if(verbose > 0) {
    std::cout << "Done computing unary factors. Time elapsed = "
        << myUtils::Timer::toc(start) << "s.\n" << std::endl;
  }
}

// generate multiple unary factors based on probability
void SegmentationFrontEnd::getUnaryFactorsFromBonnet(const cv::Mat& bonnetLabel,
    const cv::Mat& bonnetProb, size_t dstCols, size_t dstRows, int verbose) {
  if (crf_.nrNodes == 0) {
    std::cout << "0 superpixel node" << std::endl;
    throw std::runtime_error("getUnaryFactorsFromBonnet "
        "(SegmentationFrontEnd): cannot find labels without super pixel "
        "nodes");
  }

  if(verbose > 1) {
    std::cout << "\n$$$$$$$$$ Computing Unary Factors from Bonnet $$$$$"
        "$$$$" << std::endl;
  }
  auto start = myUtils::Timer::tic();

  crf_.unaryFactors.clear(); // clear crf_.unaryFactors vector
  crf_.nrClasses = 20;       // bonnet has 20 classes for cityscape dataset

  cv::Mat inputLabel, inputProb;
  cv::resize(bonnetLabel, inputLabel, cv::Size(dstCols, dstRows), 0, 0,
      cv::INTER_NEAREST);
  cv::resize(bonnetProb, inputProb, cv::Size(dstCols, dstRows), 0, 0,
      cv::INTER_NEAREST);

  // for each sp, compute the number of pixels belonging to each class
  int node = 0;
  for(auto const& superPixel : superPixels_) {
    // store the number of pixels belonging to each class for a superpixel
    std::vector<float> weightedLabelCount(crf_.nrClasses, 0.0);
    auto sum = std::accumulate(weightedLabelCount.begin(),
        weightedLabelCount.end(), 0.0);
    for(auto const& pixel: superPixel.pixels) {
      for (int label = 0; label < crf_.nrClasses; ++label) {
        float prob = inputProb.at<cv::Vec<float, 20>>(pixel.first,
            pixel.second)[label];
        weightedLabelCount[label] += prob;
      }
    }
    addUnaryFactorsFromLabelCount(node, 3, 0.15, weightedLabelCount);
    node++;
  }

  if(verbose > 1) {
    std::cout << crf_.unaryFactors.size() << " out of " << crf_.nrNodes
        << " superpixel nodes are classified." << std::endl;
  }

  if(verbose > 0) {
    std::cout << "Done computing unary factors. Time elapsed = "
        << myUtils::Timer::toc(start) << "s.\n" << std::endl;
  }
}


/* ------------------------------------------------------------------------- */
// read unary factors from label image
void SegmentationFrontEnd::getUnaryFactorsFromImage(
    const cv::Mat& labelImage, size_t pctClassified, size_t maxSamplingIter,
    bool constUnary, int verbose) {
  if (crf_.nrNodes == 0) {
    std::cout << "0 superpixel node" << std::endl;
    throw std::runtime_error("getUnaryFactorsFromImage (SegmentationFrontEnd): "
        "cannot find labels without super pixel nodes");
  }

  if(verbose > 1) {
    std::cout << "\n$$$$$$$$$ Computing Unary Factors from Ground Truth $$$$$"
        "$$$$" << std::endl;
  }
  auto start = myUtils::Timer::tic();
  crf_.unaryFactors.clear();                // clear crf_.unaryFactors vector

  // store the number of pixel in each class for each superpixel
  std::vector<std::vector<size_t>> totalLabelCount;

  getSPLabelsFromImage(labelImage, &totalLabelCount, &spLabels_,
      verbose);

  // find nodes to be classified
  size_t nrClassified = pctClassified*crf_.nrNodes/100;
  std::vector<int> indexVec;
  for(int i=0; i<crf_.nrNodes; ++i) {
    indexVec.push_back(i);
  }
  size_t nrSampling = 0;
  unsigned seed = 0;
  while(true) { // generate a list of labeled superpixels
    //      unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(indexVec.begin(), indexVec.end(),
        std::default_random_engine(seed));
    nrSampling++;

    // check if all classes are included
    std::vector<int> classFlag(crf_.nrClasses, 0);
    for(int i=0; i<nrClassified; ++i) {
      size_t foundLabel = spLabels_[indexVec[i]];
      classFlag[foundLabel] = 1;
    }
    if((find(classFlag.begin(), classFlag.end(), 0) == classFlag.end()) ||
        (nrSampling >= maxSamplingIter)) {
      break;
    } else {
      seed++;
    }
  }
  if(verbose > 2) {
    std::cout << "number of sampling attempts: " << nrSampling << std::endl;
  }

  // assign labels to classified nodes
  for(int i=0; i<nrClassified; ++i) {
    int node = indexVec[i];
    addUnaryFactorsFromLabelCount(constUnary, node, totalLabelCount[node]);
  }

  if(verbose > 1) {
    std::cout << crf_.unaryFactors.size() << " out of " << crf_.nrNodes
        << " superpixel nodes are classified." << std::endl;
  }

  if(verbose > 0) {
    std::cout << "Done computing unary factors. Time elapsed = "
        << myUtils::Timer::toc(start) << "s.\n" << std::endl;
  }
}

/* ------------------------------------------------------------------------- */
// helper function for getUnaryFactorsFromImage
// to count the number of pixel level labels each super pixel has and update
// crf_.nrClasses. Then it finds the ground-truth label for each superpixel.
void SegmentationFrontEnd::getSPLabelsFromImage(const cv::Mat& inputLabel,
    std::vector<std::vector<size_t>>* totalLabelCount,
    std::vector<size_t>* spLabels, int verbose) {
  // clear crf_.nrClasses, totalLabelCount, spLabels
  totalLabelCount->clear();
  spLabels->clear();
  crf_.nrClasses = 0;

  std::vector<cv::Vec3b> colorVec; // vector storing color value for each class
  for(auto const& superPixel : superPixels_) {
    // store the number of pixels belonging to each class for a superpixel
    std::vector<size_t> labelCount;
    for(size_t i=0; i<crf_.nrClasses; ++i) { // initialize labelCount
      labelCount.push_back(0);
    }
    for(auto const& pixel: superPixel.pixels) {
      cv::Vec3b color = inputLabel.at<cv::Vec3b>(pixel.first, pixel.second);
      size_t label = find(colorVec.begin(), colorVec.end(), color)
                - colorVec.begin();
      if (label >= crf_.nrClasses) {      // this class has not been found
        labelCount.push_back(1);
        colorVec.push_back(color);
        crf_.nrClasses += 1;
      } else {                            // this class has been found before
        labelCount[label] += 1;
      }
    }
    totalLabelCount->push_back(labelCount);// add labelCount to totalLabelCount
  }

  // find the ground truth label of each superpixel from pixel-level labels
  std::vector<size_t>::const_iterator it;
  size_t label;
  for(size_t i=0; i<crf_.nrNodes; ++i) {
    it = std::max_element(totalLabelCount->at(i).begin(),
        totalLabelCount->at(i).end());
    label = it - totalLabelCount->at(i).begin();
    spLabels->push_back(label);
  }

  if(verbose > 1) {
    std::cout << "crf_.nrClasses = " << crf_.nrClasses << std::endl;
  }

  if(verbose > 2) {
    cv::imshow("Image label", inputLabel);
    cv::waitKey(0);
  }
}

/* ------------------------------------------------------------------------- */
// helper function to add unary factors to crf_
// if constUnary is true, this function will use a *majority-takes-all*
// strategy such that one unary factor with weight 1 is created for the
// specified node;
// otherwise, if the dominant label in the given node has weight less than
// 0.8, multiple unary factors with weight greater than 0.2 will be added.
void SegmentationFrontEnd::addUnaryFactorsFromLabelCount(bool constUnary,
    int node, const std::vector<size_t>& labelCount) {
  // find most likely label for specified node
  std::vector<size_t>::const_iterator it =
      std::max_element(labelCount.begin(), labelCount.end());
  int label = it - labelCount.begin();

  // unary factor has constant weight 1
  if(constUnary) {
    UnaryFactor unaryFactor;
    unaryFactor.node = node;
    unaryFactor.label = label;
    unaryFactor.weight = 1;
    crf_.unaryFactors.push_back(unaryFactor);
  } else { // unary factor has weight based on percent pixels
    size_t sumPixels = std::accumulate(labelCount.begin(), labelCount.end(),
        0);
    double weight = (double)labelCount[label] / sumPixels;

    if(weight > 0.8) {  // one unary term is sufficient if weight is large
      UnaryFactor unaryFactor;
      unaryFactor.node = node;
      unaryFactor.label = label;
      unaryFactor.weight = weight;
      crf_.unaryFactors.push_back(unaryFactor);
    }
    else {            // add multiple unary terms
      for(int j=0; j<labelCount.size(); ++j) {
        weight = (double)labelCount[j] / sumPixels;
        if(weight > 0.2) {  // cutoff weight = 0.2
          UnaryFactor unaryFactor;
          unaryFactor.node = node;
          unaryFactor.label = j;
          unaryFactor.weight = weight;
          crf_.unaryFactors.push_back(unaryFactor);
        }
      }
    }

  }
}

// The weightedLabelCount stores a vector of weights for each label, this
// function adds the first k most probable labels of the specified node
// to crf_
// (less than k unary factors will be added if the weight is too small, but
// at least one will be added to crf_)
void SegmentationFrontEnd::addUnaryFactorsFromLabelCount(int node, size_t k,
    double minPerc, const std::vector<float>& weightedLabelCount) {
  std::priority_queue<std::pair<float, int>> q;
  for (int i = 0; i < weightedLabelCount.size(); ++i) {
    q.push(std::pair<float, int>(weightedLabelCount[i], i));
  }
  // find k largest weights
  int nrPixels = superPixels_[node].nrPixels;
  std::vector<std::pair<float, int>> q_k; // sorted vector
  float totalWeight = 0.0;
  for (size_t i = 0; i < k; ++i) {
    auto element = q.top();
    if (element.first > minPerc * nrPixels) { // min weight threshold
      q_k.push_back(element);
      totalWeight += element.first;
    }
    q.pop();
  }
  for (auto& element : q_k) {
    UnaryFactor unaryFactor;
    unaryFactor.node = node;
    unaryFactor.label = element.second;
    unaryFactor.weight = static_cast<double>(element.first/totalWeight);
    crf_.unaryFactors.push_back(unaryFactor);
  }
}

/* ---------------------------------------------------------------------- */
// return reference to imagID
const int& SegmentationFrontEnd::getID() const {
  return imgId_;
}

/* ------------------------------------------------------------------------- */
// return reference to spLabels
const std::vector<size_t>& SegmentationFrontEnd::getSPLabels() const {
  return spLabels_;
}

/* ------------------------------------------------------------------------- */
// return reference to crf_
const CRF& SegmentationFrontEnd::getCRF() const {
  return crf_;
}

/* ---------------------------------------------------------------------- */
// return reference to superPixels_
const std::vector<SuperPixel>& SegmentationFrontEnd::getSP() const {
  return superPixels_;
}

/* ------------------------------------------------------------------------- */
// display all params
void SegmentationFrontEnd::print() const {
  std::cout << "$$$$$$$$$ Image ID: " << imgId_ << " $$$$$$$$$" << std::endl;

  std::cout << "\n$$$$$$$$$ Printing Superpixels $$$$$$$$$" << std::endl;
  int count = 0;
  for(auto& superPixel : superPixels_) {
    std::cout << "Superpixel " << count << ": "
        << "center = (" << superPixel.center.first << ", "
        << superPixel.center.second << "), "
        << "average color = (" << superPixel.avgColor.val[0] << ", "
        << superPixel.avgColor.val[1] << ", " << superPixel.avgColor.val[2]
        << "), "
        << "nrPixels = " << superPixel.nrPixels << std::endl;
    ++count;
  }

  crf_.print();
}

/* ------------------------------------------------------------------------- */
// display image with cluster mean
void SegmentationFrontEnd::displayWithClusterMeans(const cv::Mat& inputImage)
    const {
  cv::Mat result(inputImage);
  for(auto const& SP : superPixels_) {
    for(auto const& pixel : SP.pixels) {
      result.at<cv::Vec3b>(pixel.first, pixel.second)[0] = SP.avgColor.val[0];
      result.at<cv::Vec3b>(pixel.first, pixel.second)[1] = SP.avgColor.val[1];
      result.at<cv::Vec3b>(pixel.first, pixel.second)[2] = SP.avgColor.val[2];
    }
  }
  cv::imshow("Clusters", result);
  cv::waitKey(0);
}

/* ------------------------------------------------------------------------- */
// assert equality up to a tolerance
bool SegmentationFrontEnd::equals(const SegmentationFrontEnd& sfe, double tol)
    const {
  // check imgId
  if (imgId_ != sfe.imgId_) {
    return false;
  }

  // check crf
  if(!crf_.equals(sfe.crf_, tol)) {
    return false;
  }

  // check superpixels (check each superpixel's avgColor, center and nrPixels)
  if(superPixels_.size() != sfe.superPixels_.size()) {
    return false;
  } else {
    for(size_t i=0; i<superPixels_.size(); ++i) {
      if(abs(superPixels_[i].nrPixels - sfe.superPixels_[i].nrPixels) > 100){
        std::cout << "Superpixel " << i << " number of superpixels not the "
            "same (" << superPixels_[i].nrPixels << " != "
            << sfe.superPixels_[i].nrPixels << ")" << std::endl;
        return false;
      }
      if(norm(superPixels_[i].avgColor - sfe.superPixels_[i].avgColor) > 1) {
        std::cout << "Superpixel " << i << " average color not the same "
            "(Euclidean distance = "
            << norm(superPixels_[i].avgColor - sfe.superPixels_[i].avgColor)
            << ")" << std::endl;
        return false;
      }
      if(fabs(superPixels_[i].center.first -
          sfe.superPixels_[i].center.first) > 1 ||
          fabs(superPixels_[i].center.second -
              sfe.superPixels_[i].center.second) > 1) {
        std::cout << "Superpixel " << i << " center not the same (rowDiff = "
            << fabs(superPixels_[i].center.first -
                sfe.superPixels_[i].center.first)
                << ", colDiff = "
                << fabs(superPixels_[i].center.second -
                    sfe.superPixels_[i].center.second)
                    << ")" << std::endl;
        return false;
      }
    }
  }

  return true;
}

} // namespace CRFsegmentation
