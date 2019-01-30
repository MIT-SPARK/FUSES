/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   IDToLabelConst.h
 * @brief  store label array to map cityscapes grayscale labelID to bonnet class
 * @author Siyi Hu, Luca Carlone
 */

#ifndef IDToLabelConst_H_
#define IDToLabelConst_H_
#include <stdlib.h>

//labelVec[i] = bonnet class of pixel having gray scale value (uint8_t) i
const std::vector<size_t> labelVec = {19, 19, 19, 19, 19, 19, 19, 0, 1, 19,
    19, 2, 3, 4, 19, 19, 19, 5, 19, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 19, 19,
    16, 17, 18};

#endif /* IDToLabelConst_H_ */


