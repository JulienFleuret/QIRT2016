/*M///////////////////////////////////////////////////////////////////////////////////////
//
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//
// Copyright (C) 2016, Laboratory of computer vision and numerical systems (LVSN), Multipolar Infrared Vision team, University Laval, all rights reserved.
// @Authors
//    Julien FLEURET, julien.fleuret.1@ulaval.ca
//
// This is the code related to the paper : "A Real Time Animal Detection And Segmentation Algorithm For IRT Images In Indoor Environments".
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/


// N O T E | N O T E | N O T E | N O T E | N O T E | N O T E | N O T E | N O T E | N O T E | N O T E | N O T E | N O T E | N O T E | N O T E | N O T E | N O T E | N O T E | N O T E
// N O T E |                                                                                                                                                               | N O T E
// N O T E | The implementations and other codes contains in this file are deeply inspired by OpenCV's and OpenCP's libraries.                                             | N O T E
// N O T E |                                                                                                                                                               | N O T E
// N O T E | All or part of it might be a direct modification of an implementation of either OpenCV's or OpenCP's implementation for the purpose of an algorithm.          | N O T E
// N O T E |                                                                                                                                                               | N O T E
// N O T E | N O T E | N O T E | N O T E | N O T E | N O T E | N O T E | N O T E | N O T E | N O T E | N O T E | N O T E | N O T E | N O T E | N O T E | N O T E | N O T E | N O T E


#ifndef SEGMENTATION
#define SEGMENTATION

#include <opencv2/core.hpp>

namespace segmentation
{

// This function clusterize the input image in three clusters (background, foreground and false detection).
//
// _src : cv::Mat : one channel matrix of any type.
//
// _background : cv::Mat : one channel matrix of the same type and size as the input one.
//
// _foreground : cv::Mat : one channel matrix of the same type and size as the input one.
//
// _fd : cv::Mat : one channel matrix of the type type and size as the input one.
//
// _contours : cv::Mat : one channel matrix of the same type and size as the input one.
//
// _map : cv::Mat : a three channel matrix where each channel represent the pixel associated with each cluster.
//
//
// Note :
//
// _background return a matrix where the pixel that have been associated with the background are set to 0 the others keep the original intensity.
//
// _foreground : same as background for the pixels associated with the foreground.
//
// _fd : same as _foreground and _background for the pixels remaining.
//
// _contour is a side discovery however the output is not always interresting neither a properly contour. USE WITH CAUTION.
//
//
// Real time.
cv::Vec2d get_structure_of_interest(cv::InputArray _src, cv::OutputArray _background, cv::OutputArray _foreground,cv::OutputArray _fd, cv::OutputArray _contours,cv::OutputArray _map);

// This function call the previous function and then refine the result using a spatial clustering.
//
// _src : cv::Mat : one channel matrix of any type.
//
// _background : cv::Mat : one channel matrix of the same type and size as the input one.
//
// _foreground : cv::Mat : one channel matrix of the same type and size as the input one.
//
// _fd : cv::Mat : one channel matrix of the type type and size as the input one.
//
// _contours : cv::Mat : one channel matrix of the same type and size as the input one.
//
// _map : cv::Mat : a three channel matrix where each channel represent the pixel associated with each cluster.
//
//
// Not real time at all.
cv::Vec2d get_structure_of_interest_refined(cv::InputArray _src, cv::OutputArray _background, cv::OutputArray _foreground,cv::OutputArray _fd, cv::OutputArray _contours,cv::OutputArray _map);


// This function will refine the _map argument removing the minor regions what may exist and then reprocess the foreground, background and false positive image.
//
// _src : cv::Mat : one channel matrix of any type.
//
// _backgroudn : cv::Mat : one channel matrix of unsigned integer either 8 bits or 32 bits per elements.
//
// _background : cv::Mat : one channel matrix of the same type and size as the input one.
//
// _foreground : cv::Mat : one channel matrix of the same type and size as the input one.
//
// false_detection : cv::Mat : one channel matrix of the type type and size as the input one.
//
// _new_map : cv::Mat : a three channel matrix where each channel represent the pixel associated with each cluster.
//
void smooth(cv::InputArray _src,cv::InputArray _map,cv::OutputArray _foreground,cv::OutputArray _background,cv::OutputArray _false_detection,cv::OutputArray _new_map);

}

#endif // SEGMENTATION

