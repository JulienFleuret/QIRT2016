
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


#ifndef REGIONS_H
#define REGIONS_H

#include <opencv2/core.hpp>

// This code is the implementation of the article :
//@article{ding2009efficient,
//  title={An efficient image segmentation technique by fast scanning and adaptive merging},
//  author={Ding, Jian-Jiun and Kuo, CJ and Hong, WC},
//  journal={CVGIP, Aug},
//  year={2009}
//}
//
// _src : cv::Mat : matrix of any supported type with up to channels channels.
//
// _dst : cv::Mat : one channel matrix of 32 bits per elements signed integers.
//
// _stats : cv::Mat : one channel matrix of 32 bits per elements signed integers.
//
// _pts : std::vector<cv::Mat2i>, std::vector<std::vector<cv::Vec2i> > : vector containing the 2D coordinates of the points associate to each cluster.
//
// thresh : cv::Scalar : minimum threshold on each channels for create a new cluster.
//
// delta : int : minimum number of points for a cluster to exist.
//
//
// Additionnal informations :
//
// _dst has the same size as the source image an contain for each pixel the cluster associate. Note the cluster index start at 1.
//
// _stats have the same structure as the one of the function cv::connectedComponentsWithStats i.e. _stats if a matrix of size : number of cluster x 5.
// For each row the first four columns represents the x, y, height and width of the rectangular bounding box contaning cluster, the last column is total area convered by the cluster (not the bounding box).
//

namespace regions
{
void fast_scan(cv::InputArray _src, cv::OutputArray _dst,cv::OutputArray _stats,cv::OutputArrayOfArrays _pts, const cv::Scalar &thresh = cv::Scalar::all(25.), const int &delta = 100);
void fast_scan(cv::InputArray _src,cv::OutputArray _dst,const cv::Scalar& thresh = cv::Scalar::all(25.),const int& delta = 100);
}


#endif // REGIONS_H
