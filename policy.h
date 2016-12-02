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


#ifndef POLICY
#define POLICY

#include <opencv2/core.hpp>

namespace policy
{

template<int flag>
struct flag2type
{
    typedef typename flag2type<CV_MAT_DEPTH(flag)>::value_type value_type;
    typedef cv::Vec<value_type,CV_MAT_CN(flag)> elem_type;

    typedef const value_type const_value_type;
    typedef const elem_type const_elem_type;

    typedef value_type* vt_pointer;
    typedef const vt_pointer const_vt_pointer;

    typedef elem_type* elem_pointer;
    typedef const elem_pointer const_elem_pointer;
};

template<>
struct flag2type<CV_8U>
{
    typedef uchar value_type;
    typedef uchar elem_type;

    typedef const uchar const_value_type;
    typedef const uchar const_elem_type;

    typedef uchar* vt_pointer;
    typedef const uchar* const_vt_pointer;

    typedef uchar* elem_pointer;
    typedef const uchar* const_elem_pointer;
};

template<>
struct flag2type<CV_8S>
{
    typedef schar value_type;
    typedef schar elem_type;

    typedef const schar const_value_type;
    typedef const schar const_elem_type;

    typedef schar* vt_pointer;
    typedef const schar* const_vt_pointer;

    typedef schar* elem_pointer;
    typedef const schar* const_elem_pointer;
};

template<>
struct flag2type<CV_16U>
{
    typedef ushort value_type;
    typedef ushort elem_type;

    typedef const ushort const_value_type;
    typedef const ushort const_elem_type;

    typedef ushort* vt_pointer;
    typedef const ushort* const_vt_pointer;

    typedef ushort* elem_pointer;
    typedef const ushort* const_elem_pointer;
};

template<>
struct flag2type<CV_16S>
{
    typedef short value_type;
    typedef short elem_type;

    typedef const short const_value_type;
    typedef const short const_elem_type;

    typedef short* vt_pointer;
    typedef const short* const_vt_pointer;

    typedef short* elem_pointer;
    typedef const short* const_elem_pointer;
};

template<>
struct flag2type<CV_32S>
{
    typedef int value_type;
    typedef int elem_type;

    typedef const int const_value_type;
    typedef const int const_elem_type;

    typedef int* vt_pointer;
    typedef const int* const_vt_pointer;

    typedef int* elem_pointer;
    typedef const int* const_elem_pointer;
};

template<>
struct flag2type<CV_32F>
{
    typedef float value_type;
    typedef float elem_type;

    typedef const float const_value_type;
    typedef const float const_elem_type;

    typedef float* vt_pointer;
    typedef const float* const_vt_pointer;

    typedef float* elem_pointer;
    typedef const float* const_elem_pointer;
};

template<>
struct flag2type<CV_64F>
{
    typedef double value_type;
    typedef double elem_type;

    typedef const double const_value_type;
    typedef const double const_elem_type;

    typedef double* vt_pointer;
    typedef const double* const_vt_pointer;

    typedef double* elem_pointer;
    typedef const double* const_elem_pointer;
};



template<>
struct flag2type<CV_8UC2>
{
    typedef uchar value_type;
    typedef cv::Vec2b elem_type;

    typedef const uchar const_value_type;
    typedef const cv::Vec2b const_elem_type;

    typedef uchar* vt_pointer;
    typedef const uchar* const_vt_pointer;

    typedef cv::Vec2b* elem_pointer;
    typedef const cv::Vec2b* const_elem_pointer;
};

template<>
struct flag2type<CV_8SC2>
{
    typedef schar value_type;
    typedef cv::Vec<schar,2> elem_type;

    typedef const schar const_value_type;
    typedef const cv::Vec<schar,2> const_elem_type;

    typedef schar* vt_pointer;
    typedef const schar* const_vt_pointer;

    typedef cv::Vec<schar,2>* elem_pointer;
    typedef const cv::Vec<schar,2>* const_elem_pointer;
};

template<>
struct flag2type<CV_16UC2>
{
    typedef ushort value_type;
    typedef cv::Vec2w elem_type;

    typedef const ushort const_value_type;
    typedef const cv::Vec2w const_elem_type;

    typedef ushort* vt_pointer;
    typedef const ushort* const_vt_pointer;

    typedef cv::Vec2w* elem_pointer;
    typedef const cv::Vec2w* const_elem_pointer;
};

template<>
struct flag2type<CV_16SC2>
{
    typedef short value_type;
    typedef cv::Vec2s elem_type;

    typedef const short const_value_type;
    typedef const cv::Vec2s const_elem_type;

    typedef short* vt_pointer;
    typedef const short* const_vt_pointer;

    typedef cv::Vec2s* elem_pointer;
    typedef const cv::Vec2s* const_elem_pointer;
};

template<>
struct flag2type<CV_32SC2>
{
    typedef int value_type;
    typedef cv::Vec2i elem_type;

    typedef const int const_value_type;
    typedef const cv::Vec2i const_elem_type;

    typedef int* vt_pointer;
    typedef const int* const_vt_pointer;

    typedef cv::Vec2i* elem_pointer;
    typedef const cv::Vec2i* const_elem_pointer;
};

template<>
struct flag2type<CV_32FC2>
{
    typedef float value_type;
    typedef cv::Vec2f elem_type;

    typedef const float const_value_type;
    typedef const cv::Vec2f const_elem_type;

    typedef float* vt_pointer;
    typedef const float* const_vt_pointer;

    typedef cv::Vec2f* elem_pointer;
    typedef const cv::Vec2f* const_elem_pointer;
};

template<>
struct flag2type<CV_64FC2>
{
    typedef double value_type;
    typedef cv::Vec2d elem_type;

    typedef const double const_value_type;
    typedef const cv::Vec2d const_elem_type;

    typedef double* vt_pointer;
    typedef const double* const_vt_pointer;

    typedef cv::Vec2d* elem_pointer;
    typedef const cv::Vec2d* const_elem_pointer;
};






template<>
struct flag2type<CV_8UC3>
{
    typedef uchar value_type;
    typedef cv::Vec3b elem_type;

    typedef const uchar const_value_type;
    typedef const cv::Vec3b const_elem_type;

    typedef uchar* vt_pointer;
    typedef const uchar* const_vt_pointer;

    typedef cv::Vec3b* elem_pointer;
    typedef const cv::Vec3b* const_elem_pointer;
};

template<>
struct flag2type<CV_8SC3>
{
    typedef schar value_type;
    typedef cv::Vec<schar,3> elem_type;

    typedef const schar const_value_type;
    typedef const cv::Vec<schar,3> const_elem_type;

    typedef schar* vt_pointer;
    typedef const schar* const_vt_pointer;

    typedef cv::Vec<schar,3>* elem_pointer;
    typedef const cv::Vec<schar,3>* const_elem_pointer;
};

template<>
struct flag2type<CV_16UC3>
{
    typedef ushort value_type;
    typedef cv::Vec3w elem_type;

    typedef const ushort const_value_type;
    typedef const cv::Vec3w const_elem_type;

    typedef ushort* vt_pointer;
    typedef const ushort* const_vt_pointer;

    typedef cv::Vec3w* elem_pointer;
    typedef const cv::Vec3w* const_elem_pointer;
};

template<>
struct flag2type<CV_16SC3>
{
    typedef short value_type;
    typedef cv::Vec3s elem_type;

    typedef const short const_value_type;
    typedef const cv::Vec3s const_elem_type;

    typedef short* vt_pointer;
    typedef const short* const_vt_pointer;

    typedef cv::Vec3s* elem_pointer;
    typedef const cv::Vec3s* const_elem_pointer;
};

template<>
struct flag2type<CV_32SC3>
{
    typedef int value_type;
    typedef cv::Vec3i elem_type;

    typedef const int const_value_type;
    typedef const cv::Vec3i const_elem_type;

    typedef int* vt_pointer;
    typedef const int* const_vt_pointer;

    typedef cv::Vec3i* elem_pointer;
    typedef const cv::Vec3i* const_elem_pointer;
};

template<>
struct flag2type<CV_32FC3>
{
    typedef float value_type;
    typedef cv::Vec3f elem_type;

    typedef const float const_value_type;
    typedef const cv::Vec3f const_elem_type;

    typedef float* vt_pointer;
    typedef const float* const_vt_pointer;

    typedef cv::Vec3f* elem_pointer;
    typedef const cv::Vec3f* const_elem_pointer;
};

template<>
struct flag2type<CV_64FC3>
{
    typedef double value_type;
    typedef cv::Vec3d elem_type;

    typedef const double const_value_type;
    typedef const cv::Vec3d const_elem_type;

    typedef double* vt_pointer;
    typedef const double* const_vt_pointer;

    typedef cv::Vec3d* elem_pointer;
    typedef const cv::Vec3d* const_elem_pointer;
};







template<>
struct flag2type<CV_8UC4>
{
    typedef uchar value_type;
    typedef cv::Vec4b elem_type;

    typedef const uchar const_value_type;
    typedef const cv::Vec4b const_elem_type;

    typedef uchar* vt_pointer;
    typedef const uchar* const_vt_pointer;

    typedef cv::Vec4b* elem_pointer;
    typedef const cv::Vec4b* const_elem_pointer;
};

template<>
struct flag2type<CV_8SC4>
{
    typedef schar value_type;
    typedef cv::Vec<schar,4> elem_type;

    typedef const schar const_value_type;
    typedef const cv::Vec<schar,4> const_elem_type;

    typedef schar* vt_pointer;
    typedef const schar* const_vt_pointer;

    typedef cv::Vec<schar,4>* elem_pointer;
    typedef const cv::Vec<schar,4>* const_elem_pointer;
};

template<>
struct flag2type<CV_16UC4>
{
    typedef ushort value_type;
    typedef cv::Vec4w elem_type;

    typedef const ushort const_value_type;
    typedef const cv::Vec4w const_elem_type;

    typedef ushort* vt_pointer;
    typedef const ushort* const_vt_pointer;

    typedef cv::Vec4w* elem_pointer;
    typedef const cv::Vec4w* const_elem_pointer;
};

template<>
struct flag2type<CV_16SC4>
{
    typedef short value_type;
    typedef cv::Vec4s elem_type;

    typedef const short const_value_type;
    typedef const cv::Vec4s const_elem_type;

    typedef short* vt_pointer;
    typedef const short* const_vt_pointer;

    typedef cv::Vec4s* elem_pointer;
    typedef const cv::Vec4s* const_elem_pointer;
};

template<>
struct flag2type<CV_32SC4>
{
    typedef int value_type;
    typedef cv::Vec4i elem_type;

    typedef const int const_value_type;
    typedef const cv::Vec4i const_elem_type;

    typedef int* vt_pointer;
    typedef const int* const_vt_pointer;

    typedef cv::Vec4i* elem_pointer;
    typedef const cv::Vec4i* const_elem_pointer;
};

template<>
struct flag2type<CV_32FC4>
{
    typedef float value_type;
    typedef cv::Vec4f elem_type;

    typedef const float const_value_type;
    typedef const cv::Vec4f const_elem_type;

    typedef float* vt_pointer;
    typedef const float* const_vt_pointer;

    typedef cv::Vec4f* elem_pointer;
    typedef const cv::Vec4f* const_elem_pointer;
};

template<>
struct flag2type<CV_64FC4>
{
    typedef double value_type;
    typedef cv::Vec4d elem_type;

    typedef const double const_value_type;
    typedef const cv::Vec4d const_elem_type;

    typedef double* vt_pointer;
    typedef const double* const_vt_pointer;

    typedef cv::Vec4d* elem_pointer;
    typedef const cv::Vec4d* const_elem_pointer;
};



}

#endif // POLICY

