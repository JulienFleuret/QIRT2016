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



#include "autothresh.h"

#include <opencv2/core/utility.hpp>

#include <mutex>
#include <thread>
#include <atomic>

//#include "private_otsu.h"

#ifdef _DEBUG
#include <iostream>
#endif

#include "support.h"

namespace autothresh
{

namespace
{


template<class _Sty,class _Rty>
double getThreshVal_Otsu_( const cv::InputArray& _src)
{
    const std::uintmax_t N = support::get_range<_Rty>();
    cv::Mat_<_Sty> img = _src.getMat();
    cv::Size size = _src.size();
    int step = (int) img.step1();
    if( _src.isContinuous() )
    {
        size.width *= size.height;
        size.height = 1;
        step = size.width;
    }



    std::size_t i(0);
    std::size_t j(0);

    std::vector<int> _h;

#if CV_AVX
    _h.reserve(cv::alignSize(N,32/sizeof(_Sty)));
#elif CV_SSE2
    _h.reserve(cv::alignSize(N,16/sizeof(_Sty)));
#else
    _h.reserve(N);
#endif
    _h.resize(N,0);

    int* h = _h.data();

    const _Sty* src = img[0];

    for( i = 0; i < (std::size_t)size.height; i++,src+=step )
    {
        j = 0;
#ifdef __x86_64__
#if CV_ENABLE_UNROLLED
        for( ; j <= (std::size_t)size.width - 8; j += 8 )
        {
            _Sty v0 = src[j];
            _Sty v1 = src[j+1];

            h[(std::size_t)v0]++;
            h[(std::size_t)v1]++;

            v0 = src[j+2];
            v1 = src[j+3];

            h[(std::size_t)v0]++;
            h[(std::size_t)v1]++;

            v0 = src[j+4];
            v1 = src[j+5];

            h[(std::size_t)v0]++;
            h[(std::size_t)v1]++;

            v0 = src[j+6];
            v1 = src[j+7];

            h[(std::size_t)v0]++;
            h[(std::size_t)v1]++;
        }
#endif
#else // if the architecture is not x86_64
        #if CV_ENABLE_UNROLLED
        for( ; j <= size.width - 4; j += 4 )
        {
            _Sty v0 = src[j];
            _Sty v1 = src[j+1];

            h[v0]++;
            h[v1]++;

            v0 = src[j+2];
            v1 = src[j+3];

            h[v0]++;
            h[v1]++;
        }
        #endif
#endif
        for( ; j < (std::size_t)size.width; j++ )
            h[(std::size_t)src[j]]++;
    }

    double mu = 0;
    double scale = 1./(size.area());

    for( i = 0; i < N; i++ )
        mu += i*(double)h[i];

    mu *= scale;
    double mu1 = 0, q1 = 0;
    double max_sigma = 0, max_val = 0;

    for( i = 0; i < N; i++ )
    {
        double p_i, q2, mu2, sigma;

        p_i = h[i]*scale;
        mu1 *= q1;
        q1 += p_i;
        q2 = 1. - q1;

        if( std::min(q1,q2) < std::numeric_limits<float>::epsilon() || std::max(q1,q2) > 1. - std::numeric_limits<float>::epsilon() )
            continue;

        mu1 = (mu1 + i*p_i)/q1;
        mu2 = (mu - q1*mu1)/q2;
        sigma = q1*q2*(mu1 - mu2)*(mu1 - mu2);
        if( sigma > max_sigma )
        {
            max_sigma = sigma;
            max_val = i;
        }
    }

    return max_val;
}

}

// CV_32S is not manage because it's range (MAX_UINT+1) is to large.
double otsu(cv::InputArray in, const int &dtype)
{

    // type alloweded CV_8U, CV_8S, CV_16U, CV_16S, the types CV_32F and CV_64F are support only if their values are in a range covered by the previous types.
    CV_DbgAssert((in.type() == in.depth()) && ((in.depth() < CV_32S) || ((in.depth()>= CV_32S) && (dtype < CV_32S) ) ) );

    const int input_depth = in.depth() < CV_32S ? in.depth() : CV_MAT_DEPTH(dtype);

    typedef double (*function_type)(cv::InputArray&);


    static const function_type funcs[7][4] =
    {
        {getThreshVal_Otsu_<uchar,uchar>, nullptr, nullptr, nullptr},
        {nullptr, getThreshVal_Otsu_<schar,schar>, nullptr, nullptr},
        {nullptr, nullptr, getThreshVal_Otsu_<ushort,ushort>, nullptr},
        {nullptr, nullptr, nullptr, getThreshVal_Otsu_<short,short>},
        {getThreshVal_Otsu_<int,uchar>, getThreshVal_Otsu_<int,schar>, getThreshVal_Otsu_<int,ushort>, getThreshVal_Otsu_<int,short>},
        {getThreshVal_Otsu_<float,uchar>, getThreshVal_Otsu_<float,schar>, getThreshVal_Otsu_<float,ushort>, getThreshVal_Otsu_<float,short>},
        {getThreshVal_Otsu_<double,uchar>, getThreshVal_Otsu_<double,schar>, getThreshVal_Otsu_<double,ushort>, getThreshVal_Otsu_<double,short>},
    };

    function_type fun = funcs[in.depth()][input_depth];

    return fun(in);

}

}
