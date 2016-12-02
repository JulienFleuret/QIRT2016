// Opencv's licence
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
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


/*
 * Copyright (c) 2016. Julien Fleuret <julien[dot]fleuret[at]ulaval[dot]ca>.
 * Released to public domain under terms of the BSD Simplified license.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of the organization nor the names of its contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 *
 *   See <http://www.opensource.org/licenses/bsd-license>
 */


#include "autothresh.h"

#include <opencv2/core/utility.hpp>

#include <mutex>
#include <thread>
#include <atomic>

//#include "private_otsu.h"



#include "support.h"

namespace autothresh
{

namespace
{

template<class _Sty,class _Rty>
double getThreshVal_Triangle_( const cv::InputArray& _src )
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



    std::int64_t i(0);
    std::int64_t j(0);

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

    for( i = 0; i < (std::int64_t)size.height; i++,src+=step )
    {
        j = 0;
#ifdef __x86_64__
#if CV_ENABLE_UNROLLED
        for( ; j <= (std::int64_t)size.width - 8; j += 8 )
        {
            _Sty v0 = src[j];
            _Sty v1 = src[j+1];

            h[(std::int64_t)v0]++;
            h[(std::int64_t)v1]++;

            v0 = src[j+2];
            v1 = src[j+3];

            h[(std::int64_t)v0]++;
            h[(std::int64_t)v1]++;

            v0 = src[j+4];
            v1 = src[j+5];

            h[(std::int64_t)v0]++;
            h[(std::int64_t)v1]++;

            v0 = src[j+6];
            v1 = src[j+7];

            h[(std::int64_t)v0]++;
            h[(std::int64_t)v1]++;
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
        for( ; j < (std::int64_t)size.width; j++ )
            h[(std::int64_t)src[j]]++;
    }

    std::int64_t left_bound = 0;
    std::int64_t right_bound = 0;
    std::int64_t max_ind = 0;
    std::int64_t max = 0;
    std::int64_t temp(0);

    bool isflipped = false;

    for( i = 0; i < (std::int64_t)N; i++ )
    {
        if( h[i] > 0 )
        {
            left_bound = i;
            break;
        }
    }
    if( left_bound > 0 )
        left_bound--;

    for( i = N-1; i > 0; i-- )
    {
        if( h[i] > 0 )
        {
            right_bound = i;
            break;
        }
    }
    if( right_bound < (int)(N-1) )
        right_bound++;

    for( i = 0; i < (std::int64_t)N; i++ )
    {
        if( h[i] > max)
        {
            max = h[i];
            max_ind = i;
        }
    }

    if( max_ind-left_bound < right_bound-max_ind)
    {
        isflipped = true;
        i = 0, j = N-1;
        while( i < j )
        {
            temp = h[i]; h[i] = h[j]; h[j] = temp;
            i++; j--;
        }
        left_bound = N-1-right_bound;
        max_ind = N-1-max_ind;
    }

    double thresh = left_bound;
    double a, b, dist = 0, tempdist;

    /*
     * We do not need to compute precise distance here. Distance is maximized, so some constants can
     * be omitted. This speeds up a computation a bit.
     */
    a = max;
    b = left_bound-max_ind;

    for( i = left_bound+1; i <= max_ind; i++ )
    {
        tempdist = a*i + b*h[i];
        if( tempdist > dist)
        {
            dist = tempdist;
            thresh = i;
        }
    }
    thresh--;

    if( isflipped )
        thresh = (std::int64_t)(N-1-thresh);

    return thresh;
}
}

// CV_32S is not manage because it's range (MAX_UINT+1) is to large.
double triangle(cv::InputArray in, const int &dtype)
{
    // type alloweded CV_8U, CV_8S, CV_16U, CV_16S, the types CV_32F and CV_64F are support only if their values are in a range covered by the previous types.
    CV_DbgAssert((in.type() == in.depth()) && ((in.depth() < CV_32S) || ((in.depth()>= CV_32S) && (dtype < CV_32S) ) ) );

    const int input_depth = in.depth() < CV_32S ? in.depth() : CV_MAT_DEPTH(dtype);

    typedef double (*function_type)(cv::InputArray&);


    static const function_type funcs[7][4] =
    {
        {getThreshVal_Triangle_<uchar,uchar>, nullptr, nullptr, nullptr},
        {nullptr, getThreshVal_Triangle_<schar,schar>, nullptr, nullptr},
        {nullptr, nullptr, getThreshVal_Triangle_<ushort,ushort>, nullptr},
        {nullptr, nullptr, nullptr, getThreshVal_Triangle_<short,short>},
        {getThreshVal_Triangle_<int,uchar>, getThreshVal_Triangle_<int,schar>, getThreshVal_Triangle_<int,ushort>, getThreshVal_Triangle_<int,short>},
        {getThreshVal_Triangle_<float,uchar>, getThreshVal_Triangle_<float,schar>, getThreshVal_Triangle_<float,ushort>, getThreshVal_Triangle_<float,short>},
        {getThreshVal_Triangle_<double,uchar>, getThreshVal_Triangle_<double,schar>, getThreshVal_Triangle_<double,ushort>, getThreshVal_Triangle_<double,short>},
    };

    function_type fun = funcs[in.depth()][input_depth];

    return fun(in);

}

}
