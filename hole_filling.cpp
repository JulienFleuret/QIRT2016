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


#include "support.h"

#include <opencv2/imgproc.hpp>

#include <mutex>

#include <memory>

#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>



namespace support
{



void hole_filling(cv::InputArray _src,cv::OutputArray _dst, const int &_area)
{
    CV_DbgAssert((_src.type()==CV_8UC1) && (!_dst.fixedType() || (_dst.fixedType() && (_dst.type() == CV_8UC1) )) );

    if((_area == 0) && (_dst.empty() || (_src.size() != _dst.size()) || (_dst.type() != CV_8UC1)))
    {
        _dst.create(_src.size(),CV_8UC1);
        _dst.setTo(cv::Scalar::all(0.));
    }

    cv::Mat1b src = _src.getMat();
    cv::Mat1b dst;

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::Scalar color(0xFF);


    if(_area == 0)
    {
        dst = _dst.getMat();


       cv::findContours(src,contours,hierarchy,cv::RETR_CCOMP,cv::CHAIN_APPROX_SIMPLE);


       for(std::size_t i=0;i<contours.size();i++)
           cv::drawContours(dst,contours,i,color,-1,8,hierarchy,0);
    }
    else
    {


        const int area = -1 ? _area : std::floor(static_cast<float>(src.total())*0.05f);


        cv::Mat1b tmp(src.size(),uchar(0));
        const int upper_bound = src.total() - (src.total()*0.1f);


        cv::findContours(src,contours,hierarchy,cv::RETR_CCOMP | cv::RETR_EXTERNAL,cv::CHAIN_APPROX_TC89_KCOS);

             tmp=cv::Mat::zeros(src.size(),src.type());

                 for(std::size_t i=0;i<contours.size();i++)
                 {

                     cv::Rect roi = cv::boundingRect(contours.at(i));

                     if( (roi.area() > area) && (roi.area() < upper_bound))
                     {

                         cv::drawContours(tmp,contours,i,color,cv::FILLED,cv::LINE_AA,hierarchy,0);

                         cv::Mat1b roi_s = src(roi);
                         cv::Mat1b roi_d = tmp(roi);

#if CV_ENABLE_UNROLLED
                         for(int r=0,c=0;r<roi_s.rows;r++,c=0)
                         {
                             const uchar* s = roi_s[r];
                             uchar* d = roi_d[r];

                             for(c=0;c<roi_s.cols-4;c+=4)
                             {
                                 uchar p1 = *(s+c);
                                 uchar p2 = *(s+c+1);

                                 if(p1==0)
                                     *(d+c) = 0;

                                 if(p2==0)
                                     *(d+c+1) = 0;

                                 p1 = *(s+c+2);
                                 p2 = *(s+c+3);

                                 if(p1==0)
                                     *(d+c+2) = 0;

                                 if(p2==0)
                                     *(d+c+3) = 0;
                             }
                             for(;c<roi_s.cols;c++)
                                 if(*(s+c)==0)
                                     *(d+c) = 0;
                         }
#else
                         for(int r=0,c=0;r<roi_s.rows;r++,c=0)
                             for(int c=0;c<roi_s.cols;c++)
                                 if(roi_s(r,c)==0)
                                     roi_d(r,c) = 0;
#endif

                     }
                     else
                         cv::drawContours(tmp,contours,i,color,cv::FILLED,cv::LINE_8,hierarchy,0);
                 }

                 dst = tmp;

    }

    dst.copyTo(_dst);

}

}

