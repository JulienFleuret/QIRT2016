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

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui.hpp>

namespace support
{

namespace
{

void rescale(cv::Mat& img)
{
    double mn(0.);
    double mx(0.);

    cv::minMaxLoc(img,&mn,&mx);

    mx-=mn;

    if(img.depth() < CV_32F)
        img.convertTo(img,CV_32F);

    img = (img-mn)/mx;
    img *= 255.;

    img.convertTo(img,CV_8U);
}

void show_(const cv::String& name,cv::Mat& img)
{
    if(img.depth() != CV_8U)
        rescale(img);

    cv::imshow(name,img);
}


}

void show(const cv::String& name,cv::InputArray img)
{
    CV_Assert( img.isMat() || img.isUMat() || img.isMatVector() || img.isUMatVector()
           #ifdef HAVE_CUDA
               || img.isGpuMatVector() || img.kind() == cv::_InputArray::CUDA_GPU_MAT
           #endif
               );

    if(img.empty())
        return;

    switch(img.kind())
    {

    case cv::_InputArray::MAT :
    {
        cv::Mat tmp = img.getMat().clone();

        show_(name,tmp);
    }
        break;
    case cv::_InputArray::STD_VECTOR_MAT:
    {
        std::vector<cv::String> names;
        std::vector<cv::Mat> images;
        cv::Mat tmp;

        img.getMatVector(images);

        if(!images.empty())
        {
            names.resize(images.size(),name);

            std::size_t i=0;

            typename std::vector<cv::Mat>::iterator it_img = images.begin();
            for(typename std::vector<cv::String>::iterator it = names.begin();it != names.end();it++,i++,it_img++)
            {
                *it += std::to_string(i);

                it_img->copyTo(tmp);

                show_(*it,*it_img);
            }
        }
    }
        break;

    case cv::_InputArray::UMAT:
    {
        cv::Mat tmp;

        img.getUMat().copyTo(tmp);

        show_(name,tmp);
    }
        break;

    case cv::_InputArray::STD_VECTOR_UMAT:
    {
        std::vector<cv::String> names;
        std::vector<cv::UMat> images;
        cv::Mat tmp;

        img.getUMatVector(images);

        if(!images.empty())
        {
            names.resize(images.size(),name);

            std::size_t i=0;

            typename std::vector<cv::UMat>::const_iterator it_img = images.begin();
            for(typename std::vector<cv::String>::iterator it = names.begin();it != names.end();it++,i++,it_img++)
            {
                *it += std::to_string(i);

                it_img->copyTo(tmp);

                show_(*it,tmp);
            }
        }
    }
        break;

#ifdef HAVE_CUDA

    case cv::_InputArray::CUDA_GPU_MAT:
    {
        cv::Mat tmp;

        img.getGpuMat().download(tmp);

        show_(name,tmp);
    }
        break;

    case cv::_InputArray::STD_VECTOR_CUDA_GPU_MAT:
    {
        std::vector<cv::String> names;
        std::vector<cv::cuda::GpuMat> images;
        cv::Mat tmp;

        img.getGpuMatVector(images);

        if(!images.empty())
        {
            names.resize(images.size(),name);

            std::size_t i=0;

            typename std::vector<cv::cuda::GpuMat>::const_iterator it_img = images.begin();
            for(typename std::vector<cv::String>::iterator it = names.begin();it != names.end();it++,i++,it_img++)
            {
                *it += std::to_string(i);

                it_img->download(tmp);

                show_(*it,tmp);
            }
        }
    }
        break;

#endif

    }

}

void spec_rsc(cv::Mat& in,const double& _min = -1.,const double& _max = -1.)
{
    if(in.depth() != CV_8U)
    {
        double max(0.);
        double min(0.);

        if( (_min == -1.) || (_max == -1.))
        {
            cv::UMat utmp = in.getUMat(cv::ACCESS_READ,cv::USAGE_ALLOCATE_DEVICE_MEMORY);

            cv::minMaxIdx(utmp,&min,&max);

            if(_min != -1.)
                min = _min;

            if(_max != -1)
                max = _max;
            else
                max-=min;
        }
        else
        {
            max = _max;
            min = _min;
        }

        in.convertTo(in,CV_32F);

        in = ((in-min)/max) * 255.;

        in.convertTo(in,CV_8U);
    }

}

void show(const cv::String& str,const cv::Mat& in,const double& min,const double& max)
{
    cv::Mat tmp = in.clone();

    spec_rsc(tmp,min,max);


    if(!tmp.empty())
        cv::imshow(str,tmp);
}

template<>
std::intmax_t get_range<std::int8_t>(){ return static_cast<std::uintmax_t>(std::numeric_limits<std::uint8_t>::max())+1;}

template<>
std::intmax_t get_range<std::uint8_t>(){ return get_range<std::int8_t>();}

template<>
std::intmax_t get_range<std::int16_t>(){ return static_cast<std::uintmax_t>(std::numeric_limits<std::uint16_t>::max())+1;}

template<>
std::intmax_t get_range<std::uint16_t>(){ return get_range<std::int16_t>();}

template<>
std::intmax_t get_range<std::int32_t>(){ return static_cast<std::uintmax_t>(std::numeric_limits<std::uint32_t>::max())+1;}

template<>
std::intmax_t get_range<std::uint32_t>(){ return get_range<std::int32_t>();}

template<>
std::intmax_t get_range<float>(){ return get_range<std::uint32_t>();}

template<>
std::intmax_t get_range<double>(){ return get_range<std::uint32_t>();}






template<>
std::string timer_t::unit<std::chrono::nanoseconds>(){ return "ns";}

template<>
std::string timer_t::unit<std::chrono::microseconds>(){ return "us";}

template<>
std::string timer_t::unit<std::chrono::milliseconds>(){ return "ms";}

template<>
std::string timer_t::unit<std::chrono::seconds>(){ return "sec";}

template<>
std::string timer_t::unit<std::chrono::minutes>(){ return "mn";}

template<>
std::string timer_t::unit<std::chrono::hours>(){ return "hr";}


}


