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


