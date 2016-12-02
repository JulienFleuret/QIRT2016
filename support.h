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


#ifndef SUPPORT_H
#define SUPPORT_H
#include <opencv2/core.hpp>

#include "policy.h"

#include <chrono>

#ifdef _DEBUG
#include <iostream>
#endif

namespace support
{

enum
{
    BLOCK_SIZE=1024
};

void show(const cv::String& name,cv::InputArray img);

// UGLY

void show(const cv::String& str,const cv::Mat& in,const double& min,const double& max);

inline void show(const cv::String& str,const cv::Mat& in,const cv::Vec2d& min_max)
{
    show(str,in,*min_max.val,*(min_max.val+1));
}

//

template<class _Ty>
std::intmax_t get_range();

template<int flag>
inline std::intmax_t get_range_from_flag(){ return get_range<typename policy::flag2type<flag>::value_type>();}


enum
{
    SWAP_SRC,
    AS_IS=2
};

void morphological_hole_filling(cv::InputArray _src, cv::OutputArray _dst, cv::InputArray struct_elem = cv::noArray(), const int &ivt = AS_IS);

namespace hal
{

void morphological_hole_filling_8u(const cv::Mat1b& _src,const cv::Mat1b& _dst,const cv::Mat1b& struct_elem,const int& ivt);
void morphological_hole_filling_8s(const cv::Mat_<schar>& _src,const cv::Mat_<schar>& _dst,const cv::Mat1b& struct_elem,const int& ivt);

void morphological_hole_filling_16u(const cv::Mat1w& _src,const cv::Mat1w& _dst,const cv::Mat1b& struct_elem,const int& ivt);
void morphological_hole_filling_16s(const cv::Mat1s& _src,const cv::Mat1s& _dst,const cv::Mat1b& struct_elem,const int& ivt);

void morphological_hole_filling_32s(const cv::Mat1i& _src,const cv::Mat1i& _dst,const cv::Mat1b& struct_elem,const int& ivt);
void morphological_hole_filling_32f(const cv::Mat1f& _src,const cv::Mat1f& _dst,const cv::Mat1b& struct_elem,const int& ivt);

void morphological_hole_filling_64f(const cv::Mat1d& _src,const cv::Mat1d& _dst,const cv::Mat1b& struct_elem,const int& ivt);

}

void hole_filling(cv::InputArray _src, cv::OutputArray _dst, const int &_area = -1);



void getColours(cv::InputArray _src,cv::OutputArray _dst,cv::OutputArray _map = cv::noArray());

void apply_random_colours(cv::InputArray _src,cv::OutputArray _dst);


class timer_t
{
private:

    std::chrono::steady_clock::time_point _start;
    std::chrono::steady_clock::time_point _stop;

    template<class _Ty>
    std::string unit(){ return std::string();}



public:

    typedef std::chrono::steady_clock::time_point time_point;

    timer_t() = default;

    timer_t(const timer_t&) = delete;
    timer_t(timer_t&&) = delete;

    ~timer_t() = default;

    timer_t& operator=(const timer_t&) = delete;
    timer_t& operator=(timer_t&&) = delete;

    inline time_point start()
    {
        this->_start = std::chrono::steady_clock::now();

        return this->_start;
    }

    inline time_point stop()
    {
        this->_stop = std::chrono::steady_clock::now();

        return this->_stop;
    }

    inline std::uintmax_t diff_raw()const
    {
        return (this->_stop - this->_start).count();
    }

    template<class _Ty>
    inline std::uintmax_t diff()
    {
        return std::chrono::duration_cast<_Ty>(this->_stop-this->_start).count();
    }

    inline std::uintmax_t diff_ns()
    {
        return this->diff<std::chrono::nanoseconds>();
    }


    inline std::uintmax_t diff_us()
    {
        return this->diff<std::chrono::microseconds>();
    }


    inline std::uintmax_t diff_ms()
    {
        return this->diff<std::chrono::milliseconds>();
    }


    inline std::uintmax_t diff_sec()
    {
        return this->diff<std::chrono::seconds>();
    }


    inline std::uintmax_t diff_mn()
    {
        return this->diff<std::chrono::minutes>();
    }


    inline std::uintmax_t diff_hr()
    {
        return this->diff<std::chrono::hours>();
    }

    template<class _Ty,class _Cty>
    inline void print(std::basic_ostream<_Cty>& ostr)
    {
        ostr<<"the time difference is : "<<this->diff<_Ty>()<<" "<<this->unit<_Ty>()<<std::endl;
    }

    template<class _Cty>
    inline void print_ns(std::basic_ostream<_Cty>& ostr)
    {
        this->print<std::chrono::nanoseconds>(ostr);
    }

    template<class _Cty>
    inline void print_us(std::basic_ostream<_Cty>& ostr)
    {
        this->print<std::chrono::microseconds>(ostr);
    }

    template<class _Cty>
    inline void print_ms(std::basic_ostream<_Cty>& ostr)
    {
        this->print<std::chrono::milliseconds>(ostr);
    }

    template<class _Cty>
    inline void print_sec(std::basic_ostream<_Cty>& ostr)
    {
        this->print<std::chrono::seconds>(ostr);
    }

    template<class _Cty>
    inline void print_mn(std::basic_ostream<_Cty>& ostr)
    {
        this->print<std::chrono::minutes>(ostr);
    }

    template<class _Cty>
    inline void print_hr(std::basic_ostream<_Cty>& ostr)
    {
        this->print<std::chrono::hours>(ostr);
    }

};

template<>
std::string timer_t::unit<std::chrono::nanoseconds>();

template<>
std::string timer_t::unit<std::chrono::microseconds>();

template<>
std::string timer_t::unit<std::chrono::milliseconds>();

template<>
std::string timer_t::unit<std::chrono::seconds>();

template<>
std::string timer_t::unit<std::chrono::minutes>();

template<>
std::string timer_t::unit<std::chrono::hours>();


namespace
{
timer_t timer;
}


}

#endif

