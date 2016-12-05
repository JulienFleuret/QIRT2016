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


#include "segmentation.h"
#include "autothresh.h"
#include "regions.h"

#include "support.h"

#include <opencv2/core/utility.hpp>
#include <opencv2/core/hal/hal.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/hal/hal.hpp>

#include <memory>

#include <tbb/blocked_range2d.h>
//#include <tbb/parallel_reduce.h>

#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>

#include <iostream>

#include <opencv2/highgui.hpp>

namespace segmentation
{


namespace
{

template<class _Ty>
class body_estimate_ : public cv::ParallelLoopBody
{

public:

    typedef _Ty value_type;
    typedef _Ty* pointer;
    typedef const _Ty* const_pointer;

private:

    const_pointer _src;
    const int _src_step1;

    pointer _bd;
    const int _bd_step1;

    pointer _fd;
    const int _fd_step1;

    const int _cols;


 public:

    inline body_estimate_(cv::Mat& src,cv::Mat& bd,cv::Mat& fd):
        _src(src.ptr<_Ty>()),
        _src_step1(src.step1()),
        _bd(bd.ptr<_Ty>()),
        _bd_step1(bd.step1()),
        _fd(fd.ptr<_Ty>()),
        _fd_step1(fd.step1()),
        _cols(src.cols)
    {}

    ~body_estimate_() = default;

    void operator()(const cv::Range& range)const
    {
        const_pointer src = cv::alignPtr(this->_src + range.start * this->_src_step1,32);
        pointer fd = cv::alignPtr(this->_fd + range.start * this->_fd_step1,32);
        pointer bd = cv::alignPtr(this->_bd + range.start * this->_bd_step1,32);

        for(int r=range.start;r<range.end;r++,src+=this->_src_step1,bd+=this->_bd_step1,fd+=this->_fd_step1)
            for(int c=0;c<this->_cols;c++)
            {
                if(*(fd+c) != 0)
                {
                    *(fd+c) = *(src+c);
                    *(bd+c) = 0;
                }
                else
                    *(bd+c) = *(src+c);
            }
    }

};




template<class _Ty>
inline void estimate_background_foregrounds(cv::Mat& src,cv::Mat& bd,cv::Mat& fd)
{
    bd = cv::Mat::zeros(src.size(),src.type());

    body_estimate_<_Ty> body(src,bd,fd);

    cv::parallel_for_(cv::Range(0,src.rows),body,0x20);
}

template<class _Ty>
class body_bckg_map_ : public cv::ParallelLoopBody
{
public:

    typedef _Ty value_type;
    typedef _Ty* pointer;
    typedef const _Ty* const_pointer;

private:


        const_pointer _src;
        const int _src_step1;

        const_pointer _map;
        const int _map_step1;

        pointer _frgd;
        const int _frgd_step1;

        pointer _bckgd;
        const int _bckgd_step1;

        const int _cols;

public:

    inline body_bckg_map_(const cv::Mat& src,const cv::Mat& map,cv::Mat& frgd,cv::Mat& backgd):
        _src(src.ptr<_Ty>()),
        _src_step1(src.step1()),
        _map(map.ptr<_Ty>()),
        _map_step1(map.step1()),
        _frgd(frgd.ptr<_Ty>()),
        _frgd_step1(frgd.step1()),
        _bckgd(backgd.ptr<_Ty>()),
        _bckgd_step1(backgd.step1()),
        _cols(src.cols)
    {}


    virtual ~body_bckg_map_() = default;


    void operator()(const cv::Range& range)const
    {
        const_pointer src = cv::alignPtr(this->_src + range.start*this->_src_step1,32);
        const_pointer map = cv::alignPtr(this->_map + range.start*this->_map_step1,32);

        pointer frgd = cv::alignPtr(this->_frgd + range.start*this->_frgd_step1,32);
        pointer bckgd = cv::alignPtr(this->_bckgd + range.start*this->_bckgd_step1,32);

        for(std::size_t r = range.start;r<(std::size_t)range.end;r++,src+=this->_src_step1,map+=this->_map_step1,frgd+=this->_frgd_step1,bckgd+=this->_bckgd_step1)
            for(std::size_t c=0;c<(std::size_t)this->_cols;c++)
                if(*(frgd + c) != 0 && *(map + c) != 0)
                {
                    *(bckgd + c) = *(src+c);
                    *(frgd+c) = 0;
                }
    }

};


template<class _Ty>
inline void update_estimations(cv::Mat& src,cv::Mat& map,cv::Mat& frgd,cv::Mat& bckgd)
{
    const int grain = 0x20;
    body_bckg_map_<_Ty> body(src,map,frgd,bckgd);

    cv::parallel_for_(cv::Range(0,src.rows),body,grain);
}


template<class _Ty>
class body_bckg_map_ctrs_ : public cv::ParallelLoopBody
{
public:

    typedef _Ty value_type;
    typedef _Ty* pointer;
    typedef const _Ty* const_pointer;

private:

        const_pointer _src;
        const int _src_step1;

        const_pointer _map;
        const int _map_step1;

        pointer _frgd;
        const int _frgd_step1;

        pointer _bckgd;
        const int _bckgd_step1;

        pointer _ctrs;
        const int _ctrs_step1;

        const int _cols;

public:

    inline body_bckg_map_ctrs_(const cv::Mat& src,const cv::Mat& map,cv::Mat& frgd,cv::Mat& backgd,cv::Mat& ctrs):
        _src(src.ptr<_Ty>()),
        _src_step1(src.step1()),
        _map(map.ptr<_Ty>()),
        _map_step1(map.step1()),
        _frgd(frgd.ptr<_Ty>()),
        _frgd_step1(frgd.step1()),
        _bckgd(backgd.ptr<_Ty>()),
        _bckgd_step1(backgd.step1()),
        _ctrs(ctrs.ptr<_Ty>()),
        _ctrs_step1(ctrs.step1()),
        _cols(src.cols)
    {}

    virtual ~body_bckg_map_ctrs_() = default;


    void operator()(const cv::Range& range)const
    {
        const_pointer src = cv::alignPtr(this->_src + range.start*this->_src_step1,32);
        const_pointer map = cv::alignPtr(this->_map + range.start*this->_map_step1,32);

        pointer frgd = cv::alignPtr(this->_frgd + range.start*this->_frgd_step1,32);
        pointer bckgd = cv::alignPtr(this->_bckgd + range.start*this->_bckgd_step1,32);
        pointer ctrs = cv::alignPtr(this->_ctrs + range.start*this->_ctrs_step1,32);

        for(std::size_t r = range.start;r<(std::size_t)range.end;r++,src+=this->_src_step1,map+=this->_map_step1,frgd+=this->_frgd_step1,bckgd+=this->_bckgd_step1,ctrs+=this->_ctrs_step1)
            for(std::size_t c=0;c<(std::size_t)this->_cols;c++)
            {
                if(*(frgd + c) != 0 && *(map + c) != 0)
                {
                    *(bckgd + c) = *(src+c);
                    *(frgd+c) = 0;
                }
                *(ctrs + c) = *(src + c) - *(bckgd + c) != 0 ? 0xFF : 0;
            }
    }

};


template<class _Ty>
void apply_background_map_ctrs(cv::Mat& src,cv::Mat& map,cv::Mat& frgd,cv::Mat& bckgd,cv::Mat& ctrs)
{

    const int grain = 0x20;
    body_bckg_map_ctrs_<_Ty> body(src,map,frgd,bckgd,ctrs);

    cv::parallel_for_(cv::Range(0,src.rows),body,grain);
}

template<class _Ty>
class body_get_frgd_ : public cv::ParallelLoopBody
{
private:

    const _Ty* _src;
    const int _src_step1;

    const uchar* _cmp;
    const int _cmp_step1;

    _Ty* _dst;
    const int _dst_step1;

    const int _cols;

public:

    typedef _Ty value_type;
    typedef _Ty* pointer;
    typedef const _Ty* const_pointer;

    inline body_get_frgd_(const cv::Mat& src,const cv::Mat1b& cmp,cv::Mat& dst):
        _src(src.ptr<_Ty>()),
        _src_step1(src.step1()),
        _cmp(cmp[0]),
        _cmp_step1(cmp.step1()),
        _dst(dst.ptr<_Ty>()),
        _dst_step1(dst.step1()),
        _cols(src.cols)
    {}

    virtual ~body_get_frgd_() = default;

    void operator ()(const cv::Range& range)const
    {
        const_pointer src = cv::alignPtr(this->_src + range.start * this->_src_step1,32);
        const uchar* cmp = cv::alignPtr(this->_cmp + range.start * this->_cmp_step1,32);
        pointer dst = cv::alignPtr(this->_dst + range.start * this->_dst_step1,32);

        for(int r=range.start;r<range.end;r++,src+=this->_src_step1,dst+=this->_dst_step1,cmp+=this->_cmp_step1)
            for(int c=0;c<this->_cols;c++)
                if(*(cmp+c) > 0)
                    *(dst + c) = *(src + c);
    }

};

template<class _Ty>
void process_frgd(cv::Mat& src,cv::Mat1b& roi,cv::Mat& frgd)
{
    frgd.create(src.size(),src.type());
    frgd.setTo(0.);


    body_get_frgd_<_Ty> body(src,roi,frgd);

    cv::parallel_for_(cv::Range(0,src.rows),body,0x20);
}




template<class _Ty,class _Oty>
class body_fp_ : public cv::ParallelLoopBody
{

private:

    const _Oty* _map;
    const int _map_step1;

    _Ty* _bckgd;
    const int _bckgd_step1;

    _Ty* _fp;
    const int _fp_step1;

//    const float _avg_fd;
//    const float _avg_bd;

    const int _cols;


public:

    typedef _Ty value_type;
    typedef _Ty* pointer;
    typedef const _Ty* const_pointer;

    inline body_fp_(cv::Mat& map,cv::Mat& bckgd,cv::Mat& fp/*,_Ty& avg_fd,_Ty& avg_bd*/):
        _map(map.ptr<_Oty>()),
        _map_step1(map.step1()),
        _bckgd(bckgd.ptr<_Ty>()),
        _bckgd_step1(bckgd.step1()),
        _fp(fp.ptr<_Ty>()),
        _fp_step1(fp.step1()),
//        _avg_fd(avg_bd),
//        _avg_bd(avg_fd),
        _cols(map.cols)
    {}

    virtual ~body_fp_() = default;

    void operator()(const cv::Range& range)const
    {
         const _Oty* map = cv::alignPtr(this->_map + range.start * this->_map_step1,32);
         pointer bckgd = cv::alignPtr(this->_bckgd + range.start * this->_bckgd_step1,32);
         pointer fp = cv::alignPtr(this->_fp + range.start * this->_fp_step1,32);

         for(int r=range.start;r<range.end;r++,
                                    map+=this->_map_step1,
                                    bckgd+=this->_bckgd_step1,
                                    fp+=this->_fp_step1)
             for(int c=0;c<this->_cols;c++)
                if((*(map + c) != 0) && (*(bckgd+c) != 0))
                    std::swap(*(fp+c),*(bckgd+c));

    }

};

template<class _Ty,class _Oty>
void false_positive(cv::Mat& map,cv::Mat& bckgd,cv::Mat& fp)
{
    fp.create(bckgd.size(),bckgd.type());
    fp.setTo(0.);

    body_fp_<_Ty,_Oty> body(map,bckgd,fp/*,avg_fd,avg_bd*/);

    cv::parallel_for_(cv::Range(0,map.rows),body);
}


// Colorize the pixels.
template<class _Ty>
class map_ : public cv::ParallelLoopBody
{
private:

    const _Ty* _fd;
    const int _fd_step1;

    const _Ty* _bd;
    const int _bd_step1;

    const _Ty* _fp;
    const int _fp_step1;

    uchar* _map;
    const int _map_step1;

    const int _cols;

public:

    typedef _Ty value_type;
    typedef _Ty* pointer;
    typedef const _Ty* const_pointer;


    inline map_(cv::Mat& fd,cv::Mat& bd,cv::Mat& fp,cv::Mat3b& map):
        _fd(fd.ptr<_Ty>()),
        _fd_step1(fd.step1()),
        _bd(bd.ptr<_Ty>()),
        _bd_step1(bd.step1()),
        _fp(fp.ptr<_Ty>()),
        _fp_step1(fp.step1()),
        _map(map.data),
        _map_step1(map.step1()),
        _cols(fd.cols)
    {}

    virtual ~map_() = default;

    void operator()(const cv::Range& range)const
    {
        const_pointer fd = cv::alignPtr(this->_fd + range.start * this->_fd_step1,32);
        const_pointer bd = cv::alignPtr(this->_bd + range.start * this->_bd_step1,32);
        const_pointer fp = cv::alignPtr(this->_fp + range.start * this->_fp_step1,32);

        cv::Vec3b* map = reinterpret_cast<cv::Vec3b*>(cv::alignPtr(this->_map + range.start * this->_map_step1));

        for(int r=range.start;r<range.end;r++,
                                    fd+=this->_fd_step1,
                                    bd+=this->_bd_step1,
                                    fp+=this->_fp_step1,
                                    map+=this->_map_step1)
            for(int c=0;c<this->_cols;c++)
            {
                if(*(fd+c) != 0)
                {
                    // BLUE
                    *(map+c) = cv::Vec3b(255,0,0);
                    continue;
                }

                if(*(fp+c) != 0)
                {
                    // RED
                    *(map+c) = cv::Vec3b(0,0,255);
                    continue;
                }

                // GREEN
                *(map+c) = cv::Vec3b(0,255,0);

            }
    }

};


template<class _Ty>
void map_proc(cv::Mat& fd,cv::Mat& bd,cv::Mat& fp,cv::Mat3b& map)
{
    map.create(fd.size());
    map.setTo(cv::Scalar::all(0.));

    map_<_Ty> body(fd,bd,fp,map);

    cv::parallel_for_(cv::Range(0,map.rows),body);
}




template<class _Ty>
cv::Vec2d worker(cv::InputArray& _src,cv::OutputArray& _background,cv::OutputArray& _foreground,cv::OutputArray& _fp,cv::OutputArray& _contours,cv::OutputArray& _map)
{

    cv::Mat src = _src.getMat();

    cv::Mat background_map;



    cv::Mat tmp_src;
    cv::Mat tmp_roi;

    cv::Mat frgd;
    cv::Mat bckgd;


    cv::Mat map;

    cv::Mat ctrs;

    cv::Mat fd;

    static cv::Matx<uchar,3,3> struct_elem(0,1,0,1,1,1,0,1,0);



    cv::Vec2d min_max;

    std::size_t area = std::floor(src.rows*0.1f) * std::floor(src.cols*0.1f);


    if(src.depth() != src.type())
        cv::extractChannel(src,src,0);

    if(src.depth() > CV_8U)
        src.convertTo(tmp_src,CV_32F);
    else
        tmp_src = src;


    // Estimation of the R.O.I.

    cv::threshold(
                  tmp_src,
                  tmp_roi,
                autothresh::otsu(src),
                  1.,
                  cv::THRESH_BINARY);

    cv::multiply(tmp_src,tmp_roi,frgd,1.,src.depth());



    static cv::Point ul(1,1);

    cv::erode(frgd,frgd,struct_elem,ul);

    cv::dilate(frgd,frgd,struct_elem,ul,4);

    estimate_background_foregrounds<_Ty>(src,bckgd,frgd);






    // Fine estimation of the background.

    cv::compare(bckgd,cv::Scalar::all(0.),background_map,cv::CMP_GT);

    cv::erode(background_map,background_map,struct_elem);

    support::hole_filling(background_map,background_map,area);

    background_map.convertTo(map,src.depth());



//     Apply the updated background estimation and update the foreground as well.



    update_estimations<_Ty>(src,map,frgd,bckgd);


    cv::subtract(src,bckgd,fd);



//    Process the false positive.

    cv::Mat fp8;
    cv::Mat fp;


    cv::normalize(src,fp8,255.,0.,cv::NORM_MINMAX,CV_8U);
    cv::threshold(fp8,fp8,0.,255.,cv::THRESH_BINARY | cv::THRESH_TRIANGLE);


    false_positive<_Ty,uchar>(fp8,bckgd,fp);



    if(_fp.needed())
        fp.copyTo(_fp);

    if(_foreground.needed())
        fd.copyTo(_foreground);


    if(_map.needed())
    {
        cv::Mat3b map;

        map_proc<_Ty>(fd,bckgd,fp,map);

        map.copyTo(_map);

    }

    if(_background.needed())
        bckgd.copyTo(_background);


    if(_contours.needed())
    {
        cv::compare(tmp_roi,cv::Scalar::all(0.),background_map,cv::CMP_GT);


        support::hole_filling(background_map,background_map,area);

        background_map.convertTo(map,src.depth());

        cv::morphologyEx(map,map,cv::MORPH_DILATE,struct_elem);


        ctrs = cv::Mat::zeros(src.size(),src.type());

        apply_background_map_ctrs<_Ty>(src,map,frgd,bckgd,ctrs);

        cv::subtract(_src,bckgd,_contours,cv::noArray(),CV_32SC1);
    }




    cv::minMaxIdx(tmp_src,min_max.val,min_max.val+1);

    *(min_max.val+1) -= *(min_max.val);

    return min_max;
}

struct init_cl_t
{
    inline init_cl_t()
    {
        if(cv::ocl::haveOpenCL())
            cv::ocl::setUseOpenCL(true);
    }

    ~init_cl_t() = default;
};

// weird but needed.
init_cl_t initCL;

}



cv::Vec2d get_structure_of_interest(cv::InputArray _src, cv::OutputArray _background, cv::OutputArray _foreground, cv::OutputArray _fd, cv::OutputArray _contours, cv::OutputArray _map)
{
    CV_DbgAssert((_src.depth() < CV_32F) && (_src.depth() == _src.type()));


    typedef cv::Vec2d (*function_type)(cv::InputArray&,cv::OutputArray&,cv::OutputArray&,cv::OutputArray&,cv::OutputArray&,cv::OutputArray& _map);

    static const function_type functions[] = {
        worker<uchar>,
        worker<schar>,
        worker<ushort>,
        worker<short>,
        worker<int>
    };


    function_type fun = functions[_src.depth()];

    return fun(_src,_background,_foreground,_fd,_contours,_map);

}


cv::Vec2d get_structure_of_interest_refined(cv::InputArray _src, cv::OutputArray _background, cv::OutputArray _foreground,cv::OutputArray _fd, cv::OutputArray _contours,cv::OutputArray _map)
{
    cv::Mat map;

    cv::Vec2d ret = get_structure_of_interest(_src,cv::noArray(),cv::noArray(),cv::noArray(),_contours,map);

    smooth(_src,map,_foreground,_background,_fd,_map);

    return ret;
}




}

