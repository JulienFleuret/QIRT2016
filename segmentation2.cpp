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


//#include "segmentation.h"
//#include "autothresh.h"
//#include "regions.h"

//#include "support.h"

//#include <opencv2/core/utility.hpp>
//#include <opencv2/core/hal/hal.hpp>
//#include <opencv2/core/ocl.hpp>
//#include <opencv2/imgproc.hpp>

//#include <memory>

//#include <tbb/blocked_range2d.h>
////#include <tbb/parallel_reduce.h>

//#include <tbb/parallel_for.h>
//#include <tbb/parallel_for_each.h>

//#include <iostream>

//#include <opencv2/highgui.hpp>

//#include <openblas/cblas.h>

//// Not really necessary, just more confortable.
//namespace cv
//{
//    typedef cv::Matx<uchar,3,3> Matx33b;
//}

//namespace segmentation
//{

//namespace
//{

//class cpy_
//{
//public:

//    typedef float value_type;
//    typedef float* pointer;
//    typedef const float* const_pointer;

//private:

//    struct core_t
//    {

//        const_pointer _src;
//        const std::size_t _src_step1;

//        pointer _dst;
//        const std::size_t _dst_step1;

//        const std::size_t _cols;

//        inline core_t(cv::Mat1f& src,cv::Mat1f& dst):
//            _src(src[0]),
//            _src_step1(src.step1()),
//            _dst(dst[0]),
//            _dst_step1(dst.step1()),
//            _cols(src.cols)
//        {}

//        ~core_t() = default;

//    };

//    std::shared_ptr<core_t> _core;

//public:

//    inline cpy_(cv::Mat1f& src,cv::Mat1f& dst):
//        _core(new core_t(src,dst))
//    {}

//    inline cpy_(const cpy_& obj):
//        _core(obj._core)
//    {}

//    inline cpy_(cpy_&& obj):
//        _core(std::move(obj._core))
//    {}

//    ~cpy_() = default;

//    inline void operator()(const tbb::blocked_range<std::size_t>& range)const
//    {
//        const_pointer src = this->_core->_src + range.begin() * this->_core->_src_step1;
//        pointer dst = this->_core->_dst + range.begin() * this->_core->_dst_step1;

//        // Not made for this usage... how very efficient.
//        cblas_saxpy(this->_core->_cols,1.f,src,1,dst,1);
//    }

//};

//class thresh_body
//{
//public:

//    typedef float value_type;
//    typedef float* pointer;
//    typedef const float* const_pointer;

//private:

//    struct core_t
//    {

//        const_pointer _src;
//        const std::size_t _src_step1;

//        pointer _dst;
//        const std::size_t _dst_step1;

//        value_type _thresh;

//        const std::size_t _cols;

//        const std::size_t _grain;

//        inline core_t(cv::Mat1f& src,cv::Mat1f& dst,const value_type& thresh,const std::size_t& grain):
//            _src(src[0]),
//            _src_step1(src.step1()),
//            _dst(dst[0]),
//            _dst(dst.step1()),
//            _thresh(thresh),
//            _cols(src.cols),
//            _grain(grain)
//        {}

//        ~core_t() = default;
//    };

//    std::shared_ptr<core_t> _core;

//public:

//    inline thresh_body(cv::Mat1f& src,cv::Mat1f& dst,const value_type& thresh,const std::size_t& grain):
//        _core(new core_t(src,dst,thresh,grain))
//    {}

//    inline thresh_body(const thresh_body& obj):
//        _core(obj._core)
//    {}

//    inline thresh_body(thresh_body&& obj):
//        _core(std::move(obj._core))
//    {}

//    ~thresh_body() = default;

//    void operator()(const tbb::blocked_range<std::size_t>& range)const
//    {
//        const_pointer src = this->_core->_src + range.begin() * this->_core->_src_step1;
//        pointer dst = this->_core->_dst + range.end() * this->_core->_dst_step1;

//#if CV_AVX
//        const std::size_t step = 8;
//        const __m256 thresh = _mm256_set1_ps(this->_core->_thresh);
//#else
//        const std::size_t step = 4;
//#if CV_SSE
//        const __m128 thresh = _mm_set1_ps(this->_core->_thresh);
//#endif

//#endif

//        const_pointer src_begin = src;
//        const_pointer src_end = src_begin + this->_core->_cols;

//        for(std::size_t r=range.begin();r<range.end();r++,src+=this->_core->_src_step1,dst+=this->_core->_dst_step1)
//        {
//            pointer it_dst = dst;
//            for(const_pointer it_src = src_begin;it_src != src_end;it_src+=step,it_dst+=step)
//            {
//#if CV_AVX
//                __m256 s = _mm256_load_ps(it_src);

//                __m256 cmp = _mm256_cmp_ps(s,thresh,_CMP_GT_OQ);

//                __m256 d = _mm256_and_ps(cmp,s);

//                _mm256_stream_ps(it_dst,d);

//#elif CV_SSE

//                __m128 s = _mm_load_ps(it_src);

//                __m128 cmp = _mm_cmpgt_ps(s,thresh);

//                __m128 d = _mm_and_ps(cmp,s);

//                _mm_stream_ps(it_dst,d);
//#else

//                if(*it_src > this->_core->_thresh)
//                    *it_dst = *it_src;

//                if(*(it_src+1) > this->_core->_thresh)
//                    *(it_dst+1) = *(it_src+1);

//                if(*(it_src+2) > this->_core->_thresh)
//                    *(it_dst+2) = *(it_src+2);

//                if(*(it_src+3) > this->_core->_thresh)
//                    *(it_dst+3) = *(it_src+3);
//#endif
//            }
//        }

//    }

//    void operator()(const tbb::blocked_range2d<std::size_t>& range)const
//    {

//        const tbb::blocked_range<std::size_t>& rows = range.rows();
//        const tbb::blocked_range<std::size_t>& tmp = range.cols();

//        tbb::blocked_range<std::size_t> cols(tmp.begin() * this->_core->_grain, std::min(tmp.end() * this->_core->_grain, this->_core->_cols));

//        const_pointer src = this->_core->_src + rows.begin() * this->_core->_src_step1 + cols.begin();
//        pointer dst = this->_core->_dst + rows.end() * this->_core->_dst_step1 + cols.begin();

//#if CV_AVX
//        const std::size_t step = 8;
//        const __m256 thresh = _mm256_set1_ps(this->_core->_thresh);
//#else
//        const std::size_t step = 4;
//#if CV_SSE
//        const __m128 thresh = _mm_set1_ps(this->_core->_thresh);
//#endif

//#endif

//        const_pointer src_begin = src;
//        const_pointer src_end = src_begin + cols.size();

//        for(std::size_t r=rows.begin();r<rows.end();r++,src+=this->_core->_src_step1,dst+=this->_core->_dst_step1)
//        {
//            pointer it_dst = dst;
//            for(const_pointer it_src = src_begin;it_src != src_end;it_src+=step,it_dst+=step)
//            {
//#if CV_AVX
//                __m256 s = _mm256_load_ps(it_src);

//                __m256 cmp = _mm256_cmp_ps(s,thresh,_CMP_GT_OQ);

//                __m256 d = _mm256_and_ps(cmp,s);

//                _mm256_stream_ps(it_dst,d);

//#elif CV_SSE

//                __m128 s = _mm_load_ps(it_src);

//                __m128 cmp = _mm_cmpgt_ps(s,thresh);

//                __m128 d = _mm_and_ps(cmp,s);

//                _mm_stream_ps(it_dst,d);
//#else

//                if(*it_src > this->_core->_thresh)
//                    *it_dst = *it_src;

//                if(*(it_src+1) > this->_core->_thresh)
//                    *(it_dst+1) = *(it_src+1);

//                if(*(it_src+2) > this->_core->_thresh)
//                    *(it_dst+2) = *(it_src+2);

//                if(*(it_src+3) > this->_core->_thresh)
//                    *(it_dst+3) = *(it_src+3);
//#endif
//            }
//        }

//    }

//};


//class body_estimate_
//{

//public:

//    typedef float value_type;
//    typedef float* pointer;
//    typedef const float* const_pointer;

//private:

//    struct core_t
//    {
//        pointer _src;
//        const std::size_t _src_step1;

//        pointer _bd;
//        const std::size_t _bd_step1;

//        pointer _fd;
//        const std::size_t _fd_step1;

//        const std::size_t _cols;
//        const std::size_t _grain;

//        inline core_t(cv::Mat1f& src,
//                      cv::Mat1f& bd,
//                      cv::Mat1f& fd,
//                      const std::size_t& grain
//                      ):
//            _src(src[0]),
//            _src_step1(src.step1()),
//            _bd(bd[0]),
//            _bd_step1(bd.step1()),
//            _fd(fd[0]),
//            _fd_step1(fd.step1()),
//            _cols(src.cols),
//            _grain(grain)
//        {}

//        ~core_t() = default;
//    };

//    std::shared_ptr<core_t> _core;

// public:

//    inline body_estimate_(cv::Mat& src,cv::Mat& bd,cv::Mat& fd,const std::size_t& grain):
//        _core(new core_t(src,bd,fd,grain))
//    {}

//    inline body_estimate_(const body_estimate_& obj):
//        _core(obj._core)
//    {}

//    inline body_estimate_(body_estimate_&& obj):
//        _core(std::move(obj._core))
//    {}

//    ~body_estimate_() = default;

//    inline void operator()(const tbb::blocked_range<std::size_t>& tmp)const
//    {

//#if CV_AVX
//        static const std::size_t step = 8;
//        static __m256 zeros = _mm256_setzero_ps();
//#else
//        static const std::size_t step = 4;

//#if CV_SSE
//        static __m128 zeros = _mm_setzero_ps();
//#endif

//#endif

//        tbb::blocked_range<std::size_t> range(tmp.begin() * this->_core->_grain, std::min(tmp.end() * this->_core->_grain, this->_core->_cols));

//        pointer src = this->_core->_src + range.begin();
//        pointer fd = this->_core->_fd + range.begin();
//        pointer bd = this->_core->_bd + range.begin();

//        for(std::size_t l=0;l<range.size();l+=step,src+=step,fd+=step,bd+=step)
//        {
//#if CV_AVX

//            __m256 s = _mm256_load_ps(src);
//            __m256 f = _mm256_load_ps(fd);

//            __m256 cmp = _mm256_cmp_ps(f,zeros,_CMP_EQ_OQ);

//            f = _mm256_andnot_ps(cmp,s);
//            __m256 b = _mm256_and_ps(cmp,s);

//            _mm256_stream_ps(fd,f);
//            _mm256_stream_ps(bd,b);

//#elif CV_SSE

//            __m128 s = _mm_load_ps(src);
//            __m128 f = _mm_load_ps(fd);

//            __m128 cmp = _mm_cmpeq_ps(f,zeros);

//            f = _mm_andnot_ps(cmp,zeros);
//            __m128 b = _mm_and_ps(cmp,s);

//            _mm_stream_ps(fd,f);
//            _mm_stream_ps(bd,b);

//#else
//            if(*fd != 0)
//            {
//                *fd = *src;
//                *bd = 0;
//            }
//            else
//                *bd = *src;



//            if(*(fd+1) != 0)
//            {
//                *(fd+1) = *(src+1);
//                *(bd+1) = 0;
//            }
//            else
//                *(bd+1) = *(src+1);



//            if(*(fd+2) != 0)
//            {
//                *(fd+2) = *(src+2);
//                *(bd+2) = 0;
//            }
//            else
//                *(bd+2) = *(src+2);



//            if(*(fd+3) != 0)
//            {
//                *(fd+3) = *(src+3);
//                *(bd+3) = 0;
//            }
//            else
//                *(bd+3) = *(src+3);


//#endif
//        }
//    }



//};

//class cmp_gt_
//{
//public:

//    typedef float value_type;
//    typedef float* pointer;
//    typedef const float* const_pointer;

//private:

//    struct core_t
//    {
//        pointer _src;
//        const std::size_t _src_step1;

//        pointer _dst;
//        const std::size_t _dst_step1;

//        const std::size_t _cols;
//        const std::size_t _grain;

//        inline core_t(cv::Mat1f& src,cv::Mat1f& dst,const std::size_t& grain):
//            _src(src[0]),
//            _src_step1(src.step1()),
//            _dst(dst[0]),
//            _dst_step1(dst.step1()),
//            _cols(src.cols),
//            _grain(grain)
//        {}

//        ~core_t() = default;
//    };

//    std::shared_ptr<core_t> _core;

//public:

//    inline cmp_gt_(cv::Mat1f& src,cv::Mat1f& dst,const std::size_t& grain):
//        _core(src,dst,grain)
//    {}

//    inline cmp_gt_(const cmp_gt_& obj):
//        _core(obj._core)
//    {}

//    inline cmp_gt_(cmp_gt_&& obj):
//        _core(std::move(obj._core))
//    {}

//    ~cmp_gt_() = default;

//    void operator()(const tbb::blocked_range<std::size_t>& tmp)const
//    {
//#if CV_AVX
//        //0xffffffff
//        static const std::size_t step = 8;
//        static const __m256 right = _mm256_set1_ps(0xff);
//        static const __m256 zeros = _mm256_setzero_ps();
//#elif CV_SSE
//        static const std::size_t step = 4;
//        static const __m128 right = _mm_set1_ps(0xff);
//        static const __m128 zeros = _mm_setzero_ps();
//#else
//        static const std::size_t step = 4;
//        static const float right = 0xff;
//#endif

//        tbb::blocked_range<std::size_t> range(tmp.begin() * this->_core->_grain,tmp.end() * this->_core->_grain);

//        pointer src = this->_core->_src + range.begin() * this->_core->_src_step1;
//        pointer dst = this->_core->_dst + range.begin() * this->_core->_dst_step1;

//        for(std::size_t r = 0; r<range.size();c+=step,src+=step,dst+=step)
//        {
//#if CV_AVX
//            __m256 s = _mm256_load_ps(src);

//            s = _mm256_cmp_ps(s,zeros,_CMP_EQ_OQ);
//            s = _mm256_andnot_ps(s,right);

//            _mm256_stream_ps(dst,s);
//#elif CV_SSE
//            __m128 s = _mm_load_ps(src);

//            s = _mm_cmpeq_ps(s,zeros);
//            s = _mm_andnot_ps(s,right);

//            _mm_stream_ps(dst,s);
//#else

//            if(*src > 0)
//                *dst = right;

//            if(*(src+1) > 0)
//                *(dst+1) = right;

//            if(*(src+2) > 0)
//                *(dst+2) = right;

//            if(*(src+3) > 0)
//                *(dst+3) = right;
//#endif
//        }

//    }

//};

//}

//cv::Vec2d get_structure_of_interest(cv::InputArray _src, cv::OutputArray _background, cv::OutputArray _foreground, cv::OutputArray _fp, cv::OutputArray _contours,cv::OutputArray _map)
//{
//    CV_DbgAssert( (_src.isMat() || ((_src.kind() & cv::_InputArray::EXPR) == cv::_InputArray::EXPR) ) && (_src.depth() < CV_32F) && (_src.depth() == _src.type()));

//    cv::Mat1f src = _src.getMat();
//    cv::Mat1f fgd;
//    cv::Mat1f bgd;
//    cv::Mat1f bm;
//    const std::size_t src_depth = _src.depth();
//    std::size_t grain = 0;
//    std::size_t cols = 0;

//    // structuring element.
//    static const cv::Matx33b se(0,1,0,1,1,1,0,1,0);

//#if CV_AVX

//    const std::size_t algn = 8;

//#else

//    const std::size_t algn = 4;

//#endif


//        if((src.cols%algn) != 0 || !src.isContinuous())
//        {
//            cv::Mat1f tmp = cv::Mat1f::zeros(src.rows,cv::alignSize(src.cols,algn));

//            cpy_ copy_body(src,tmp);

//            tbb::parallel_for(tbb::blocked_range<std::size_t>(0,tmp.rows,0x20),copy_body);

//            src = tmp;

//            fgd = cv::Mat1f::zeros(src.size());
//            bgd = cv::Mat1f::zeros(src.size());
//            bm = cv::Mat1f::zeros(src.size());

//        }

//        std::size_t S = std::floor(static_cast<float>(src.rows)*0.1f) * std::floor(static_cast<float>(src.cols)*0.1f);
//        std::size_t N = 3;


//        // Threshold the image.

//        float thresh = autothresh::otsu(src,src_depth);

//        if(src.total() <= 0x186A0)
//        {
//            grain = 0x01;

//            thresh_body th_body(src,fgd,thresh,grain);

//            tbb::parallel_for(tbb::blocked_range<std::size_t>(0,src.cols),th_body);
//        }
//        else
//        {
//            grain = src.cols > 0x400 ? 0x400 : 0x20;

//            cols = std::ceil(static_cast<float>(src.cols)/static_cast<float>(grain));

//            thresh_body th_body(src,fgd,thresh,grain);

//            tbb::parallel_for(tbb::blocked_range2d<std::size_t>(0,src.rows,0,cols),th_body);
//        }



//        // Coarse estimation of the foreground and background.

//        cv::erode(fgd,fgd,se);
//        cv::dilate(fgd,fgd,se,cv::Point(-1,-1),N+1);


//        grain = src.total() > 0x186A0 ? 0x186A0 : 0x400;
//        cols = std::ceil(static_cast<float>(src.total())/static_cast<float>(grain));

//        body_estimate_ fd_gd(src,fgd,bgd,grain);

//        tbb::parallel_for(tbb::blocked_range<std::size_t>(0,cols),fd_gd);



//        // Fine estimation of the background.

//        cmp_gt_ cmp_body(bgd,bm,grain);

//        tbb::parallel_for(tbb::blocked_range<std::size_t>(0,cols),cmp_body);


//        cv::erode(bm,bm,se);




//}


//cv::Vec2d get_structure_of_interest_refined(cv::InputArray _src, cv::OutputArray _background, cv::OutputArray _foreground,cv::OutputArray _fp, cv::OutputArray _contours,cv::OutputArray _map)
//{
//    cv::Mat map;

//    cv::Vec2d ret = get_structure_of_interest(_src,cv::noArray(),cv::noArray(),cv::noArray(),_contours,map);

//    smooth(_src,map,_foreground,_background,_fp,_map);

//    return ret;
//}

//}
