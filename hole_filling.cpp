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

#include <opencv2/imgproc.hpp>

#include <mutex>

#include <memory>

#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>



//#undef CV_AVX2
//#define CV_AVX2 0

//#undef CV_SSE2
//#define CV_SSE2 0

//#undef CV_SSE3
//#define CV_SSE3 0

cv::Mutex mtx;


namespace support
{


//namespace
//{
//class ctrs_proc : public cv::ParallelLoopBody
//{
//private:

//    std::vector<std::vector<cv::Point> >& _contours;
//    std::vector<cv::Vec4i>& _hierarchy;
//    cv::Scalar& _colour;

//    cv::Mat1b& _src;
//    cv::Mat1b _tmp;

//    const int& _area;
//    const int& _upper_bound;

//    mutable std::mutex _mtx;

//public:

//    inline ctrs_proc(std::vector<std::vector<cv::Point> >& contours,std::vector<cv::Vec4i>& hierarchy,cv::Scalar& colour,cv::Mat1b& src,const int& area,const int& upper_bound):
//        _contours(contours),
//        _hierarchy(hierarchy),
//        _colour(colour),
//        _src(src),
//        _tmp(src.size(),uchar(0)),
//        _area(area),
//        _upper_bound(upper_bound)
//    {}

//    ~ctrs_proc() = default;

//    void operator()(const cv::Range& range)const
//    {
//        for(int i=range.start;i<range.end;i++)
//        {
//            cv::Rect roi = cv::boundingRect(this->_contours.at(i));

//            std::lock_guard<std::mutex> lck(this->_mtx);

//            if( (roi.area() > this->_area) && (roi.area() < this->_upper_bound))
//            {

//                cv::drawContours(this->_tmp,this->_contours,i,this->_colour,cv::FILLED,cv::LINE_AA,this->_hierarchy,0);


//                cv::Mat1b roi_s = this->_src(roi);
//                cv::Mat1b roi_d = this->_tmp(roi);


//#if CV_ENABLE_UNROLLED
//                for(int r=0,c=0;r<roi_s.rows;r++,c=0)
//                {
//                    const uchar* s = roi_s[r];
//                    uchar* d = roi_d[r];

//                    for(c=0;c<roi_s.cols-4;c+=4)
//                    {
//                        uchar p1 = *(s+c);
//                        uchar p2 = *(s+c+1);

//                        if(p1==0)
//                            *(d+c) = 0;

//                        if(p2==0)
//                            *(d+c+1) = 0;

//                        p1 = *(s+c+2);
//                        p2 = *(s+c+3);

//                        if(p1==0)
//                            *(d+c+2) = 0;

//                        if(p2==0)
//                            *(d+c+3) = 0;
//                    }
//                    for(;c<roi_s.cols;c++)
//                        if(*(s+c)==0)
//                            *(d+c) = 0;
//                }
//#else
//                for(int r=0,c=0;r<roi_s.rows;r++,c=0)
//                    for(int c=0;c<roi_s.cols;c++)
//                        if(roi_s(r,c)==0)
//                            roi_d(r,c) = 0;
//#endif

//            }
//            else
//                cv::drawContours(this->_tmp,this->_contours,i,this->_colour,cv::FILLED,cv::LINE_8,this->_hierarchy,0);



//        }
//    }

//    inline operator cv::Mat1b()const{ return this->_tmp;}
//};
//}


namespace
{

template<class _Ty>
class base
{


public:
    typedef typename std::iterator_traits<_Ty*>::value_type value_type;
    typedef typename std::iterator_traits<_Ty*>::pointer pointer;
    typedef typename std::iterator_traits<const _Ty*>::pointer const_pointer;

protected:

    struct core_t
    {
        const_pointer _src;
        const std::size_t _src_step1;

        pointer _dst1;
        const std::size_t _dst1_step1;

        pointer _dst2;
        const std::size_t _dst2_step1;

        const std::size_t _cols;
        const std::size_t _grain;

        const bool _is_inline;

#if CV_SSE2
        const bool _is_algn;
#if CV_AVX2
        static const int algn = 32;
        static const int steps = 32/sizeof(_Ty);
#else
        static const int algn = 16;
        static const int steps = 16/sizeof(_Ty);
#endif

#endif



        inline core_t(cv::Mat_<value_type>& src,cv::Mat_<value_type>& dst1,cv::Mat_<value_type>& dst2,const std::size_t& grain):
            _src(src[0]),
            _src_step1(src.isContinuous() && dst1.isContinuous() && dst2.isContinuous() ? 1 : src.step1()),
            _dst1(dst1[0]),
            _dst1_step1(src.isContinuous() && dst1.isContinuous() && dst2.isContinuous() ? 1 : dst1.step1()),
            _dst2(dst2[0]),
            _dst2_step1(src.isContinuous() && dst1.isContinuous() && dst2.isContinuous() ? 1 : dst2.step1()),
            _cols(src.isContinuous() && dst1.isContinuous() && dst2.isContinuous() ? src.total() : src.cols),
            _grain(grain),
            _is_inline(src.isContinuous() && dst1.isContinuous() && dst2.isContinuous())
#if CV_SSE2
          //            ,_is_algn( ((src.isContinuous() ? src.total()%steps : src.cols%steps) == 0) &&
//                       ((dst1.isContinuous() ? dst1.total()%steps : dst1.cols%steps) == 0) &&
//                       ((dst2.isContinuous() ? dst2.total()%steps : dst2.cols%steps) == 0) &&
//                       src.isContinuous() &&
//                       dst1.isContinuous() &&
//                       dst2.isContinuous() )
          ,_is_algn( src.isContinuous() && dst1.isContinuous() && dst2.isContinuous() && ((src.step%algn)==0) && ((dst1.step%algn)==0) && ((dst2.step%algn)==0) )
#endif
        {}

        ~core_t() = default;
    };

    std::shared_ptr<core_t> _core;

public:

    inline base(cv::Mat_<value_type>& src,cv::Mat_<value_type>& dst1, cv::Mat_<value_type>& dst2, const std::size_t& grain):
        _core(new core_t(src, dst1, dst2, grain))
    {}

    inline base(const base& obj):
        _core(obj._core)
    {}

    inline base(base&& obj):
        _core(std::move(obj._core))
    {}

    virtual ~base() = default;

    virtual void operator ()(const tbb::blocked_range<std::size_t>&)const = 0;
    virtual void operator()(const tbb::blocked_range2d<std::size_t>&)const = 0;

};

template<class _Ty>
class step1_as_is : public base<_Ty>
{
public:

    typedef typename base<_Ty>::value_type value_type;
    typedef typename base<_Ty>::pointer pointer;
    typedef typename base<_Ty>::const_pointer const_pointer;

    typedef base<_Ty> MyBase;

    inline step1_as_is(cv::Mat_<value_type>& src,cv::Mat_<value_type>& dst1,cv::Mat_<value_type>& dst2,const std::size_t& grain):
        MyBase(src,dst1,dst2,grain)
    {}

    inline step1_as_is(const step1_as_is& obj):
        MyBase(obj)
    {}

    inline step1_as_is(step1_as_is&& obj):
        MyBase(std::move(obj))
    {}

    virtual ~step1_as_is() = default;

    virtual void operator()(const tbb::blocked_range<std::size_t>& tmp)const;
    virtual void operator()(const tbb::blocked_range2d<std::size_t>& range)const;



};

#if CV_SSE2

template<>
void step1_as_is<uchar>::operator()(const tbb::blocked_range<std::size_t>& tmp)const
{
    tbb::blocked_range<std::size_t> range(tmp.begin() * this->_core->_grain,std::min( tmp.end() * this->_core->_grain,this->_core->_cols ) );

    const_pointer src = this->_core->_src + range.begin() * this->_core->_src_step1;
    pointer dst1 = this->_core->_dst1 + range.begin() * this->_core->_dst1_step1;
    pointer dst2 = this->_core->_dst2 + range.begin() * this->_core->_dst2_step1;

#if CV_AVX2
    static const std::size_t step = 32;
    static const __m256i mask = _mm256_set1_epi8(0xFF);
    static const __m256i zeros = _mm256_setzero_si256();
#else
    static const std::size_t step = 16;
    static const __m128i mask = _mm_set1_epi8(0xFF);
    static const __m128i zeros = _mm_setzero_si128();
#endif

    if(this->_core->_is_algn)
    {
        for(std::size_t r=range.begin(); r<range.end(); r++,
                                                        src+=this->_core->_src_step1,
                                                        dst1+=this->_core->_dst1_step1,
                                                        dst2+=this->_core->_dst2_step1)
        {
            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            for(std::size_t c=0; c<this->_core->_cols; c+=step,
                                                       it_src+=step,
                                                       it_dst1+=step,
                                                       it_dst2+=step)
            {
#if CV_AVX2
                __m256i I = _mm256_load_si256(reinterpret_cast<const __m256i*>(it_src));

                __m256i cmp = _mm256_cmpgt_epi8(I,zeros);

                I = _mm256_and_si256(cmp,mask);
                __m256i Ic = _mm256_andnot_si256(cmp,mask);

                I = _mm256_or_si256(I,Ic);
                I = _mm256_andnot_si256(I,mask);

                _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst1),I);
                _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst2),Ic);

#else

                __m128i I = _mm_load_si128(reinterpret_cast<const __m128i*>(it_src));

                __m128i cmp = _mm_cmpgt_epi8(I,zeros);

                I = _mm_and_si128(cmp,mask);
                __m128i Ic = _mm_andnot_si128(cmp,mask);

                I = _mm_or_si128(I,Ic);
                I = _mm_andnot_si128(I,mask);

                _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst1),I);
                _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst2),Ic);
#endif
            }
        }
    }
    else
    {
        const std::size_t stop = this->_core->_cols - (this->_core->_cols%step);

        for(std::size_t r=range.begin(); r<range.end(); r++,
                                                        src+=this->_core->_src_step1,
                                                        dst1+=this->_core->_dst1_step1,
                                                        dst2+=this->_core->_dst2_step1)
        {
            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            std::size_t c=0;

            for(;c<stop; c+=step,
                                       it_src+=step,
                                       it_dst1+=step,
                                       it_dst2+=step)
            {
#if CV_AVX2
                __m256i I = _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(it_src));

                __m256i cmp = _mm256_cmpgt_epi8(I,zeros);

                I = _mm256_and_si256(cmp,mask);
                __m256i Ic = _mm256_andnot_si256(cmp,mask);

                I = _mm256_or_si256(I,Ic);
                I = _mm256_andnot_si256(I,mask);

                _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst1),I);
                _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst2),Ic);

#else

#if CV_SSE3
                __m128i I = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(it_src));
#else
                __m128i I = _mm_loadu_si128(reinterpret_cast<const __m128i*>(it_src));
#endif

                __m128i cmp = _mm_cmpgt_epi8(I,zeros);

                I = _mm_and_si128(cmp,mask);
                __m128i Ic = _mm_andnot_si128(cmp,mask);

                I = _mm_or_si128(I,Ic);
                I = _mm_andnot_si128(I,mask);

                _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst1),I);
                _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst2),Ic);
#endif
            }

            for(;c<this->_core->_cols; c++, it_src++, it_dst1++, it_dst2++)
            {
                value_type I = *it_src > 0 ? 0x0 : 0xFF;
                value_type Ic = ~I;

                I = ~(I | Ic);

                *it_dst1 = I;
                *it_dst2 = Ic;
            }
        }
    }
}

template<>
void step1_as_is<uchar>::operator()(const tbb::blocked_range2d<std::size_t>& range)const
{

    const tbb::blocked_range<std::size_t>& rows = range.rows();
    const tbb::blocked_range<std::size_t>& tmp = range.cols();

    tbb::blocked_range<std::size_t> cols(tmp.begin() * this->_core->_grain, std::min( tmp.end() * this->_core->_grain, this->_core->_cols));

    const_pointer src = this->_core->_src + rows.begin() * this->_core->_src_step1 + cols.begin();
    pointer dst1 = this->_core->_dst1 + rows.begin() * this->_core->_dst1_step1 + cols.begin();
    pointer dst2 = this->_core->_dst2 + rows.begin() * this->_core->_dst2_step1 + cols.begin();

#if CV_AVX2
    static const std::size_t step = 32;
    static const __m256i mask = _mm256_set1_epi8(0xFF);
    static const __m256i zeros = _mm256_setzero_si256();
#else
    static const std::size_t step = 16;
    static const __m128i mask = _mm_set1_epi8(0xFF);
    static const __m128i zeros = _mm_setzero_si128();
#endif


    if(this->_core->_is_algn)
    {
        for(std::size_t r=rows.begin();r<rows.end();r++,src+=this->_core->_src_step1,dst1+=this->_core->_dst1_step1,dst2+=this->_core->_dst2_step1)
        {

            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            for(std::size_t c=0;c<cols.size();c+=step,it_src+=step,it_dst1+=step,it_dst2+=step)
            {
#if CV_AVX2
                __m256i I = _mm256_load_si256(reinterpret_cast<const __m256i*>(it_src));

                __m256i cmp = _mm256_cmpgt_epi8(I,zeros);

                I = _mm256_and_si256(cmp,mask);
                __m256i Ic = _mm256_andnot_si256(cmp,mask);

                I = _mm256_or_si256(I,Ic);
                I = _mm256_andnot_si256(I,mask);

                _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst1),I);
                _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst2),Ic);

#else

                __m128i I = _mm_load_si128(reinterpret_cast<const __m128i*>(it_src));

                __m128i cmp = _mm_cmpgt_epi8(I,zeros);

                I = _mm_and_si128(cmp,mask);
                __m128i Ic = _mm_andnot_si128(cmp,mask);

                I = _mm_or_si128(I,Ic);
                I = _mm_andnot_si128(I,mask);

                _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst1),I);
                _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst2),Ic);
#endif
            }
        }
    }
    else
    {


        const std::size_t stop = cols.size() - (cols.size()%step);

        for(std::size_t r=rows.begin(); r<rows.end(); r++,
                                                      src+=this->_core->_src_step1,
                                                      dst1+=this->_core->_dst1_step1,
                                                      dst2+=this->_core->_dst2_step1)
        {
            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            std::size_t c=0;

            for(;c<stop; c+=step,
                         it_src+=step,
                         it_dst1+=step,
                         it_dst2+=step)
            {
#if CV_AVX2
                __m256i I = _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(it_src));

                __m256i cmp = _mm256_cmpgt_epi8(I,zeros);

                I = _mm256_and_si256(cmp,mask);
                __m256i Ic = _mm256_andnot_si256(cmp,mask);

                I = _mm256_or_si256(I,Ic);
                I = _mm256_andnot_si256(I,mask);

                _mm256_storeu_si256(reinterpret_cast<__m256i*>(it_dst1),I);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(it_dst2),Ic);

#else

#if CV_SSE3
                __m128i I = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(it_src));
#else
                __m128i I = _mm_load_si128(reinterpret_cast<const __m128i*>(it_src));
#endif
                __m128i cmp = _mm_cmpgt_epi8(I,zeros);

                I = _mm_and_si128(cmp,mask);
                __m128i Ic = _mm_andnot_si128(cmp,mask);

                I = _mm_or_si128(I,Ic);
                I = _mm_andnot_si128(I,mask);


                _mm_storeu_si128(reinterpret_cast<__m128i*>(it_dst1),I);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(it_dst2),Ic);
#endif
            }

            for(;c<cols.size();c++,it_src++,it_dst1++,it_dst2++)
            {
                value_type I = *it_src > 0 ? 0x0 : 0xFF;
                value_type Ic = ~I;

                I = ~(I | Ic);

                *it_dst1 = I;
                *it_dst2 = Ic;
            }
        }

    }


}


template<>
void step1_as_is<schar>::operator()(const tbb::blocked_range<std::size_t>& tmp)const
{
    tbb::blocked_range<std::size_t> range(tmp.begin() * this->_core->_grain,std::min( tmp.end() * this->_core->_grain,this->_core->_cols ) );

    const_pointer src = this->_core->_src + range.begin() * this->_core->_src_step1;
    pointer dst1 = this->_core->_dst1 + range.begin() * this->_core->_dst1_step1;
    pointer dst2 = this->_core->_dst2 + range.begin() * this->_core->_dst2_step1;

#if CV_AVX2
    static const std::size_t step = 32;
    static const __m256i mask = _mm256_set1_epi8(0xFF);
    static const __m256i zeros = _mm256_setzero_si256();
#else
    static const std::size_t step = 16;
    static const __m128i mask = _mm_set1_epi8(0xFF);
    static const __m128i zeros = _mm_setzero_si128();
#endif

    if(this->_core->_is_algn)
    {
        for(std::size_t r=range.begin(); r<range.end(); r++,
                                                        src+=this->_core->_src_step1,
                                                        dst1+=this->_core->_dst1_step1,
                                                        dst2+=this->_core->_dst2_step1)
        {
            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            for(std::size_t c=0; c<this->_core->_cols; c+=step,
                                                       it_src+=step,
                                                       it_dst1+=step,
                                                       it_dst2+=step)
            {
#if CV_AVX2
                __m256i I = _mm256_load_si256(reinterpret_cast<const __m256i*>(it_src));

                __m256i cmp = _mm256_cmpgt_epi8(I,zeros);

                I = _mm256_and_si256(cmp,mask);
                __m256i Ic = _mm256_andnot_si256(cmp,mask);

                I = _mm256_or_si256(I,Ic);
                I = _mm256_andnot_si256(I,mask);

                _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst1),I);
                _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst2),Ic);

#else

                __m128i I = _mm_load_si128(reinterpret_cast<const __m128i*>(it_src));

                __m128i cmp = _mm_cmpgt_epi8(I,zeros);

                I = _mm_and_si128(cmp,mask);
                __m128i Ic = _mm_andnot_si128(cmp,mask);

                I = _mm_or_si128(I,Ic);
                I = _mm_andnot_si128(I,mask);

                _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst1),I);
                _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst2),Ic);
#endif
            }
        }
    }
    else
    {
        const std::size_t stop = this->_core->_cols - (this->_core->_cols%step);

        for(std::size_t r=range.begin(); r<range.end(); r++,
                                                        src+=this->_core->_src_step1,
                                                        dst1+=this->_core->_dst1_step1,
                                                        dst2+=this->_core->_dst2_step1)
        {
            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            std::size_t c=0;

            for(;c<stop; c+=step,
                                       it_src+=step,
                                       it_dst1+=step,
                                       it_dst2+=step)
            {
#if CV_AVX2
                __m256i I = _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(it_src));

                __m256i cmp = _mm256_cmpgt_epi8(I,zeros);

                I = _mm256_and_si256(cmp,mask);
                __m256i Ic = _mm256_andnot_si256(cmp,mask);

                I = _mm256_or_si256(I,Ic);
                I = _mm256_andnot_si256(I,mask);

                _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst1),I);
                _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst2),Ic);

#else

#if CV_SSE3
                __m128i I = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(it_src));
#else
                __m128i I = _mm_loadu_si128(reinterpret_cast<const __m128i*>(it_src));
#endif

                __m128i cmp = _mm_cmpgt_epi8(I,zeros);

                I = _mm_and_si128(cmp,mask);
                __m128i Ic = _mm_andnot_si128(cmp,mask);

                I = _mm_or_si128(I,Ic);
                I = _mm_andnot_si128(I,mask);

                _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst1),I);
                _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst2),Ic);
#endif
            }

            for(;c<this->_core->_cols; c++, it_src++, it_dst1++, it_dst2++)
            {
                value_type I = *it_src > 0 ? 0x0 : 0xFF;
                value_type Ic = ~I;

                I = ~(I | Ic);

                *it_dst1 = I;
                *it_dst2 = Ic;
            }
        }
    }
}

template<>
void step1_as_is<schar>::operator()(const tbb::blocked_range2d<std::size_t>& range)const
{

    const tbb::blocked_range<std::size_t>& rows = range.rows();
    const tbb::blocked_range<std::size_t>& tmp = range.cols();

    tbb::blocked_range<std::size_t> cols(tmp.begin() * this->_core->_grain, std::min( tmp.end() * this->_core->_grain, this->_core->_cols));

    const_pointer src = this->_core->_src + rows.begin() * this->_core->_src_step1 + cols.begin();
    pointer dst1 = this->_core->_dst1 + rows.begin() * this->_core->_dst1_step1 + cols.begin();
    pointer dst2 = this->_core->_dst2 + rows.begin() * this->_core->_dst2_step1 + cols.begin();

#if CV_AVX2
    static const std::size_t step = 32;
    static const __m256i mask = _mm256_set1_epi8(0xFF);
    static const __m256i zeros = _mm256_setzero_si256();
#else
    static const std::size_t step = 16;
    static const __m128i mask = _mm_set1_epi8(0xFF);
    static const __m128i zeros = _mm_setzero_si128();
#endif


    if(this->_core->_is_algn)
    {
        for(std::size_t r=rows.begin();r<rows.end();r++,src+=this->_core->_src_step1,dst1+=this->_core->_dst1_step1,dst2+=this->_core->_dst2_step1)
        {

            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            for(std::size_t c=0;c<cols.size();c+=step,it_src+=step,it_dst1+=step,it_dst2+=step)
            {
#if CV_AVX2
                __m256i I = _mm256_load_si256(reinterpret_cast<const __m256i*>(it_src));

                __m256i cmp = _mm256_cmpgt_epi8(I,zeros);

                I = _mm256_and_si256(cmp,mask);
                __m256i Ic = _mm256_andnot_si256(cmp,mask);

                I = _mm256_or_si256(I,Ic);
                I = _mm256_andnot_si256(I,mask);

                _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst1),I);
                _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst2),Ic);

#else

                __m128i I = _mm_load_si128(reinterpret_cast<const __m128i*>(it_src));

                __m128i cmp = _mm_cmpgt_epi8(I,zeros);

                I = _mm_and_si128(cmp,mask);
                __m128i Ic = _mm_andnot_si128(cmp,mask);

                I = _mm_or_si128(I,Ic);
                I = _mm_andnot_si128(I,mask);

                _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst1),I);
                _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst2),Ic);
#endif
            }
        }
    }
    else
    {


        const std::size_t stop = cols.size() - (cols.size()%step);

        for(std::size_t r=rows.begin(); r<rows.end(); r++,
                                                      src+=this->_core->_src_step1,
                                                      dst1+=this->_core->_dst1_step1,
                                                      dst2+=this->_core->_dst2_step1)
        {
            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            std::size_t c=0;

            for(;c<stop; c+=step,
                         it_src+=step,
                         it_dst1+=step,
                         it_dst2+=step)
            {
#if CV_AVX2
                __m256i I = _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(it_src));

                __m256i cmp = _mm256_cmpgt_epi8(I,zeros);

                I = _mm256_and_si256(cmp,mask);
                __m256i Ic = _mm256_andnot_si256(cmp,mask);

                I = _mm256_or_si256(I,Ic);
                I = _mm256_andnot_si256(I,mask);

                _mm256_storeu_si256(reinterpret_cast<__m256i*>(it_dst1),I);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(it_dst2),Ic);

#else

#if CV_SSE3
                __m128i I = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(it_src));
#else
                __m128i I = _mm_load_si128(reinterpret_cast<const __m128i*>(it_src));
#endif
                __m128i cmp = _mm_cmpgt_epi8(I,zeros);

                I = _mm_and_si128(cmp,mask);
                __m128i Ic = _mm_andnot_si128(cmp,mask);

                I = _mm_or_si128(I,Ic);
                I = _mm_andnot_si128(I,mask);


                _mm_storeu_si128(reinterpret_cast<__m128i*>(it_dst1),I);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(it_dst2),Ic);
#endif
            }

            for(;c<cols.size();c++,it_src++,it_dst1++,it_dst2++)
            {
                value_type I = *it_src > 0 ? 0x0 : 0xFF;
                value_type Ic = ~I;

                I = ~(I | Ic);

                *it_dst1 = I;
                *it_dst2 = Ic;
            }
        }

    }


}


template<>
void step1_as_is<ushort>::operator()(const tbb::blocked_range<std::size_t>& tmp)const
{
    tbb::blocked_range<std::size_t> range(tmp.begin() * this->_core->_grain,std::min( tmp.end() * this->_core->_grain,this->_core->_cols ) );


    mtx.lock();
    std::cout<<"CHECK "<<range.begin()<<" "<<range.end()<<" "<<this->_core->_cols<<" "<<this->_core->_is_inline<<" "<<this->_core->_is_algn<<std::endl;
    mtx.unlock();


    const_pointer src = this->_core->_src + range.begin() * this->_core->_src_step1;
    pointer dst1 = this->_core->_dst1 + range.begin() * this->_core->_dst1_step1;
    pointer dst2 = this->_core->_dst2 + range.begin() * this->_core->_dst2_step1;

    static const std::size_t step = MyBase::core_t::steps;

#if CV_AVX2
//    static const std::size_t step = 0x10;
    static const __m256i mask = _mm256_set1_epi16(0xFFFF);
    static const __m256i zeros = _mm256_setzero_si256();
#else
//    static const std::size_t step = 0x08;
    static const __m128i mask = _mm_set1_epi16(0xFFFF);
    static const __m128i zeros = _mm_setzero_si128();
#endif

    if(this->_core->_is_algn)
    {

        if(!this->_core->_is_inline)
        {
            for(std::size_t r=range.begin(); r<range.end(); r++,
                src+=this->_core->_src_step1,
                dst1+=this->_core->_dst1_step1,
                dst2+=this->_core->_dst2_step1)
            {
                const_pointer it_src = src;
                pointer it_dst1 = dst1;
                pointer it_dst2 = dst2;

                for(std::size_t c=0; c<this->_core->_cols; c+=step,
                    it_src+=step,
                    it_dst1+=step,
                    it_dst2+=step)
                {
#if CV_AVX2
                    __m256i I = _mm256_load_si256(reinterpret_cast<const __m256i*>(it_src));

                    __m256i cmp = _mm256_cmpgt_epi16(I,zeros);

                    I = _mm256_and_si256(cmp,mask);
                    __m256i Ic = _mm256_andnot_si256(cmp,mask);

                    I = _mm256_or_si256(I,Ic);
                    I = _mm256_andnot_si256(I,mask);

                    _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst1),I);
                    _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst2),Ic);

#else

                    __m128i I = _mm_load_si128(reinterpret_cast<const __m128i*>(it_src));

                    __m128i cmp = _mm_cmpgt_epi16(I,zeros);

                    I = _mm_and_si128(cmp,mask);
                    __m128i Ic = _mm_andnot_si128(cmp,mask);

                    I = _mm_or_si128(I,Ic);
                    I = _mm_andnot_si128(I,mask);

                    _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst1),I);
                    _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst2),Ic);
#endif
                }
            }
        }
        else
        {
            for(std::size_t c=0; c<this->_core->_cols; c+=step,
                src+=step,
                dst1+=step,
                dst2+=step)
            {
#if CV_AVX2
                __m256i I = _mm256_load_si256(reinterpret_cast<const __m256i*>(src));

                __m256i cmp = _mm256_cmpgt_epi16(I,zeros);

                I = _mm256_and_si256(cmp,mask);
                __m256i Ic = _mm256_andnot_si256(cmp,mask);

                I = _mm256_or_si256(I,Ic);
                I = _mm256_andnot_si256(I,mask);

                _mm256_stream_si256(reinterpret_cast<__m256i*>(dst1),I);
                _mm256_stream_si256(reinterpret_cast<__m256i*>(dst2),Ic);

#else

                __m128i I = _mm_load_si128(reinterpret_cast<const __m128i*>(src));

                __m128i cmp = _mm_cmpgt_epi16(I,zeros);

                I = _mm_and_si128(cmp,mask);
                __m128i Ic = _mm_andnot_si128(cmp,mask);

                I = _mm_or_si128(I,Ic);
                I = _mm_andnot_si128(I,mask);

                _mm_stream_si128(reinterpret_cast<__m128i*>(dst1),I);
                _mm_stream_si128(reinterpret_cast<__m128i*>(dst2),Ic);
#endif
            }
        }
    }
    else
    {



        if(!this->_core->_is_inline)
        {

            const std::size_t stop = this->_core->_cols - (this->_core->_cols%step);

            for(std::size_t r=range.begin(); r<range.end(); r++,
                src+=this->_core->_src_step1,
                dst1+=this->_core->_dst1_step1,
                dst2+=this->_core->_dst2_step1)
            {
                const_pointer it_src = src;
                pointer it_dst1 = dst1;
                pointer it_dst2 = dst2;

                std::size_t c=0;

                for(;c<stop; c+=step,
                    it_src+=step,
                    it_dst1+=step,
                    it_dst2+=step)
                {
#if CV_AVX2
                    __m256i I = _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(it_src));

                    __m256i cmp = _mm256_cmpgt_epi16(I,zeros);

                    I = _mm256_and_si256(cmp,mask);
                    __m256i Ic = _mm256_andnot_si256(cmp,mask);

                    I = _mm256_or_si256(I,Ic);
                    I = _mm256_andnot_si256(I,mask);

                    _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst1),I);
                    _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst2),Ic);

#else

#if CV_SSE3
                    __m128i I = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(it_src));
#else
                    __m128i I = _mm_loadu_si128(reinterpret_cast<const __m128i*>(it_src));
#endif

                    __m128i cmp = _mm_cmpgt_epi16(I,zeros);

                    I = _mm_and_si128(cmp,mask);
                    __m128i Ic = _mm_andnot_si128(cmp,mask);

                    I = _mm_or_si128(I,Ic);
                    I = _mm_andnot_si128(I,mask);

                    _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst1),I);
                    _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst2),Ic);
#endif
                }

                for(;c<this->_core->_cols; c++, it_src++, it_dst1++, it_dst2++)
                {
                    value_type I = *it_src > 0 ? 0x0 : 0xFFFF;
                    value_type Ic = ~I;

                    I = ~(I | Ic);

                    *it_dst1 = I;
                    *it_dst2 = Ic;
                }
            }
        }
        else
        {

            const std::size_t stop = range.size() - (range.size()%step);

                std::size_t c=0;

                for(;c<stop; c+=step,
                    src+=step,
                    dst1+=step,
                    dst2+=step)
                {
#if CV_AVX2
                    __m256i I = _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(src));

                    __m256i cmp = _mm256_cmpgt_epi16(I,zeros);

                    I = _mm256_and_si256(cmp,mask);
                    __m256i Ic = _mm256_andnot_si256(cmp,mask);

                    I = _mm256_or_si256(I,Ic);
                    I = _mm256_andnot_si256(I,mask);

                    _mm256_stream_si256(reinterpret_cast<__m256i*>(dst1),I);
                    _mm256_stream_si256(reinterpret_cast<__m256i*>(dst2),Ic);

#else

#if CV_SSE3
                    __m128i I = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(src));
#else
                    __m128i I = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
#endif

                    __m128i cmp = _mm_cmpgt_epi16(I,zeros);

                    I = _mm_and_si128(cmp,mask);
                    __m128i Ic = _mm_andnot_si128(cmp,mask);

                    I = _mm_or_si128(I,Ic);
                    I = _mm_andnot_si128(I,mask);

                    _mm_stream_si128(reinterpret_cast<__m128i*>(dst1),I);
                    _mm_stream_si128(reinterpret_cast<__m128i*>(dst2),Ic);
#endif
                }

                for(;c<range.size(); c++, src++, dst1++, dst2++)
                {
                    value_type I = *src > 0 ? 0x0 : 0xFFFF;
                    value_type Ic = ~I;

                    I = ~(I | Ic);

                    *dst1 = I;
                    *dst2 = Ic;
                }


        }


    }



}

template<>
void step1_as_is<ushort>::operator()(const tbb::blocked_range2d<std::size_t>& range)const
{

    const tbb::blocked_range<std::size_t>& rows = range.rows();
    const tbb::blocked_range<std::size_t>& tmp = range.cols();

    tbb::blocked_range<std::size_t> cols(tmp.begin() * this->_core->_grain, std::min( tmp.end() * this->_core->_grain, this->_core->_cols));

    const_pointer src = this->_core->_src + rows.begin() * this->_core->_src_step1 + cols.begin();
    pointer dst1 = this->_core->_dst1 + rows.begin() * this->_core->_dst1_step1 + cols.begin();
    pointer dst2 = this->_core->_dst2 + rows.begin() * this->_core->_dst2_step1 + cols.begin();

#if CV_AVX2
    static const std::size_t step = 16;
    static const __m256i mask = _mm256_set1_epi16(0xFFFF);
    static const __m256i zeros = _mm256_setzero_si256();
#else
    static const std::size_t step = 8;
    static const __m128i mask = _mm_set1_epi16(0xFFFF);
    static const __m128i zeros = _mm_setzero_si128();
#endif


    if(this->_core->_is_algn)
    {
        for(std::size_t r=rows.begin();r<rows.end();r++,src+=this->_core->_src_step1,dst1+=this->_core->_dst1_step1,dst2+=this->_core->_dst2_step1)
        {

            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            for(std::size_t c=0;c<cols.size();c+=step,it_src+=step,it_dst1+=step,it_dst2+=step)
            {
#if CV_AVX2
                __m256i I = _mm256_load_si256(reinterpret_cast<const __m256i*>(it_src));

                __m256i cmp = _mm256_cmpgt_epi16(I,zeros);

                I = _mm256_and_si256(cmp,mask);
                __m256i Ic = _mm256_andnot_si256(cmp,mask);

                I = _mm256_or_si256(I,Ic);
                I = _mm256_andnot_si256(I,mask);

                _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst1),I);
                _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst2),Ic);

#else

                __m128i I = _mm_load_si128(reinterpret_cast<const __m128i*>(it_src));

                __m128i cmp = _mm_cmpgt_epi16(I,zeros);

                I = _mm_and_si128(cmp,mask);
                __m128i Ic = _mm_andnot_si128(cmp,mask);

                I = _mm_or_si128(I,Ic);
                I = _mm_andnot_si128(I,mask);

                _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst1),I);
                _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst2),Ic);
#endif
            }
        }
    }
    else
    {


        const std::size_t stop = cols.size() - (cols.size()%step);

        for(std::size_t r=rows.begin(); r<rows.end(); r++,
                                                      src+=this->_core->_src_step1,
                                                      dst1+=this->_core->_dst1_step1,
                                                      dst2+=this->_core->_dst2_step1)
        {
            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            std::size_t c=0;

            for(;c<stop; c+=step,
                         it_src+=step,
                         it_dst1+=step,
                         it_dst2+=step)
            {
#if CV_AVX2
                __m256i I = _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(it_src));

                __m256i cmp = _mm256_cmpgt_epi16(I,zeros);

                I = _mm256_and_si256(cmp,mask);
                __m256i Ic = _mm256_andnot_si256(cmp,mask);

                I = _mm256_or_si256(I,Ic);
                I = _mm256_andnot_si256(I,mask);

                _mm256_storeu_si256(reinterpret_cast<__m256i*>(it_dst1),I);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(it_dst2),Ic);

#else

#if CV_SSE3
                __m128i I = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(it_src));
#else
                __m128i I = _mm_load_si128(reinterpret_cast<const __m128i*>(it_src));
#endif
                __m128i cmp = _mm_cmpgt_epi16(I,zeros);

                I = _mm_and_si128(cmp,mask);
                __m128i Ic = _mm_andnot_si128(cmp,mask);

                I = _mm_or_si128(I,Ic);
                I = _mm_andnot_si128(I,mask);


                _mm_storeu_si128(reinterpret_cast<__m128i*>(it_dst1),I);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(it_dst2),Ic);
#endif
            }

            for(;c<cols.size();c++,it_src++,it_dst1++,it_dst2++)
            {
                value_type I = *it_src > 0 ? 0x0 : 0xFFFF;
                value_type Ic = ~I;

                I = ~(I | Ic);

                *it_dst1 = I;
                *it_dst2 = Ic;
            }
        }

    }


}


template<>
void step1_as_is<short>::operator()(const tbb::blocked_range<std::size_t>& tmp)const
{
    tbb::blocked_range<std::size_t> range(tmp.begin() * this->_core->_grain,std::min( tmp.end() * this->_core->_grain,this->_core->_cols ) );

    const_pointer src = this->_core->_src + range.begin() * this->_core->_src_step1;
    pointer dst1 = this->_core->_dst1 + range.begin() * this->_core->_dst1_step1;
    pointer dst2 = this->_core->_dst2 + range.begin() * this->_core->_dst2_step1;


#if CV_AVX2
    static const std::size_t step = 16;
    static const __m256i mask = _mm256_set1_epi16(0xFFFF);
    static const __m256i zeros = _mm256_setzero_si256();
#else
    static const std::size_t step = 8;
    static const __m128i mask = _mm_set1_epi16(0xFFFF);
    static const __m128i zeros = _mm_setzero_si128();
#endif

    if(this->_core->_is_algn)
    {
        for(std::size_t r=range.begin(); r<range.end(); r++,
                                                        src+=this->_core->_src_step1,
                                                        dst1+=this->_core->_dst1_step1,
                                                        dst2+=this->_core->_dst2_step1)
        {
            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            for(std::size_t c=0; c<this->_core->_cols; c+=step,
                                                       it_src+=step,
                                                       it_dst1+=step,
                                                       it_dst2+=step)
            {
#if CV_AVX2
                __m256i I = _mm256_load_si256(reinterpret_cast<const __m256i*>(it_src));

                __m256i cmp = _mm256_cmpgt_epi16(I,zeros);

                I = _mm256_and_si256(cmp,mask);
                __m256i Ic = _mm256_andnot_si256(cmp,mask);

                I = _mm256_or_si256(I,Ic);
                I = _mm256_andnot_si256(I,mask);

                _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst1),I);
                _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst2),Ic);

#else

                __m128i I = _mm_load_si128(reinterpret_cast<const __m128i*>(it_src));

                __m128i cmp = _mm_cmpgt_epi16(I,zeros);

                I = _mm_and_si128(cmp,mask);
                __m128i Ic = _mm_andnot_si128(cmp,mask);

                I = _mm_or_si128(I,Ic);
                I = _mm_andnot_si128(I,mask);

                _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst1),I);
                _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst2),Ic);
#endif
            }
        }
    }
    else
    {

        const std::size_t stop = this->_core->_cols - (this->_core->_cols%step);

        for(std::size_t r=range.begin(); r<range.end(); r++,
                                                        src+=this->_core->_src_step1,
                                                        dst1+=this->_core->_dst1_step1,
                                                        dst2+=this->_core->_dst2_step1)
        {
            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            std::size_t c=0;

            for(;c<stop; c+=step,
                                       it_src+=step,
                                       it_dst1+=step,
                                       it_dst2+=step)
            {
#if CV_AVX2
                __m256i I = _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(it_src));

                __m256i cmp = _mm256_cmpgt_epi16(I,zeros);

                I = _mm256_and_si256(cmp,mask);
                __m256i Ic = _mm256_andnot_si256(cmp,mask);

                I = _mm256_or_si256(I,Ic);
                I = _mm256_andnot_si256(I,mask);

                _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst1),I);
                _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst2),Ic);

#else

#if CV_SSE3
                __m128i I = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(it_src));
#else
                __m128i I = _mm_loadu_si128(reinterpret_cast<const __m128i*>(it_src));
#endif

                __m128i cmp = _mm_cmpgt_epi16(I,zeros);

                I = _mm_and_si128(cmp,mask);
                __m128i Ic = _mm_andnot_si128(cmp,mask);

                I = _mm_or_si128(I,Ic);
                I = _mm_andnot_si128(I,mask);

                _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst1),I);
                _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst2),Ic);
#endif
            }

            for(;c<this->_core->_cols; c++, it_src++, it_dst1++, it_dst2++)
            {
                value_type I = *it_src > 0 ? 0x0 : 0xFFFF;
                value_type Ic = ~I;

                I = ~(I | Ic);

                *it_dst1 = I;
                *it_dst2 = Ic;
            }
        }


    }



}

template<>
void step1_as_is<short>::operator()(const tbb::blocked_range2d<std::size_t>& range)const
{

    const tbb::blocked_range<std::size_t>& rows = range.rows();
    const tbb::blocked_range<std::size_t>& tmp = range.cols();

    tbb::blocked_range<std::size_t> cols(tmp.begin() * this->_core->_grain, std::min( tmp.end() * this->_core->_grain, this->_core->_cols));

    const_pointer src = this->_core->_src + rows.begin() * this->_core->_src_step1 + cols.begin();
    pointer dst1 = this->_core->_dst1 + rows.begin() * this->_core->_dst1_step1 + cols.begin();
    pointer dst2 = this->_core->_dst2 + rows.begin() * this->_core->_dst2_step1 + cols.begin();

#if CV_AVX2
    static const std::size_t step = 16;
    static const __m256i mask = _mm256_set1_epi16(0xFFFF);
    static const __m256i zeros = _mm256_setzero_si256();
#else
    static const std::size_t step = 8;
    static const __m128i mask = _mm_set1_epi16(0xFFFF);
    static const __m128i zeros = _mm_setzero_si128();
#endif


    if(this->_core->_is_algn)
    {
        for(std::size_t r=rows.begin();r<rows.end();r++,src+=this->_core->_src_step1,dst1+=this->_core->_dst1_step1,dst2+=this->_core->_dst2_step1)
        {

            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            for(std::size_t c=0;c<cols.size();c+=step,it_src+=step,it_dst1+=step,it_dst2+=step)
            {
#if CV_AVX2
                __m256i I = _mm256_load_si256(reinterpret_cast<const __m256i*>(it_src));

                __m256i cmp = _mm256_cmpgt_epi16(I,zeros);

                I = _mm256_and_si256(cmp,mask);
                __m256i Ic = _mm256_andnot_si256(cmp,mask);

                I = _mm256_or_si256(I,Ic);
                I = _mm256_andnot_si256(I,mask);

                _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst1),I);
                _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst2),Ic);

#else

                __m128i I = _mm_load_si128(reinterpret_cast<const __m128i*>(it_src));

                __m128i cmp = _mm_cmpgt_epi16(I,zeros);

                I = _mm_and_si128(cmp,mask);
                __m128i Ic = _mm_andnot_si128(cmp,mask);

                I = _mm_or_si128(I,Ic);
                I = _mm_andnot_si128(I,mask);

                _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst1),I);
                _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst2),Ic);
#endif
            }
        }
    }
    else
    {


        const std::size_t stop = cols.size() - (cols.size()%step);

        for(std::size_t r=rows.begin(); r<rows.end(); r++,
                                                      src+=this->_core->_src_step1,
                                                      dst1+=this->_core->_dst1_step1,
                                                      dst2+=this->_core->_dst2_step1)
        {
            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            std::size_t c=0;

            for(;c<stop; c+=step,
                         it_src+=step,
                         it_dst1+=step,
                         it_dst2+=step)
            {
#if CV_AVX2
                __m256i I = _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(it_src));

                __m256i cmp = _mm256_cmpgt_epi16(I,zeros);

                I = _mm256_and_si256(cmp,mask);
                __m256i Ic = _mm256_andnot_si256(cmp,mask);

                I = _mm256_or_si256(I,Ic);
                I = _mm256_andnot_si256(I,mask);

                _mm256_storeu_si256(reinterpret_cast<__m256i*>(it_dst1),I);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(it_dst2),Ic);

#else

#if CV_SSE3
                __m128i I = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(it_src));
#else
                __m128i I = _mm_load_si128(reinterpret_cast<const __m128i*>(it_src));
#endif
                __m128i cmp = _mm_cmpgt_epi16(I,zeros);

                I = _mm_and_si128(cmp,mask);
                __m128i Ic = _mm_andnot_si128(cmp,mask);

                I = _mm_or_si128(I,Ic);
                I = _mm_andnot_si128(I,mask);


                _mm_storeu_si128(reinterpret_cast<__m128i*>(it_dst1),I);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(it_dst2),Ic);
#endif
            }

            for(;c<cols.size();c++,it_src++,it_dst1++,it_dst2++)
            {
                value_type I = *it_src > 0 ? 0x0 : 0xFFFF;
                value_type Ic = ~I;

                I = ~(I | Ic);

                *it_dst1 = I;
                *it_dst2 = Ic;
            }
        }

    }


}


template<>
void step1_as_is<int>::operator()(const tbb::blocked_range<std::size_t>& tmp)const
{
    tbb::blocked_range<std::size_t> range(tmp.begin() * this->_core->_grain,std::min( tmp.end() * this->_core->_grain,this->_core->_cols ) );

    const_pointer src = this->_core->_src + range.begin() * this->_core->_src_step1;
    pointer dst1 = this->_core->_dst1 + range.begin() * this->_core->_dst1_step1;
    pointer dst2 = this->_core->_dst2 + range.begin() * this->_core->_dst2_step1;

#if CV_AVX2
    static const std::size_t step = 8;
    static const __m256i mask = _mm256_set1_epi32(0xFFFFFFFF);
    static const __m256i zeros = _mm256_setzero_si256();
#else
    static const std::size_t step = 4;
    static const __m128i mask = _mm_set1_epi32(0xFFFFFFFF);
    static const __m128i zeros = _mm_setzero_si128();
#endif

#if CV_SSE2
    if(this->_core->_is_algn)
    {
        for(std::size_t r=range.begin(); r<range.end(); r++,
                                                        src+=this->_core->_src_step1,
                                                        dst1+=this->_core->_dst1_step1,
                                                        dst2+=this->_core->_dst2_step1)
        {
            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            for(std::size_t c=0; c<this->_core->_cols; c+=step,
                                                       it_src+=step,
                                                       it_dst1+=step,
                                                       it_dst2+=step)
            {
#if CV_AVX2
                __m256i I = _mm256_load_si256(reinterpret_cast<const __m256i*>(it_src));

                __m256i cmp = _mm256_cmpgt_epi16(I,zeros);

                I = _mm256_and_si256(cmp,mask);
                __m256i Ic = _mm256_andnot_si256(cmp,mask);

                I = _mm256_or_si256(I,Ic);
                I = _mm256_andnot_si256(I,mask);

                _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst1),I);
                _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst2),Ic);

#else

                __m128i I = _mm_load_si128(reinterpret_cast<const __m128i*>(it_src));

                __m128i cmp = _mm_cmpgt_epi16(I,zeros);

                I = _mm_and_si128(cmp,mask);
                __m128i Ic = _mm_andnot_si128(cmp,mask);

                I = _mm_or_si128(I,Ic);
                I = _mm_andnot_si128(I,mask);

                _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst1),I);
                _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst2),Ic);
#endif
            }
        }
    }
    else
    {
#endif
        const std::size_t stop = this->_core->_cols - (this->_core->_cols%step);

        for(std::size_t r=range.begin(); r<range.end(); r++,
                                                        src+=this->_core->_src_step1,
                                                        dst1+=this->_core->_dst1_step1,
                                                        dst2+=this->_core->_dst2_step1)
        {
            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            std::size_t c=0;

            for(;c<stop; c+=step,
                                       it_src+=step,
                                       it_dst1+=step,
                                       it_dst2+=step)
            {
#if CV_AVX2
                __m256i I = _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(it_src));

                __m256i cmp = _mm256_cmpgt_epi16(I,zeros);

                I = _mm256_and_si256(cmp,mask);
                __m256i Ic = _mm256_andnot_si256(cmp,mask);

                I = _mm256_or_si256(I,Ic);
                I = _mm256_andnot_si256(I,mask);

                _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst1),I);
                _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst2),Ic);

#else

#if CV_SSE3
                __m128i I = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(it_src));
#else
                __m128i I = _mm_loadu_si128(reinterpret_cast<const __m128i*>(it_src));
#endif

                __m128i cmp = _mm_cmpgt_epi16(I,zeros);

                I = _mm_and_si128(cmp,mask);
                __m128i Ic = _mm_andnot_si128(cmp,mask);

                I = _mm_or_si128(I,Ic);
                I = _mm_andnot_si128(I,mask);

                _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst1),I);
                _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst2),Ic);
#endif
            }

            for(;c<this->_core->_cols; c++, it_src++, it_dst1++, it_dst2++)
            {
                value_type I = *it_src > 0 ? 0x0 : 0xFFFFFFFF;
                value_type Ic = ~I;

                I = ~(I | Ic);

                *it_dst1 = I;
                *it_dst2 = Ic;
            }
        }

#if CV_SSE2
    }
#endif


}

template<>
void step1_as_is<int>::operator()(const tbb::blocked_range2d<std::size_t>& range)const
{

    const tbb::blocked_range<std::size_t>& rows = range.rows();
    const tbb::blocked_range<std::size_t>& tmp = range.cols();

    tbb::blocked_range<std::size_t> cols(tmp.begin() * this->_core->_grain, std::min( tmp.end() * this->_core->_grain, this->_core->_cols));

    const_pointer src = this->_core->_src + rows.begin() * this->_core->_src_step1 + cols.begin();
    pointer dst1 = this->_core->_dst1 + rows.begin() * this->_core->_dst1_step1 + cols.begin();
    pointer dst2 = this->_core->_dst2 + rows.begin() * this->_core->_dst2_step1 + cols.begin();

#if CV_AVX2
    static const std::size_t step = 8;
    static const __m256i mask = _mm256_set1_epi32(0xFFFFFFFF);
    static const __m256i zeros = _mm256_setzero_si256();
#else
    static const std::size_t step = 4;
    static const __m128i mask = _mm_set1_epi32(0xFFFFFFFF);
    static const __m128i zeros = _mm_setzero_si128();
#endif


    if(this->_core->_is_algn)
    {
        for(std::size_t r=rows.begin();r<rows.end();r++,src+=this->_core->_src_step1,dst1+=this->_core->_dst1_step1,dst2+=this->_core->_dst2_step1)
        {

            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            for(std::size_t c=0;c<cols.size();c+=step,it_src+=step,it_dst1+=step,it_dst2+=step)
            {
#if CV_AVX2
                __m256i I = _mm256_load_si256(reinterpret_cast<const __m256i*>(it_src));

                __m256i cmp = _mm256_cmpgt_epi32(I,zeros);

                I = _mm256_and_si256(cmp,mask);
                __m256i Ic = _mm256_andnot_si256(cmp,mask);

                I = _mm256_or_si256(I,Ic);
                I = _mm256_andnot_si256(I,mask);

                _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst1),I);
                _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst2),Ic);

#else

                __m128i I = _mm_load_si128(reinterpret_cast<const __m128i*>(it_src));

                __m128i cmp = _mm_cmpgt_epi32(I,zeros);

                I = _mm_and_si128(cmp,mask);
                __m128i Ic = _mm_andnot_si128(cmp,mask);

                I = _mm_or_si128(I,Ic);
                I = _mm_andnot_si128(I,mask);

                _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst1),I);
                _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst2),Ic);
#endif
            }
        }
    }
    else
    {


        const std::size_t stop = cols.size() - (cols.size()%step);

        for(std::size_t r=rows.begin(); r<rows.end(); r++,
                                                      src+=this->_core->_src_step1,
                                                      dst1+=this->_core->_dst1_step1,
                                                      dst2+=this->_core->_dst2_step1)
        {
            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            std::size_t c=0;

            for(;c<stop; c+=step,
                         it_src+=step,
                         it_dst1+=step,
                         it_dst2+=step)
            {
#if CV_AVX2
                __m256i I = _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(it_src));

                __m256i cmp = _mm256_cmpgt_epi32(I,zeros);

                I = _mm256_and_si256(cmp,mask);
                __m256i Ic = _mm256_andnot_si256(cmp,mask);

                I = _mm256_or_si256(I,Ic);
                I = _mm256_andnot_si256(I,mask);

                _mm256_storeu_si256(reinterpret_cast<__m256i*>(it_dst1),I);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(it_dst2),Ic);

#else

#if CV_SSE3
                __m128i I = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(it_src));
#else
                __m128i I = _mm_load_si128(reinterpret_cast<const __m128i*>(it_src));
#endif
                __m128i cmp = _mm_cmpgt_epi32(I,zeros);

                I = _mm_and_si128(cmp,mask);
                __m128i Ic = _mm_andnot_si128(cmp,mask);

                I = _mm_or_si128(I,Ic);
                I = _mm_andnot_si128(I,mask);


                _mm_storeu_si128(reinterpret_cast<__m128i*>(it_dst1),I);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(it_dst2),Ic);
#endif
            }

            for(;c<cols.size();c++,it_src++,it_dst1++,it_dst2++)
            {
                value_type I = *it_src > 0 ? 0x0 : 0xFFFFFFFF;
                value_type Ic = ~I;

                I = ~(I | Ic);

                *it_dst1 = I;
                *it_dst2 = Ic;
            }
        }

    }


}


template<>
void step1_as_is<float>::operator()(const tbb::blocked_range<std::size_t>& tmp)const
{
    tbb::blocked_range<std::size_t> range(tmp.begin() * this->_core->_grain,std::min( tmp.end() * this->_core->_grain,this->_core->_cols ) );

    const_pointer src = this->_core->_src + range.begin() * this->_core->_src_step1;
    pointer dst1 = this->_core->_dst1 + range.begin() * this->_core->_dst1_step1;
    pointer dst2 = this->_core->_dst2 + range.begin() * this->_core->_dst2_step1;

#if CV_AVX2
    static const std::size_t step = 8;
    static const __m256 mask = _mm256_set1_ps(0xFFFFFFFF);
    static const __m256 zeros = _mm256_setzero_ps();
#else
    static const std::size_t step = 4;
    static const __m128 mask = _mm_set1_ps(0xFFFFFFFF);
    static const __m128 zeros = _mm_setzero_ps();
#endif

#if CV_SSE2
    if(this->_core->_is_algn)
    {
        for(std::size_t r=range.begin(); r<range.end(); r++,
                                                        src+=this->_core->_src_step1,
                                                        dst1+=this->_core->_dst1_step1,
                                                        dst2+=this->_core->_dst2_step1)
        {
            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            for(std::size_t c=0; c<this->_core->_cols; c+=step,
                                                       it_src+=step,
                                                       it_dst1+=step,
                                                       it_dst2+=step)
            {
#if CV_AVX2
                __m256 I = _mm256_load_ps(it_src);

                __m256 cmp = _mm256_cmp_ps(I,zeros,_CMP_GT_OQ);

                I = _mm256_and_ps(cmp,mask);
                __m256 Ic = _mm256_andnot_ps(cmp,mask);

                I = _mm256_or_ps(I,Ic);
                I = _mm256_andnot_ps(I,mask);

                _mm256_stream_ps(it_dst1,I);
                _mm256_stream_ps(it_dst2,Ic);

#else

                __m128 I = _mm_load_ps(it_src);

                __m128 cmp = _mm_cmpgt_ps(I,zeros);

                I = _mm_and_ps(cmp,mask);
                __m128 Ic = _mm_andnot_ps(cmp,mask);

                I = _mm_or_ps(I,Ic);
                I = _mm_andnot_ps(I,mask);

                _mm_stream_ps(it_dst1,I);
                _mm_stream_ps(it_dst2,Ic);
#endif
            }
        }
    }
    else
    {
#endif
        const std::size_t stop = this->_core->_cols - (this->_core->_cols%step);

        for(std::size_t r=range.begin(); r<range.end(); r++,
                                                        src+=this->_core->_src_step1,
                                                        dst1+=this->_core->_dst1_step1,
                                                        dst2+=this->_core->_dst2_step1)
        {
            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            std::size_t c=0;

            for(;c<stop; c+=step,
                                       it_src+=step,
                                       it_dst1+=step,
                                       it_dst2+=step)
            {
#if CV_AVX2
                __m256 I = _mm256_loadu_ps(it_src);

                __m256 cmp = _mm256_cmp_ps(I,zeros,_CMP_GT_OQ);

                I = _mm256_and_ps(cmp,mask);
                __m256 Ic = _mm256_andnot_ps(cmp,mask);

                I = _mm256_or_ps(I,Ic);
                I = _mm256_andnot_ps(I,mask);

                _mm256_storeu_ps(it_dst1,I);
                _mm256_storeu_ps(it_dst2,Ic);

#else

                __m128 I = _mm_loadu_ps(it_src);

                __m128 cmp = _mm_cmpgt_ps(I,zeros);

                I = _mm_and_ps(cmp,mask);
                __m128 Ic = _mm_andnot_ps(cmp,mask);

                I = _mm_or_ps(I,Ic);
                I = _mm_andnot_ps(I,mask);


                _mm_storeu_ps(it_dst1,I);
                _mm_storeu_ps(it_dst2,Ic);
#endif
            }

            for(;c<this->_core->_cols; c++, it_src++, it_dst1++, it_dst2++)
            {
                uint I = *it_src > 0 ? 0x0 : 0xFFFFFFFF;
                uint Ic = ~I;

                I = ~(I | Ic);

                *it_dst1 = I;
                *it_dst2 = Ic;
            }
        }

#if CV_SSE2
    }
#endif


}

template<>
void step1_as_is<float>::operator()(const tbb::blocked_range2d<std::size_t>& range)const
{

    const tbb::blocked_range<std::size_t>& rows = range.rows();
    const tbb::blocked_range<std::size_t>& tmp = range.cols();

    tbb::blocked_range<std::size_t> cols(tmp.begin() * this->_core->_grain, std::min( tmp.end() * this->_core->_grain, this->_core->_cols));

    const_pointer src = this->_core->_src + rows.begin() * this->_core->_src_step1 + cols.begin();
    pointer dst1 = this->_core->_dst1 + rows.begin() * this->_core->_dst1_step1 + cols.begin();
    pointer dst2 = this->_core->_dst2 + rows.begin() * this->_core->_dst2_step1 + cols.begin();

#if CV_AVX2
    static const std::size_t step = 8;
    static const __m256 mask = _mm256_set1_ps(0xFFFFFFFF);
    static const __m256 zeros = _mm256_setzero_ps();
#else
    static const std::size_t step = 4;
    static const __m128 mask = _mm_set1_ps(0xFFFFFFFF);
    static const __m128 zeros = _mm_setzero_ps();
#endif


    if(this->_core->_is_algn)
    {
        for(std::size_t r=rows.begin();r<rows.end();r++,src+=this->_core->_src_step1,dst1+=this->_core->_dst1_step1,dst2+=this->_core->_dst2_step1)
        {

            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            for(std::size_t c=0;c<cols.size();c+=step,it_src+=step,it_dst1+=step,it_dst2+=step)
            {
#if CV_AVX2
                __m256 I = _mm256_load_ps(it_src);

                __m256 cmp = _mm256_cmp_ps(I,zeros,_CMP_GT_OQ);

                I = _mm256_and_ps(cmp,mask);
                __m256 Ic = _mm256_andnot_ps(cmp,mask);

                I = _mm256_or_ps(I,Ic);
                I = _mm256_andnot_ps(I,mask);

                _mm256_stream_ps(it_dst1,I);
                _mm256_stream_ps(it_dst2,Ic);

#else

                __m128 I = _mm_load_ps(it_src);

                __m128 cmp = _mm_cmpgt_ps(I,zeros);

                I = _mm_and_ps(cmp,mask);
                __m128 Ic = _mm_andnot_ps(cmp,mask);

                I = _mm_or_ps(I,Ic);
                I = _mm_andnot_ps(I,mask);

                _mm_stream_ps(it_dst1,I);
                _mm_stream_ps(it_dst2,Ic);
#endif
            }
        }
    }
    else
    {
        const std::size_t stop = cols.size() - (cols.size()%step);

        for(std::size_t r=rows.begin(); r<rows.end(); r++,
                                                      src+=this->_core->_src_step1,
                                                      dst1+=this->_core->_dst1_step1,
                                                      dst2+=this->_core->_dst2_step1)
        {
            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            std::size_t c=0;

            for(;c<stop; c+=step,
                         it_src+=step,
                         it_dst1+=step,
                         it_dst2+=step)
            {
#if CV_AVX2
                __m256 I = _mm256_loadu_ps(it_src);

                __m256 cmp = _mm256_cmp_ps(I,zeros,_CMP_GT_OQ);

                I = _mm256_and_ps(cmp,mask);
                __m256 Ic = _mm256_andnot_ps(cmp,mask);

                I = _mm256_or_ps(I,Ic);
                I = _mm256_andnot_ps(I,mask);

                _mm256_storeu_ps(it_dst1,I);
                _mm256_storeu_ps(it_dst2,Ic);

#else

                __m128 I = _mm_loadu_ps(it_src);

                __m128 cmp = _mm_cmpgt_ps(I,zeros);

                I = _mm_and_ps(cmp,mask);
                __m128 Ic = _mm_andnot_ps(cmp,mask);

                I = _mm_or_ps(I,Ic);
                I = _mm_andnot_ps(I,mask);


                _mm_storeu_ps(it_dst1,I);
                _mm_storeu_ps(it_dst2,Ic);
#endif
            }

            for(;c<cols.size();c++,it_src++,it_dst1++,it_dst2++)
            {
                uint I = *it_src > 0 ? 0x0 : 0xFFFFFFFF;
                uint Ic = ~I;

                I = ~(I | Ic);

                *it_dst1 = I;
                *it_dst2 = Ic;
            }
        }

    }


}


template<>
void step1_as_is<double>::operator()(const tbb::blocked_range<std::size_t>& tmp)const
{
    tbb::blocked_range<std::size_t> range(tmp.begin() * this->_core->_grain,std::min( tmp.end() * this->_core->_grain,this->_core->_cols ) );

    const_pointer src = this->_core->_src + range.begin() * this->_core->_src_step1;
    pointer dst1 = this->_core->_dst1 + range.begin() * this->_core->_dst1_step1;
    pointer dst2 = this->_core->_dst2 + range.begin() * this->_core->_dst2_step1;

#if CV_AVX2
    static const std::size_t step = 4;
    static const __m256d mask = _mm256_set1_pd(0xFFFFFFFFFFFFFFFF);
    static const __m256d zeros = _mm256_setzero_pd();
#else
    static const std::size_t step = 2;
    static const __m128d mask = _mm_set1_pd(0xFFFFFFFFFFFFFFFF);
    static const __m256d zeros = _mm256_setzero_pd();
#endif

#if CV_SSE2
    if(this->_core->_is_algn)
    {
        for(std::size_t r=range.begin(); r<range.end(); r++,
                                                        src+=this->_core->_src_step1,
                                                        dst1+=this->_core->_dst1_step1,
                                                        dst2+=this->_core->_dst2_step1)
        {
            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            for(std::size_t c=0; c<this->_core->_cols; c+=step,
                                                       it_src+=step,
                                                       it_dst1+=step,
                                                       it_dst2+=step)
            {
#if CV_AVX2
                __m256d I = _mm256_load_pd(it_src);

                __m256d cmp = _mm256_cmp_pd(I,zeros,_CMP_GT_OQ);

                I = _mm256_and_pd(cmp,mask);
                __m256d Ic = _mm256_andnot_pd(cmp,mask);

                I = _mm256_or_pd(I,Ic);
                I = _mm256_andnot_pd(I,mask);

                _mm256_stream_pd(it_dst1,I);
                _mm256_stream_pd(it_dst2,Ic);

#else

                __m128d I = _mm_load_pd(it_src);

                __m128d cmp = _mm_cmpgt_pd(I,zeros);

                I = _mm_and_pd(cmp,mask);
                __m128d Ic = _mm_andnot_pd(cmp,mask);

                I = _mm_or_pd(I,Ic);
                I = _mm_andnot_pd(I,mask);

                _mm_stream_pd(it_dst1,I);
                _mm_stream_pd(it_dst2,Ic);
#endif
            }
        }
    }
    else
    {
#endif
        const std::size_t stop = this->_core->_cols - (this->_core->_cols%step);

        for(std::size_t r=range.begin(); r<range.end(); r++,
                                                        src+=this->_core->_src_step1,
                                                        dst1+=this->_core->_dst1_step1,
                                                        dst2+=this->_core->_dst2_step1)
        {
            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            std::size_t c=0;

            for(;c<stop; c+=step,
                                       it_src+=step,
                                       it_dst1+=step,
                                       it_dst2+=step)
            {
#if CV_AVX2
                __m256d I = _mm256_loadu_pd(it_src);

                __m256d cmp = _mm256_cmp_pd(I,zeros,_CMP_GT_OQ);

                I = _mm256_and_pd(cmp,mask);
                __m256d Ic = _mm256_andnot_pd(cmp,mask);

                I = _mm256_or_pd(I,Ic);
                I = _mm256_andnot_pd(I,mask);

                _mm256_storeu_pd(it_dst1,I);
                _mm256_storeu_pd(it_dst2,Ic);

#else

                __m128d I = _mm_loadu_pd(it_src);

                __m128d cmp = _mm_cmpgt_pd(I,zeros);

                I = _mm_and_pd(cmp,mask);
                __m128d Ic = _mm_andnot_pd(cmp,mask);

                I = _mm_or_pd(I,Ic);
                I = _mm_andnot_pd(I,mask);


                _mm_storeu_pd(it_dst1,I);
                _mm_storeu_pd(it_dst2,Ic);
#endif
            }

            for(;c<this->_core->_cols; c++, it_src++, it_dst1++, it_dst2++)
            {
                uint I = *it_src > 0 ? 0x0 : 0xFFFFFFFF;
                uint Ic = ~I;

                I = ~(I | Ic);

                *it_dst1 = I;
                *it_dst2 = Ic;
            }
        }

#if CV_SSE2
    }
#endif


}

template<>
void step1_as_is<double>::operator()(const tbb::blocked_range2d<std::size_t>& range)const
{

    const tbb::blocked_range<std::size_t>& rows = range.rows();
    const tbb::blocked_range<std::size_t>& tmp = range.cols();

    tbb::blocked_range<std::size_t> cols(tmp.begin() * this->_core->_grain, std::min( tmp.end() * this->_core->_grain, this->_core->_cols));

    const_pointer src = this->_core->_src + rows.begin() * this->_core->_src_step1 + cols.begin();
    pointer dst1 = this->_core->_dst1 + rows.begin() * this->_core->_dst1_step1 + cols.begin();
    pointer dst2 = this->_core->_dst2 + rows.begin() * this->_core->_dst2_step1 + cols.begin();

#if CV_AVX2
    static const std::size_t step = 4;
    static const __m256d mask = _mm256_set1_pd(0xFFFFFFFFFFFFFFFF);
    static const __m256d zeros = _mm256_setzero_pd();
#else
    static const std::size_t step = 2;
    static const __m128d mask = _mm_set1_pd(0xFFFFFFFFFFFFFFFF);
    static const __m128d zeros = _mm_setzero_pd();
#endif


    if(this->_core->_is_algn)
    {
        for(std::size_t r=rows.begin();r<rows.end();r++,src+=this->_core->_src_step1,dst1+=this->_core->_dst1_step1,dst2+=this->_core->_dst2_step1)
        {

            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            for(std::size_t c=0;c<cols.size();c+=step,it_src+=step,it_dst1+=step,it_dst2+=step)
            {
#if CV_AVX2
                __m256d I = _mm256_load_pd(it_src);

                __m256d cmp = _mm256_cmp_pd(I,zeros,_CMP_GT_OQ);

                I = _mm256_and_pd(cmp,mask);
                __m256d Ic = _mm256_andnot_pd(cmp,mask);

                I = _mm256_or_pd(I,Ic);
                I = _mm256_andnot_pd(I,mask);

                _mm256_stream_pd(it_dst1,I);
                _mm256_stream_pd(it_dst2,Ic);

#else

                __m128d I = _mm_load_pd(it_src);

                __m128d cmp = _mm_cmpgt_pd(I,zeros);

                I = _mm_and_pd(cmp,mask);
                __m128d Ic = _mm_andnot_pd(cmp,mask);

                I = _mm_or_pd(I,Ic);
                I = _mm_andnot_pd(I,mask);

                _mm_stream_pd(it_dst1,I);
                _mm_stream_pd(it_dst2,Ic);
#endif
            }
        }
    }
    else
    {


        const std::size_t stop = cols.size() - (cols.size()%step);

        for(std::size_t r=rows.begin(); r<rows.end(); r++,
                                                      src+=this->_core->_src_step1,
                                                      dst1+=this->_core->_dst1_step1,
                                                      dst2+=this->_core->_dst2_step1)
        {
            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            std::size_t c=0;

            for(;c<stop; c+=step,
                         it_src+=step,
                         it_dst1+=step,
                         it_dst2+=step)
            {
#if CV_AVX2
                __m256d I = _mm256_loadu_pd(it_src);

                __m256d cmp = _mm256_cmp_pd(I,zeros,_CMP_GT_OQ);

                I = _mm256_and_pd(cmp,mask);
                __m256d Ic = _mm256_andnot_pd(cmp,mask);

                I = _mm256_or_pd(I,Ic);
                I = _mm256_andnot_pd(I,mask);

                _mm256_storeu_pd(it_dst1,I);
                _mm256_storeu_pd(it_dst2,Ic);

#else

                __m128d I = _mm_loadu_pd(it_src);

                __m128d cmp = _mm_cmpgt_pd(I,zeros);

                I = _mm_and_pd(cmp,mask);
                __m128d Ic = _mm_andnot_pd(cmp,mask);

                I = _mm_or_pd(I,Ic);
                I = _mm_andnot_pd(I,mask);


                _mm_storeu_pd(it_dst1,I);
                _mm_storeu_pd(it_dst2,Ic);
#endif
            }

            for(;c<cols.size();c++,it_src++,it_dst1++,it_dst2++)
            {
                uint I = *it_src > 0 ? 0x0 : 0xFFFFFFFF;
                uint Ic = ~I;

                I = ~(I | Ic);

                *it_dst1 = I;
                *it_dst2 = Ic;
            }
        }

    }


}


#else

template<>
void step1_as_is<uchar>::operator()(const tbb::blocked_range<std::size_t>& tmp)const
{
    tbb::blocked_range<std::size_t> range(tmp.begin() * this->_core->_grain,std::min( tmp.end() * this->_core->_grain,this->_core->_cols ) );

    const_pointer src = this->_core->_src + range.begin() * this->_core->_src_step1;
    pointer dst1 = this->_core->_dst1 + range.begin() * this->_core->_dst1_step1;
    pointer dst2 = this->_core->_dst2 + range.begin() * this->_core->_dst2_step1;

    static const std::size_t step = 4;


        const std::size_t stop = this->_core->_cols - (this->_core->_cols%step);

        for(std::size_t r=range.begin(); r<range.end(); r++,
                                                        src+=this->_core->_src_step1,
                                                        dst1+=this->_core->_dst1_step1,
                                                        dst2+=this->_core->_dst2_step1)
        {
            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            std::size_t c=0;

            for(;c<this->_core->_cols; c+=step,
                                       it_src+=step,
                                       it_dst1+=step,
                                       it_dst2+=step)
            {

                value_type I = *it_src;
                value_type Ic = ~I;

                I |= Ic;

                *it_dst1 = I;
                *it_dst2 = Ic;


                I = *(it_src+1);
                Ic = ~I;

                I |= Ic;

                *(it_dst1+1) = I;
                *(it_dst2+1) = Ic;


                I = *(it_src+2);
                Ic = ~I;

                I |= Ic;

                *(it_dst1+2) = I;
                *(it_dst2+2) = Ic;


                I = *(it_src+3);
                Ic = ~I;

                I |= Ic;

                *(it_dst1+3) = I;
                *(it_dst2+3) = Ic;

            }

            for(;c<this->_core->_cols; c++, it_src++, it_dst1++, it_dst2++)
            {
                value_type I = *it_src;
                value_type Ic = ~I;

                I |= Ic;

                *it_dst1 = I;
                *it_dst2 = Ic;
            }
        }
}

template<>
void step1_as_is<uchar>::operator()(const tbb::blocked_range2d<std::size_t>& range)const
{

    const tbb::blocked_range<std::size_t>& rows = range.rows();
    const tbb::blocked_range<std::size_t>& tmp = range.cols();

    tbb::blocked_range<std::size_t> cols(tmp.begin() * this->_core->_grain, std::min( tmp.end() * this->_core->_grain, this->_core->_cols));

    const_pointer src = this->_core->_src + rows.begin() * this->_core->_src_step1 + cols.begin();
    pointer dst1 = this->_core->_dst1 + rows.begin() * this->_core->_dst1_step1 + cols.begin();
    pointer dst2 = this->_core->_dst2 + rows.begin() * this->_core->_dst2_step1 + cols.begin();

    static const std::size_t step = 4;


        const std::size_t stop = cols.size() - (cols.size()%step);

        for(std::size_t r=rows.begin(); r<rows.end(); r++,
                                                      src+=this->_core->_src_step1,
                                                      dst1+=this->_core->_dst1_step1,
                                                      dst2+=this->_core->_dst2_step1)
        {
            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            std::size_t c=0;

            for(;c<stop; c+=step,
                         it_src+=step,
                         it_dst1+=step,
                         it_dst2+=step)
            {

                value_type I = *it_src > 0 ? 0x0 : 0xFF;
                value_type Ic = ~I;

                I = ~(I | Ic);

                *it_dst1 = I;
                *it_dst2 = Ic;


                I = *(it_src+1) > 0 ? 0x0 : 0xFF;
                Ic = ~I;

                I = ~(I | Ic);

                *(it_dst1+1) = I;
                *(it_dst2+1) = Ic;


                I = *(it_src+2) > 0 ? 0x0 : 0xFF;
                Ic = ~I;

                I = ~(I | Ic);

                *(it_dst1+2) = I;
                *(it_dst2+2) = Ic;


                I = *(it_src+3) > 0 ? 0x0 : 0xFF;
                Ic = ~I;

                I = ~(I | Ic);

                *(it_dst1+3) = I;
                *(it_dst2+3) = Ic;

            }

            for(;c<cols.size();c++,it_src++,it_dst1++,it_dst2++)
            {
                value_type I = *it_src > 0 ? 0x0 : 0xFF;
                value_type Ic = ~I;

                I = ~(I | Ic);

                *it_dst1 = I;
                *it_dst2 = Ic;
            }
        }
}



template<>
void step1_as_is<schar>::operator()(const tbb::blocked_range<std::size_t>& tmp)const
{
    tbb::blocked_range<std::size_t> range(tmp.begin() * this->_core->_grain,std::min( tmp.end() * this->_core->_grain,this->_core->_cols ) );

    const_pointer src = this->_core->_src + range.begin() * this->_core->_src_step1;
    pointer dst1 = this->_core->_dst1 + range.begin() * this->_core->_dst1_step1;
    pointer dst2 = this->_core->_dst2 + range.begin() * this->_core->_dst2_step1;

    static const std::size_t step = 4;


        const std::size_t stop = this->_core->_cols - (this->_core->_cols%step);

        for(std::size_t r=range.begin(); r<range.end(); r++,
                                                        src+=this->_core->_src_step1,
                                                        dst1+=this->_core->_dst1_step1,
                                                        dst2+=this->_core->_dst2_step1)
        {
            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            std::size_t c=0;

            for(;c<this->_core->_cols; c+=step,
                                       it_src+=step,
                                       it_dst1+=step,
                                       it_dst2+=step)
            {

                value_type I = *it_src > 0 ? 0x0 : 0xFF;
                value_type Ic = ~I;

                I = ~(I | Ic);

                *it_dst1 = I;
                *it_dst2 = Ic;


                I = *(it_src+1) > 0 ? 0x0 : 0xFF;
                Ic = ~I;

                I = ~(I | Ic);

                *(it_dst1+1) = I;
                *(it_dst2+1) = Ic;


                I = *(it_src+2) > 0 ? 0x0 : 0xFF;
                Ic = ~I;

                I = ~(I | Ic);

                *(it_dst1+2) = I;
                *(it_dst2+2) = Ic;


                I = *(it_src+3) > 0 ? 0x0 : 0xFF;
                Ic = ~I;

                I = ~(I | Ic);

                *(it_dst1+3) = I;
                *(it_dst2+3) = Ic;

            }

            for(;c<this->_core->_cols; c++, it_src++, it_dst1++, it_dst2++)
            {
                value_type I = *it_src > 0 ? 0x0 : 0xFF;
                value_type Ic = ~I;

                I = ~(I | Ic);

                *it_dst1 = I;
                *it_dst2 = Ic;
            }
        }
}

template<>
void step1_as_is<schar>::operator()(const tbb::blocked_range2d<std::size_t>& range)const
{

    const tbb::blocked_range<std::size_t>& rows = range.rows();
    const tbb::blocked_range<std::size_t>& tmp = range.cols();

    tbb::blocked_range<std::size_t> cols(tmp.begin() * this->_core->_grain, std::min( tmp.end() * this->_core->_grain, this->_core->_cols));

    const_pointer src = this->_core->_src + rows.begin() * this->_core->_src_step1 + cols.begin();
    pointer dst1 = this->_core->_dst1 + rows.begin() * this->_core->_dst1_step1 + cols.begin();
    pointer dst2 = this->_core->_dst2 + rows.begin() * this->_core->_dst2_step1 + cols.begin();

    static const std::size_t step = 4;


        const std::size_t stop = cols.size() - (cols.size()%step);

        for(std::size_t r=rows.begin(); r<rows.end(); r++,
                                                      src+=this->_core->_src_step1,
                                                      dst1+=this->_core->_dst1_step1,
                                                      dst2+=this->_core->_dst2_step1)
        {
            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            std::size_t c=0;

            for(;c<stop; c+=step,
                         it_src+=step,
                         it_dst1+=step,
                         it_dst2+=step)
            {

                value_type I = *it_src > 0 ? 0x0 : 0xFF;
                value_type Ic = ~I;

                I = ~(I | Ic);

                *it_dst1 = I;
                *it_dst2 = Ic;


                I = *(it_src+1) > 0 ? 0x0 : 0xFF;
                Ic = ~I;

                I = ~(I | Ic);

                *(it_dst1+1) = I;
                *(it_dst2+1) = Ic;


                I = *(it_src+2) > 0 ? 0x0 : 0xFF;
                Ic = ~I;

                I = ~(I | Ic);

                *(it_dst1+2) = I;
                *(it_dst2+2) = Ic;


                I = *(it_src+3) > 0 ? 0x0 : 0xFF;
                Ic = ~I;

                I = ~(I | Ic);

                *(it_dst1+3) = I;
                *(it_dst2+3) = Ic;

            }

            for(;c<cols.size();c++,it_src++,it_dst1++,it_dst2++)
            {
                value_type I = *it_src > 0 ? 0x0 : 0xFF;
                value_type Ic = ~I;

                I = ~(I | Ic);

                *it_dst1 = I;
                *it_dst2 = Ic;
            }
        }
}


template<>
void step1_as_is<ushort>::operator()(const tbb::blocked_range<std::size_t>& tmp)const
{
    tbb::blocked_range<std::size_t> range(tmp.begin() * this->_core->_grain,std::min( tmp.end() * this->_core->_grain,this->_core->_cols ) );

    const_pointer src = this->_core->_src + range.begin() * this->_core->_src_step1;
    pointer dst1 = this->_core->_dst1 + range.begin() * this->_core->_dst1_step1;
    pointer dst2 = this->_core->_dst2 + range.begin() * this->_core->_dst2_step1;

    static const std::size_t step = 4;


        const std::size_t stop = this->_core->_cols - (this->_core->_cols%step);

        for(std::size_t r=range.begin(); r<range.end(); r++,
                                                        src+=this->_core->_src_step1,
                                                        dst1+=this->_core->_dst1_step1,
                                                        dst2+=this->_core->_dst2_step1)
        {
            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            std::size_t c=0;

            for(;c<this->_core->_cols; c+=step,
                                       it_src+=step,
                                       it_dst1+=step,
                                       it_dst2+=step)
            {

                value_type I = *it_src > 0 ? 0x0 : 0xFFFF;
                value_type Ic = ~I;

                I = ~(I | Ic);

                *it_dst1 = I;
                *it_dst2 = Ic;


                I = *(it_src+1) > 0 ? 0x0 : 0xFFFF;
                Ic = ~I;

                I = ~(I | Ic);

                *(it_dst1+1) = I;
                *(it_dst2+1) = Ic;


                I = *(it_src+2) > 0 ? 0x0 : 0xFFFF;
                Ic = ~I;

                I = ~(I | Ic);

                *(it_dst1+2) = I;
                *(it_dst2+2) = Ic;


                I = *(it_src+3) > 0 ? 0x0 : 0xFFFF;
                Ic = ~I;

                I = ~(I | Ic);

                *(it_dst1+3) = I;
                *(it_dst2+3) = Ic;

            }

            for(;c<this->_core->_cols; c++, it_src++, it_dst1++, it_dst2++)
            {
                value_type I = *it_src > 0 ? 0x0 : 0xFFFF;
                value_type Ic = ~I;

                I = ~(I | Ic);

                *it_dst1 = I;
                *it_dst2 = Ic;
            }
        }
}

template<>
void step1_as_is<ushort>::operator()(const tbb::blocked_range2d<std::size_t>& range)const
{

    const tbb::blocked_range<std::size_t>& rows = range.rows();
    const tbb::blocked_range<std::size_t>& tmp = range.cols();

    tbb::blocked_range<std::size_t> cols(tmp.begin() * this->_core->_grain, std::min( tmp.end() * this->_core->_grain, this->_core->_cols));

    const_pointer src = this->_core->_src + rows.begin() * this->_core->_src_step1 + cols.begin();
    pointer dst1 = this->_core->_dst1 + rows.begin() * this->_core->_dst1_step1 + cols.begin();
    pointer dst2 = this->_core->_dst2 + rows.begin() * this->_core->_dst2_step1 + cols.begin();

    static const std::size_t step = 4;


        const std::size_t stop = cols.size() - (cols.size()%step);

        for(std::size_t r=rows.begin(); r<rows.end(); r++,
                                                      src+=this->_core->_src_step1,
                                                      dst1+=this->_core->_dst1_step1,
                                                      dst2+=this->_core->_dst2_step1)
        {
            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            std::size_t c=0;

            for(;c<stop; c+=step,
                         it_src+=step,
                         it_dst1+=step,
                         it_dst2+=step)
            {

                value_type I = *it_src > 0 ? 0x0 : 0xFFFF;
                value_type Ic = ~I;

                I = ~(I | Ic);

                *it_dst1 = I;
                *it_dst2 = Ic;


                I = *(it_src+1) > 0 ? 0x0 : 0xFFFF;
                Ic = ~I;

                I = ~(I | Ic);

                *(it_dst1+1) = I;
                *(it_dst2+1) = Ic;


                I = *(it_src+2) > 0 ? 0x0 : 0xFFFF;
                Ic = ~I;

                I = ~(I | Ic);

                *(it_dst1+2) = I;
                *(it_dst2+2) = Ic;


                I = *(it_src+3) > 0 ? 0x0 : 0xFFFF;
                Ic = ~I;

                I = ~(I | Ic);

                *(it_dst1+3) = I;
                *(it_dst2+3) = Ic;

            }

            for(;c<cols.size();c++,it_src++,it_dst1++,it_dst2++)
            {
                value_type I = *it_src > 0 ? 0x0 : 0xFFFF;
                value_type Ic = ~I;

                I = ~(I | Ic);

                *it_dst1 = I;
                *it_dst2 = Ic;
            }
        }
}

template<>
void step1_as_is<short>::operator()(const tbb::blocked_range<std::size_t>& tmp)const
{
    tbb::blocked_range<std::size_t> range(tmp.begin() * this->_core->_grain,std::min( tmp.end() * this->_core->_grain,this->_core->_cols ) );

    const_pointer src = this->_core->_src + range.begin() * this->_core->_src_step1;
    pointer dst1 = this->_core->_dst1 + range.begin() * this->_core->_dst1_step1;
    pointer dst2 = this->_core->_dst2 + range.begin() * this->_core->_dst2_step1;

    static const std::size_t step = 4;


        const std::size_t stop = this->_core->_cols - (this->_core->_cols%step);

        for(std::size_t r=range.begin(); r<range.end(); r++,
                                                        src+=this->_core->_src_step1,
                                                        dst1+=this->_core->_dst1_step1,
                                                        dst2+=this->_core->_dst2_step1)
        {
            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            std::size_t c=0;

            for(;c<this->_core->_cols; c+=step,
                                       it_src+=step,
                                       it_dst1+=step,
                                       it_dst2+=step)
            {

                value_type I = *it_src > 0 ? 0x0 : 0xFFFF;
                value_type Ic = ~I;

                I = ~(I | Ic);

                *it_dst1 = I;
                *it_dst2 = Ic;


                I = *(it_src+1) > 0 ? 0x0 : 0xFFFF;
                Ic = ~I;

                I = ~(I | Ic);

                *(it_dst1+1) = I;
                *(it_dst2+1) = Ic;


                I = *(it_src+2) > 0 ? 0x0 : 0xFFFF;
                Ic = ~I;

                I = ~(I | Ic);

                *(it_dst1+2) = I;
                *(it_dst2+2) = Ic;


                I = *(it_src+3) > 0 ? 0x0 : 0xFFFF;
                Ic = ~I;

                I = ~(I | Ic);

                *(it_dst1+3) = I;
                *(it_dst2+3) = Ic;

            }

            for(;c<this->_core->_cols; c++, it_src++, it_dst1++, it_dst2++)
            {
                value_type I = *it_src > 0 ? 0x0 : 0xFFFF;
                value_type Ic = ~I;

                I = ~(I | Ic);

                *it_dst1 = I;
                *it_dst2 = Ic;
            }
        }
}

template<>
void step1_as_is<short>::operator()(const tbb::blocked_range2d<std::size_t>& range)const
{

    const tbb::blocked_range<std::size_t>& rows = range.rows();
    const tbb::blocked_range<std::size_t>& tmp = range.cols();

    tbb::blocked_range<std::size_t> cols(tmp.begin() * this->_core->_grain, std::min( tmp.end() * this->_core->_grain, this->_core->_cols));

    const_pointer src = this->_core->_src + rows.begin() * this->_core->_src_step1 + cols.begin();
    pointer dst1 = this->_core->_dst1 + rows.begin() * this->_core->_dst1_step1 + cols.begin();
    pointer dst2 = this->_core->_dst2 + rows.begin() * this->_core->_dst2_step1 + cols.begin();

    static const std::size_t step = 4;


        const std::size_t stop = cols.size() - (cols.size()%step);

        for(std::size_t r=rows.begin(); r<rows.end(); r++,
                                                      src+=this->_core->_src_step1,
                                                      dst1+=this->_core->_dst1_step1,
                                                      dst2+=this->_core->_dst2_step1)
        {
            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            std::size_t c=0;

            for(;c<stop; c+=step,
                         it_src+=step,
                         it_dst1+=step,
                         it_dst2+=step)
            {

                value_type I = *it_src > 0 ? 0x0 : 0xFFFF;
                value_type Ic = ~I;

                I = ~(I | Ic);

                *it_dst1 = I;
                *it_dst2 = Ic;


                I = *(it_src+1) > 0 ? 0x0 : 0xFFFF;
                Ic = ~I;

                I = ~(I | Ic);

                *(it_dst1+1) = I;
                *(it_dst2+1) = Ic;


                I = *(it_src+2) > 0 ? 0x0 : 0xFFFF;
                Ic = ~I;

                I = ~(I | Ic);

                *(it_dst1+2) = I;
                *(it_dst2+2) = Ic;


                I = *(it_src+3) > 0 ? 0x0 : 0xFFFF;
                Ic = ~I;

                I = ~(I | Ic);

                *(it_dst1+3) = I;
                *(it_dst2+3) = Ic;

            }

            for(;c<cols.size();c++,it_src++,it_dst1++,it_dst2++)
            {
                value_type I = *it_src > 0 ? 0x0 : 0xFFFF;
                value_type Ic = ~I;

                I = ~(I | Ic);

                *it_dst1 = I;
                *it_dst2 = Ic;
            }
        }
}

template<>
void step1_as_is<int>::operator()(const tbb::blocked_range<std::size_t>& tmp)const
{
    tbb::blocked_range<std::size_t> range(tmp.begin() * this->_core->_grain,std::min( tmp.end() * this->_core->_grain,this->_core->_cols ) );

    const_pointer src = this->_core->_src + range.begin() * this->_core->_src_step1;
    pointer dst1 = this->_core->_dst1 + range.begin() * this->_core->_dst1_step1;
    pointer dst2 = this->_core->_dst2 + range.begin() * this->_core->_dst2_step1;

    static const std::size_t step = 4;


        const std::size_t stop = this->_core->_cols - (this->_core->_cols%step);

        for(std::size_t r=range.begin(); r<range.end(); r++,
                                                        src+=this->_core->_src_step1,
                                                        dst1+=this->_core->_dst1_step1,
                                                        dst2+=this->_core->_dst2_step1)
        {
            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            std::size_t c=0;

            for(;c<this->_core->_cols; c+=step,
                                       it_src+=step,
                                       it_dst1+=step,
                                       it_dst2+=step)
            {

                value_type I = *it_src > 0 ? 0x0 : 0xFFFFFFFF;
                value_type Ic = ~I;

                I = ~(I | Ic);

                *it_dst1 = I;
                *it_dst2 = Ic;


                I = *(it_src+1) > 0 ? 0x0 : 0xFFFFFFFF;
                Ic = ~I;

                I = ~(I | Ic);

                *(it_dst1+1) = I;
                *(it_dst2+1) = Ic;


                I = *(it_src+2) > 0 ? 0x0 : 0xFFFFFFFF;
                Ic = ~I;

                I = ~(I | Ic);

                *(it_dst1+2) = I;
                *(it_dst2+2) = Ic;


                I = *(it_src+3) > 0 ? 0x0 : 0xFFFFFFFF;
                Ic = ~I;

                I = ~(I | Ic);

                *(it_dst1+3) = I;
                *(it_dst2+3) = Ic;

            }

            for(;c<this->_core->_cols; c++, it_src++, it_dst1++, it_dst2++)
            {
                value_type I = *it_src > 0 ? 0x0 : 0xFFFFFFFF;
                value_type Ic = ~I;

                I = ~(I | Ic);

                *it_dst1 = I;
                *it_dst2 = Ic;
            }
        }
}

template<>
void step1_as_is<int>::operator()(const tbb::blocked_range2d<std::size_t>& range)const
{

    const tbb::blocked_range<std::size_t>& rows = range.rows();
    const tbb::blocked_range<std::size_t>& tmp = range.cols();

    tbb::blocked_range<std::size_t> cols(tmp.begin() * this->_core->_grain, std::min( tmp.end() * this->_core->_grain, this->_core->_cols));

    const_pointer src = this->_core->_src + rows.begin() * this->_core->_src_step1 + cols.begin();
    pointer dst1 = this->_core->_dst1 + rows.begin() * this->_core->_dst1_step1 + cols.begin();
    pointer dst2 = this->_core->_dst2 + rows.begin() * this->_core->_dst2_step1 + cols.begin();

    static const std::size_t step = 4;


        const std::size_t stop = cols.size() - (cols.size()%step);

        for(std::size_t r=rows.begin(); r<rows.end(); r++,
                                                      src+=this->_core->_src_step1,
                                                      dst1+=this->_core->_dst1_step1,
                                                      dst2+=this->_core->_dst2_step1)
        {
            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            std::size_t c=0;

            for(;c<stop; c+=step,
                         it_src+=step,
                         it_dst1+=step,
                         it_dst2+=step)
            {

                value_type I = *it_src > 0 ? 0x0 : 0xFFFFFFFF;
                value_type Ic = ~I;

                I = ~(I | Ic);

                *it_dst1 = I;
                *it_dst2 = Ic;


                I = *(it_src+1) > 0 ? 0x0 : 0xFFFFFFFF;
                Ic = ~I;

                I = ~(I | Ic);

                *(it_dst1+1) = I;
                *(it_dst2+1) = Ic;


                I = *(it_src+2) > 0 ? 0x0 : 0xFFFFFFFF;
                Ic = ~I;

                I = ~(I | Ic);

                *(it_dst1+2) = I;
                *(it_dst2+2) = Ic;


                I = *(it_src+3) > 0 ? 0x0 : 0xFFFFFFFF;
                Ic = ~I;

                I = ~(I | Ic);

                *(it_dst1+3) = I;
                *(it_dst2+3) = Ic;

            }

            for(;c<cols.size();c++,it_src++,it_dst1++,it_dst2++)
            {
                value_type I = *it_src > 0 ? 0x0 : 0xFFFFFFFF;
                value_type Ic = ~I;

                I = ~(I | Ic);

                *it_dst1 = I;
                *it_dst2 = Ic;
            }
        }
}

template<>
void step1_as_is<float>::operator()(const tbb::blocked_range<std::size_t>& tmp)const
{
    tbb::blocked_range<std::size_t> range(tmp.begin() * this->_core->_grain,std::min( tmp.end() * this->_core->_grain,this->_core->_cols ) );

    const_pointer src = this->_core->_src + range.begin() * this->_core->_src_step1;
    pointer dst1 = this->_core->_dst1 + range.begin() * this->_core->_dst1_step1;
    pointer dst2 = this->_core->_dst2 + range.begin() * this->_core->_dst2_step1;

    static const std::size_t step = 4;


        const std::size_t stop = this->_core->_cols - (this->_core->_cols%step);

        for(std::size_t r=range.begin(); r<range.end(); r++,
                                                        src+=this->_core->_src_step1,
                                                        dst1+=this->_core->_dst1_step1,
                                                        dst2+=this->_core->_dst2_step1)
        {
            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            std::size_t c=0;

            for(;c<this->_core->_cols; c+=step,
                                       it_src+=step,
                                       it_dst1+=step,
                                       it_dst2+=step)
            {

                uint I = *it_src > 0 ? 0x0 : 0xFFFFFFFF;
                uint Ic = ~I;

                I = ~(I | Ic);

                *it_dst1 = I;
                *it_dst2 = Ic;


                I = *(it_src+1) > 0 ? 0x0 : 0xFFFFFFFF;
                Ic = ~I;

                I = ~(I | Ic);

                *(it_dst1+1) = I;
                *(it_dst2+1) = Ic;


                I = *(it_src+2) > 0 ? 0x0 : 0xFFFFFFFF;
                Ic = ~I;

                I = ~(I | Ic);

                *(it_dst1+2) = I;
                *(it_dst2+2) = Ic;


                I = *(it_src+3) > 0 ? 0x0 : 0xFFFFFFFF;
                Ic = ~I;

                I = ~(I | Ic);

                *(it_dst1+3) = I;
                *(it_dst2+3) = Ic;

            }

            for(;c<this->_core->_cols; c++, it_src++, it_dst1++, it_dst2++)
            {
                uint I = *it_src > 0 ? 0x0 : 0xFFFFFFFF;
                uint Ic = ~I;

                I = ~(I | Ic);

                *it_dst1 = I;
                *it_dst2 = Ic;
            }
        }
}

template<>
void step1_as_is<float>::operator()(const tbb::blocked_range2d<std::size_t>& range)const
{

    const tbb::blocked_range<std::size_t>& rows = range.rows();
    const tbb::blocked_range<std::size_t>& tmp = range.cols();

    tbb::blocked_range<std::size_t> cols(tmp.begin() * this->_core->_grain, std::min( tmp.end() * this->_core->_grain, this->_core->_cols));

    const_pointer src = this->_core->_src + rows.begin() * this->_core->_src_step1 + cols.begin();
    pointer dst1 = this->_core->_dst1 + rows.begin() * this->_core->_dst1_step1 + cols.begin();
    pointer dst2 = this->_core->_dst2 + rows.begin() * this->_core->_dst2_step1 + cols.begin();

    static const std::size_t step = 4;


    const std::size_t stop = cols.size() - (cols.size()%step);

    for(std::size_t r=rows.begin(); r<rows.end(); r++,
        src+=this->_core->_src_step1,
        dst1+=this->_core->_dst1_step1,
        dst2+=this->_core->_dst2_step1)
    {
        const_pointer it_src = src;
        pointer it_dst1 = dst1;
        pointer it_dst2 = dst2;

        std::size_t c=0;

        for(;c<stop; c+=step,
            it_src+=step,
            it_dst1+=step,
            it_dst2+=step)
        {

            uint I = *it_src > 0 ? 0x0 : 0xFFFFFFFF;
            uint Ic = ~I;

            I = ~(I | Ic);

            *it_dst1 = I;
            *it_dst2 = Ic;


            I = *(it_src+1) > 0 ? 0x0 : 0xFFFFFFFF;
            Ic = ~I;

            I = ~(I | Ic);

            *(it_dst1+1) = I;
            *(it_dst2+1) = Ic;


            I = *(it_src+2) > 0 ? 0x0 : 0xFFFFFFFF;
            Ic = ~I;

            I = ~(I | Ic);

            *(it_dst1+2) = I;
            *(it_dst2+2) = Ic;


            I = *(it_src+3) > 0 ? 0x0 : 0xFFFFFFFF;
            Ic = ~I;

            I = ~(I | Ic);

            *(it_dst1+3) = I;
            *(it_dst2+3) = Ic;

        }

        for(;c<cols.size();c++,it_src++,it_dst1++,it_dst2++)
        {

            int I = *it_src > 0 ? 0x0 : 0xFFFFFFFF;
            int Ic = ~I;

            I = ~(I | Ic);

            *it_dst1 = I;
            *it_dst2 = Ic;
        }
    }
}


template<>
void step1_as_is<double>::operator()(const tbb::blocked_range<std::size_t>& tmp)const
{
    tbb::blocked_range<std::size_t> range(tmp.begin() * this->_core->_grain,std::min( tmp.end() * this->_core->_grain,this->_core->_cols ) );

    const_pointer src = this->_core->_src + range.begin() * this->_core->_src_step1;
    pointer dst1 = this->_core->_dst1 + range.begin() * this->_core->_dst1_step1;
    pointer dst2 = this->_core->_dst2 + range.begin() * this->_core->_dst2_step1;

    static const std::size_t step = 4;


        const std::size_t stop = this->_core->_cols - (this->_core->_cols%step);

        for(std::size_t r=range.begin(); r<range.end(); r++,
                                                        src+=this->_core->_src_step1,
                                                        dst1+=this->_core->_dst1_step1,
                                                        dst2+=this->_core->_dst2_step1)
        {
            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            std::size_t c=0;

            for(;c<this->_core->_cols; c+=step,
                                       it_src+=step,
                                       it_dst1+=step,
                                       it_dst2+=step)
            {

                uint I = *it_src > 0 ? 0x0 : 0xFFFFFFFF;
                uint Ic = ~I;

                I = ~(I | Ic);

                *it_dst1 = I;
                *it_dst2 = Ic;


                I = *(it_src+1) > 0 ? 0x0 : 0xFFFFFFFF;
                Ic = ~I;

                I = ~(I | Ic);

                *(it_dst1+1) = I;
                *(it_dst2+1) = Ic;


                I = *(it_src+2) > 0 ? 0x0 : 0xFFFFFFFF;
                Ic = ~I;

                I = ~(I | Ic);

                *(it_dst1+2) = I;
                *(it_dst2+2) = Ic;


                I = *(it_src+3) > 0 ? 0x0 : 0xFFFFFFFF;
                Ic = ~I;

                I = ~(I | Ic);

                *(it_dst1+3) = I;
                *(it_dst2+3) = Ic;

            }

            for(;c<this->_core->_cols; c++, it_src++, it_dst1++, it_dst2++)
            {
                uint I = *it_src > 0 ? 0x0 : 0xFFFFFFFF;
                uint Ic = ~I;

                I = ~(I | Ic);

                *it_dst1 = I;
                *it_dst2 = Ic;
            }
        }
}

template<>
void step1_as_is<double>::operator()(const tbb::blocked_range2d<std::size_t>& range)const
{

    const tbb::blocked_range<std::size_t>& rows = range.rows();
    const tbb::blocked_range<std::size_t>& tmp = range.cols();

    tbb::blocked_range<std::size_t> cols(tmp.begin() * this->_core->_grain, std::min( tmp.end() * this->_core->_grain, this->_core->_cols));

    const_pointer src = this->_core->_src + rows.begin() * this->_core->_src_step1 + cols.begin();
    pointer dst1 = this->_core->_dst1 + rows.begin() * this->_core->_dst1_step1 + cols.begin();
    pointer dst2 = this->_core->_dst2 + rows.begin() * this->_core->_dst2_step1 + cols.begin();

    static const std::size_t step = 4;


    const std::size_t stop = cols.size() - (cols.size()%step);

    for(std::size_t r=rows.begin(); r<rows.end(); r++,
        src+=this->_core->_src_step1,
        dst1+=this->_core->_dst1_step1,
        dst2+=this->_core->_dst2_step1)
    {
        const_pointer it_src = src;
        pointer it_dst1 = dst1;
        pointer it_dst2 = dst2;

        std::size_t c=0;

        for(;c<stop; c+=step,
            it_src+=step,
            it_dst1+=step,
            it_dst2+=step)
        {

            uint I = *it_src > 0 ? 0x0 : 0xFFFFFFFF;
            uint Ic = ~I;

            I = ~(I | Ic);

            *it_dst1 = I;
            *it_dst2 = Ic;


            I = *(it_src+1) > 0 ? 0x0 : 0xFFFFFFFF;
            Ic = ~I;

            I = ~(I | Ic);

            *(it_dst1+1) = I;
            *(it_dst2+1) = Ic;


            I = *(it_src+2) > 0 ? 0x0 : 0xFFFFFFFF;
            Ic = ~I;

            I = ~(I | Ic);

            *(it_dst1+2) = I;
            *(it_dst2+2) = Ic;


            I = *(it_src+3) > 0 ? 0x0 : 0xFFFFFFFF;
            Ic = ~I;

            I = ~(I | Ic);

            *(it_dst1+3) = I;
            *(it_dst2+3) = Ic;

        }

        for(;c<cols.size();c++,it_src++,it_dst1++,it_dst2++)
        {

            uint I = *it_src > 0 ? 0x0 : 0xFFFFFFFF;
            uint Ic = ~I;

            I = ~(I | Ic);

            *it_dst1 = I;
            *it_dst2 = Ic;
        }
    }
}


#endif

template<class _Ty>
class step1_swap : public base<_Ty>
{
public:

    typedef typename base<_Ty>::value_type value_type;
    typedef typename base<_Ty>::pointer pointer;
    typedef typename base<_Ty>::const_pointer const_pointer;

    typedef base<_Ty> MyBase;

    inline step1_swap(cv::Mat_<value_type>& src,cv::Mat_<value_type>& dst1,cv::Mat_<value_type>& dst2,const std::size_t& grain):
        MyBase(src,dst1,dst2,grain)
    {}

    inline step1_swap(const step1_swap& obj):
        MyBase(obj)
    {}

    inline step1_swap(step1_swap&& obj):
        MyBase(std::move(obj))
    {}

    virtual ~step1_swap() = default;

    virtual void operator()(const tbb::blocked_range<std::size_t>& tmp)const;

    virtual void operator()(const tbb::blocked_range2d<std::size_t>& range)const;

};

#if CV_SSE2


template<>
void step1_swap<uchar>::operator()(const tbb::blocked_range<std::size_t>& tmp)const
{

    tbb::blocked_range<std::size_t> range(tmp.begin() * this->_core->_grain,std::min( tmp.end() * this->_core->_grain, this->_core->_cols));

    const_pointer src = this->_core->_src + range.begin() * this->_core->_src_step1;
    pointer dst1 = this->_core->_dst1 + range.begin() * this->_core->_dst1_step1;
    pointer dst2 = this->_core->_dst2 + range.begin() * this->_core->_dst2_step1;

#if CV_AVX2
    static const std::size_t step = 32;
    static const __m256i mask = _mm256_set1_epi8(0xFF);
    static const __m256i zeros = _mm256_setzero_si256();
#else
    const std::size_t step = 16;
    const __m128i mask = _mm_set1_epi8(0xFF);
    static const __m128i zeros = _mm_setzero_si128();
#endif

    if(this->_core->_is_algn)
    {
        for(std::size_t r=range.begin();r<range.end();r++,src+=this->_core->_src_step1,dst1+=this->_core->_dst1_step1,dst2+=this->_core->_dst2_step1)
        {
            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            for(std::size_t c=0; c<this->_core->_cols; c+=step,
                                                       it_src+=step,
                                                       it_dst1+=step,
                                                       it_dst2+=step)
            {
#if CV_AVX2
                __m256i I = _mm256_load_si256(reinterpret_cast<const __m256i*>(it_src));

                __m256i cmp = _mm256_cmpgt_epi8(I,zeros);
                __m256i Ic = _mm256_andnot_si256(cmp,mask);

                I = _mm256_or_si256(I,Ic);
                I = _mm256_andnot_si256(I,mask);

                _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst1),I);
                _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst2),Ic);
#else
                __m128i I = _mm_load_si128(reinterpret_cast<const __m128i*>(it_src));

                __m128i cmp = _mm_cmpgt_epi8(I,zeros);

                __m128i Ic = _mm_andnot_si128(cmp,mask);

                I = _mm_or_si128(I,Ic);
                I = _mm_andnot_si128(I,mask);

                _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst1),I);
                _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst2),Ic);
#endif
            }
        }
    }
    else
    {

        const std::size_t stop = this->_core->_cols - (this->_core->_cols%step);

        for(std::size_t r=range.begin(); r<range.end(); r++,
                                                        src+=this->_core->_src_step1,
                                                        dst1+=this->_core->_dst1_step1,
                                                        dst2+=this->_core->_dst2_step1)
        {
            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            std::size_t c=0;

            for(;c<stop; c+=step,
                         it_src+=step,
                         it_dst1+=step,
                         it_dst2+=step)
            {
#if CV_AVX2
                __m256i I = _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(it_src));

                __m256i cmp = _mm256_cmpgt_epi8(I,zeros);

                __m256i Ic = _mm256_andnot_si256(cmp,mask);

                I = _mm256_or_si256(I,Ic);
                I = _mm256_andnot_si256(I,mask);

                _mm256_storeu_si256(reinterpret_cast<__m256i*>(it_dst1),I);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(it_dst2),Ic);
#else

#if CV_SSE3
                __m128i I = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(it_src));
#else
                __m128i I = _mm_loadu_si128(reinterpret_cast<const __m128i*>(it_src));
#endif

                __m128i cmp = _mm_cmpgt_epi8(I,zeros);

                __m128i Ic = _mm_andnot_si128(cmp,mask);

                I = _mm_or_si128(I,Ic);
                I = _mm_andnot_si128(I,mask);

                _mm_storeu_si128(reinterpret_cast<__m128i*>(it_dst1),I);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(it_dst2),Ic);
#endif
            }

            for(;c<this->_core->_cols; c++,
                                       it_src++,
                                       it_dst1++,
                                       it_dst2++)
            {
                value_type I = *it_src > 0 ? 0xFF : 0x0;
                value_type Ic = ~I;

                I = ~(I | Ic);

                *it_dst1 = I;
                *it_dst2 = Ic;
            }
        }

    }

}

template<>
void step1_swap<uchar>::operator()(const tbb::blocked_range2d<std::size_t>& range)const
{


    const tbb::blocked_range<std::size_t>& rows = range.rows();
    const tbb::blocked_range<std::size_t>& tmp = range.cols();

    tbb::blocked_range<std::size_t> cols(tmp.begin() * this->_core->_grain, std::min( tmp.end() * this->_core->_grain, this->_core->_cols));

    const_pointer src = this->_core->_src + rows.begin() * this->_core->_src_step1 + cols.begin();
    pointer dst1 = this->_core->_dst1 + rows.begin() * this->_core->_dst1_step1 + cols.begin();
    pointer dst2 = this->_core->_dst2 + rows.begin() * this->_core->_dst2_step1 + cols.begin();

#if CV_AVX2
    static const std::size_t step = 32;
    static const __m256i mask = _mm256_set1_epi8(0xFF);
    static const __m256i zeros = _mm256_setzero_si256();
#else
    static const std::size_t step = 16;
    static const __m128i mask = _mm_set1_epi8(0xFF);
    static const __m256i zeros = _mm256_setzero_si256();
#endif


    if(this->_core->_is_algn)
    {
        for(std::size_t r=rows.begin(); r<rows.end(); r++,
                                                      src+=this->_core->_src_step1,
                                                      dst1+=this->_core->_dst1_step1,
                                                      dst2+=this->_core->_dst2_step1)
        {

            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            for(std::size_t c=0; c<cols.size(); c+=step,
                                                it_src+=step,
                                                it_dst1+=step,
                                                it_dst2+=step)
            {
#if CV_AVX2
                __m256i I = _mm256_load_si256(reinterpret_cast<const __m256i*>(it_src));

                __m256i cmp = _mm256_cmpgt_epi8(I,zeros);

                __m256i Ic = _mm256_andnot_si256(cmp,mask);

                I = _mm256_or_si256(I,Ic);
                I = _mm256_andnot_si256(I,mask);

                _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst1),I);
                _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst2),Ic);

#else
                __m128i I = _mm_load_si128(reinterpret_cast<const __m128i*>(it_src));

                __m128i cmp = _mm_cmpgt_epi8(I,zeros);

                __m128i Ic = _mm_andnot_si128(cmp,mask);

                I = _mm_or_si128(I,Ic);
                I = _mm_andnot_si128(I,mask);

                _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst1),I);
                _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst2),Ic);
#endif
            }
        }
    }
    else
    {

        const std::size_t stop = cols.size() - (cols.size()%step);

        for(std::size_t r=rows.begin(); r<rows.end(); r++,
                                                      src+=this->_core->_src_step1,
                                                      dst1+=this->_core->_dst1_step1,
                                                      dst2+=this->_core->_dst2_step1)
        {
            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            std::size_t c=0;

            for(;c<stop; c+=step,
                         it_src+=step,
                         it_dst1+=step,
                         it_dst2+=step)
            {
#if CV_AVX2

                __m256i I = _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(it_src));
                __m256i cmp = _mm256_cmpgt_epi8(I,zeros);

                __m256i Ic = _mm256_andnot_si256(cmp,mask);

                I = _mm256_or_si256(I,Ic);
                I = _mm256_andnot_si256(I,mask);

                _mm256_storeu_si256(reinterpret_cast<__m256i*>(it_dst1),I);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(it_dst2),Ic);

#else

#if CV_SSE3
                __m128i I = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(it_src));
#else
                __m128i I = _mm_loadu_si128(reinterpret_cast<const __m128i*>(it_src));
#endif
                __m128i cmp = _mm_cmpgt_epi8(I,zeros);

                __m128i Ic = _mm_andnot_si128(cmp,mask);

                I = _mm_or_si128(I,Ic);
                I = _mm_andnot_si128(I,mask);

                _mm_storeu_si128(reinterpret_cast<__m128i*>(it_dst1),I);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(it_dst2),Ic);
#endif
            }

            for(;c<cols.size();c++,
                               it_src++,
                               it_dst1++,
                               it_dst2++)
            {
                value_type I = *it_src > 0 ? 0xFF : 0x0;
                value_type Ic = ~I;

                I = ~(I | Ic);

                *it_dst1 = I;
                *it_dst2 = Ic;
            }
        }
    }

}


template<>
void step1_swap<schar>::operator()(const tbb::blocked_range<std::size_t>& tmp)const
{

    tbb::blocked_range<std::size_t> range(tmp.begin() * this->_core->_grain,std::min( tmp.end() * this->_core->_grain, this->_core->_cols));

    const_pointer src = this->_core->_src + range.begin() * this->_core->_src_step1;
    pointer dst1 = this->_core->_dst1 + range.begin() * this->_core->_dst1_step1;
    pointer dst2 = this->_core->_dst2 + range.begin() * this->_core->_dst2_step1;

#if CV_AVX2
    static const std::size_t step = 32;
    static const __m256i mask = _mm256_set1_epi8(0xFF);
    static const __m256i zeros = _mm256_setzero_si256();
#else
    const std::size_t step = 16;
    const __m128i mask = _mm_set1_epi8(0xFF);
    static const __m128i zeros = _mm_setzero_si128();
#endif

    if(this->_core->_is_algn)
    {
        for(std::size_t r=range.begin();r<range.end();r++,src+=this->_core->_src_step1,dst1+=this->_core->_dst1_step1,dst2+=this->_core->_dst2_step1)
        {
            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            for(std::size_t c=0; c<this->_core->_cols; c+=step,
                                                       it_src+=step,
                                                       it_dst1+=step,
                                                       it_dst2+=step)
            {
#if CV_AVX2
                __m256i I = _mm256_load_si256(reinterpret_cast<const __m256i*>(it_src));

                __m256i cmp = _mm256_cmpgt_epi8(I,zeros);
                __m256i Ic = _mm256_andnot_si256(cmp,mask);

                I = _mm256_or_si256(I,Ic);
                I = _mm256_andnot_si256(I,mask);

                _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst1),I);
                _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst2),Ic);
#else
                __m128i I = _mm_load_si128(reinterpret_cast<const __m128i*>(it_src));

                __m128i cmp = _mm_cmpgt_epi8(I,zeros);

                __m128i Ic = _mm_andnot_si128(cmp,mask);

                I = _mm_or_si128(I,Ic);
                I = _mm_andnot_si128(I,mask);

                _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst1),I);
                _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst2),Ic);
#endif
            }
        }
    }
    else
    {

        const std::size_t stop = this->_core->_cols - (this->_core->_cols%step);

        for(std::size_t r=range.begin(); r<range.end(); r++,
                                                        src+=this->_core->_src_step1,
                                                        dst1+=this->_core->_dst1_step1,
                                                        dst2+=this->_core->_dst2_step1)
        {
            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            std::size_t c=0;

            for(;c<stop; c+=step,
                         it_src+=step,
                         it_dst1+=step,
                         it_dst2+=step)
            {
#if CV_AVX2
                __m256i I = _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(it_src));

                __m256i cmp = _mm256_cmpgt_epi8(I,zeros);

                __m256i Ic = _mm256_andnot_si256(cmp,mask);

                I = _mm256_or_si256(I,Ic);
                I = _mm256_andnot_si256(I,mask);

                _mm256_storeu_si256(reinterpret_cast<__m256i*>(it_dst1),I);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(it_dst2),Ic);
#else

#if CV_SSE3
                __m128i I = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(it_src));
#else
                __m128i I = _mm_loadu_si128(reinterpret_cast<const __m128i*>(it_src));
#endif

                __m128i cmp = _mm_cmpgt_epi8(I,zeros);

                __m128i Ic = _mm_andnot_si128(cmp,mask);

                I = _mm_or_si128(I,Ic);
                I = _mm_andnot_si128(I,mask);

                _mm_storeu_si128(reinterpret_cast<__m128i*>(it_dst1),I);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(it_dst2),Ic);
#endif
            }

            for(;c<this->_core->_cols; c++,
                                       it_src++,
                                       it_dst1++,
                                       it_dst2++)
            {
                value_type I = *it_src > 0 ? 0xFF : 0x0;
                value_type Ic = ~I;

                I = ~(I | Ic);

                *it_dst1 = I;
                *it_dst2 = Ic;
            }
        }

    }

}

template<>
void step1_swap<schar>::operator()(const tbb::blocked_range2d<std::size_t>& range)const
{


    const tbb::blocked_range<std::size_t>& rows = range.rows();
    const tbb::blocked_range<std::size_t>& tmp = range.cols();

    tbb::blocked_range<std::size_t> cols(tmp.begin() * this->_core->_grain, std::min( tmp.end() * this->_core->_grain, this->_core->_cols));

    const_pointer src = this->_core->_src + rows.begin() * this->_core->_src_step1 + cols.begin();
    pointer dst1 = this->_core->_dst1 + rows.begin() * this->_core->_dst1_step1 + cols.begin();
    pointer dst2 = this->_core->_dst2 + rows.begin() * this->_core->_dst2_step1 + cols.begin();

#if CV_AVX2
    static const std::size_t step = 32;
    static const __m256i mask = _mm256_set1_epi8(0xFF);
    static const __m256i zeros = _mm256_setzero_si256();
#else
    static const std::size_t step = 16;
    static const __m128i mask = _mm_set1_epi8(0xFF);
    static const __m256i zeros = _mm256_setzero_si256();
#endif


    if(this->_core->_is_algn)
    {
        for(std::size_t r=rows.begin(); r<rows.end(); r++,
                                                      src+=this->_core->_src_step1,
                                                      dst1+=this->_core->_dst1_step1,
                                                      dst2+=this->_core->_dst2_step1)
        {

            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            for(std::size_t c=0; c<cols.size(); c+=step,
                                                it_src+=step,
                                                it_dst1+=step,
                                                it_dst2+=step)
            {
#if CV_AVX2
                __m256i I = _mm256_load_si256(reinterpret_cast<const __m256i*>(it_src));

                __m256i cmp = _mm256_cmpgt_epi8(I,zeros);

                __m256i Ic = _mm256_andnot_si256(cmp,mask);

                I = _mm256_or_si256(I,Ic);
                I = _mm256_andnot_si256(I,mask);

                _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst1),I);
                _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst2),Ic);

#else
                __m128i I = _mm_load_si128(reinterpret_cast<const __m128i*>(it_src));

                __m128i cmp = _mm_cmpgt_epi8(I,zeros);

                __m128i Ic = _mm_andnot_si128(cmp,mask);

                I = _mm_or_si128(I,Ic);
                I = _mm_andnot_si128(I,mask);

                _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst1),I);
                _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst2),Ic);
#endif
            }
        }
    }
    else
    {

        const std::size_t stop = cols.size() - (cols.size()%step);

        for(std::size_t r=rows.begin(); r<rows.end(); r++,
                                                      src+=this->_core->_src_step1,
                                                      dst1+=this->_core->_dst1_step1,
                                                      dst2+=this->_core->_dst2_step1)
        {
            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            std::size_t c=0;

            for(;c<stop; c+=step,
                         it_src+=step,
                         it_dst1+=step,
                         it_dst2+=step)
            {
#if CV_AVX2

                __m256i I = _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(it_src));
                __m256i cmp = _mm256_cmpgt_epi8(I,zeros);

                __m256i Ic = _mm256_andnot_si256(cmp,mask);

                I = _mm256_or_si256(I,Ic);
                I = _mm256_andnot_si256(I,mask);

                _mm256_storeu_si256(reinterpret_cast<__m256i*>(it_dst1),I);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(it_dst2),Ic);

#else

#if CV_SSE3
                __m128i I = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(it_src));
#else
                __m128i I = _mm_loadu_si128(reinterpret_cast<const __m128i*>(it_src));
#endif
                __m128i cmp = _mm_cmpgt_epi8(I,zeros);

                __m128i Ic = _mm_andnot_si128(cmp,mask);

                I = _mm_or_si128(I,Ic);
                I = _mm_andnot_si128(I,mask);

                _mm_storeu_si128(reinterpret_cast<__m128i*>(it_dst1),I);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(it_dst2),Ic);
#endif
            }

            for(;c<cols.size();c++,
                               it_src++,
                               it_dst1++,
                               it_dst2++)
            {
                value_type I = *it_src > 0 ? 0xFF : 0x0;
                value_type Ic = ~I;

                I = ~(I | Ic);

                *it_dst1 = I;
                *it_dst2 = Ic;
            }
        }
    }

}


template<>
void step1_swap<ushort>::operator()(const tbb::blocked_range<std::size_t>& tmp)const
{



    tbb::blocked_range<std::size_t> range(tmp.begin() * this->_core->_grain,std::min( tmp.end() * this->_core->_grain, this->_core->_cols));

    mtx.lock();
    std::cout<<"THERE "<<range.begin()<<" "<<range.end()<<" "<<this->_core->_cols<<" "<<this->_core->_is_algn<<" "<<this->_core->_is_inline<<std::endl;
    mtx.unlock();

    const_pointer src = this->_core->_src + range.begin() * this->_core->_src_step1;
    pointer dst1 = this->_core->_dst1 + range.begin() * this->_core->_dst1_step1;
    pointer dst2 = this->_core->_dst2 + range.begin() * this->_core->_dst2_step1;


    static const std::size_t step = this->_core->steps;
#if CV_AVX2
    static const __m256i mask = _mm256_set1_epi16(0xFFFF);
    static const __m256i zeros = _mm256_setzero_si256();
#else
    const __m128i mask = _mm_set1_epi16(0xFFFF);
    static const __m128i zeros = _mm_setzero_si128();
#endif

    if(this->_core->_is_algn)
    {
        if(!this->_core->_is_inline)
        {
            for(std::size_t r=range.begin();r<range.end();r++,src+=this->_core->_src_step1,dst1+=this->_core->_dst1_step1,dst2+=this->_core->_dst2_step1)
            {
                const_pointer it_src = src;
                pointer it_dst1 = dst1;
                pointer it_dst2 = dst2;

                for(std::size_t c=0; c<this->_core->_cols; c+=step,
                    it_src+=step,
                    it_dst1+=step,
                    it_dst2+=step)
                {
#if CV_AVX2
                    __m256i I = _mm256_load_si256(reinterpret_cast<const __m256i*>(it_src));

                    __m256i cmp = _mm256_cmpgt_epi16(I,zeros);
                    __m256i Ic = _mm256_andnot_si256(cmp,mask);

                    I = _mm256_or_si256(I,Ic);
                    I = _mm256_andnot_si256(I,mask);

                    _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst1),I);
                    _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst2),Ic);
#else
                    __m128i I = _mm_load_si128(reinterpret_cast<const __m128i*>(it_src));

                    __m128i cmp = _mm_cmpgt_epi16(I,zeros);

                    __m128i Ic = _mm_andnot_si128(cmp,mask);

                    I = _mm_or_si128(I,Ic);
                    I = _mm_andnot_si128(I,mask);

                    _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst1),I);
                    _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst2),Ic);
#endif
                }
            }
        }
        else
        {
            for(std::size_t c=0; c<this->_core->_cols; c+=step,
                src+=step,
                dst1+=step,
                dst2+=step)
            {
#if CV_AVX2
                __m256i I = _mm256_load_si256(reinterpret_cast<const __m256i*>(src));

                __m256i cmp = _mm256_cmpgt_epi16(I,zeros);
                __m256i Ic = _mm256_andnot_si256(cmp,mask);

                I = _mm256_or_si256(I,Ic);
                I = _mm256_andnot_si256(I,mask);

                _mm256_stream_si256(reinterpret_cast<__m256i*>(dst1),I);
                _mm256_stream_si256(reinterpret_cast<__m256i*>(dst2),Ic);
#else
                __m128i I = _mm_load_si128(reinterpret_cast<const __m128i*>(src));

                __m128i cmp = _mm_cmpgt_epi16(I,zeros);

                __m128i Ic = _mm_andnot_si128(cmp,mask);

                I = _mm_or_si128(I,Ic);
                I = _mm_andnot_si128(I,mask);

                _mm_stream_si128(reinterpret_cast<__m128i*>(dst1),I);
                _mm_stream_si128(reinterpret_cast<__m128i*>(dst2),Ic);
#endif
            }
        }
    }
    else
    {

        const std::size_t stop = this->_core->_cols - (this->_core->_cols%step);



        if(!this->_core->_is_inline)
        {

            for(std::size_t r=range.begin(); r<range.end(); r++,
                src+=this->_core->_src_step1,
                dst1+=this->_core->_dst1_step1,
                dst2+=this->_core->_dst2_step1)
            {
                const_pointer it_src = src;
                pointer it_dst1 = dst1;
                pointer it_dst2 = dst2;

                std::size_t c=0;

                for(;c<stop; c+=step,
                    it_src+=step,
                    it_dst1+=step,
                    it_dst2+=step)
                {
#if CV_AVX2
                    __m256i I = _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(it_src));

                    __m256i cmp = _mm256_cmpgt_epi16(I,zeros);

                    __m256i Ic = _mm256_andnot_si256(cmp,mask);

                    I = _mm256_or_si256(I,Ic);
                    I = _mm256_andnot_si256(I,mask);

                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(it_dst1),I);
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(it_dst2),Ic);
#else

#if CV_SSE3
                    __m128i I = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(it_src));
#else
                    __m128i I = _mm_loadu_si128(reinterpret_cast<const __m128i*>(it_src));
#endif

                    __m128i cmp = _mm_cmpgt_epi16(I,zeros);

                    __m128i Ic = _mm_andnot_si128(cmp,mask);

                    I = _mm_or_si128(I,Ic);
                    I = _mm_andnot_si128(I,mask);

                    _mm_storeu_si128(reinterpret_cast<__m128i*>(it_dst1),I);
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(it_dst2),Ic);
#endif
                }

                for(;c<this->_core->_cols; c++,
                    it_src++,
                    it_dst1++,
                    it_dst2++)
                {
                    value_type I = *it_src > 0 ? 0xFFFF : 0x0;
                    value_type Ic = ~I;

                    I = ~(I | Ic);

                    *it_dst1 = I;
                    *it_dst2 = Ic;
                }
            }
        }
        else
        {
            std::size_t c=0;

            for(;c<stop; c+=step,
                src+=step,
                dst1+=step,
                dst2+=step)
            {
#if CV_AVX2
                __m256i I = _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(src));

                __m256i cmp = _mm256_cmpgt_epi16(I,zeros);

                __m256i Ic = _mm256_andnot_si256(cmp,mask);

                I = _mm256_or_si256(I,Ic);
                I = _mm256_andnot_si256(I,mask);

                _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst1),I);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst2),Ic);
#else

#if CV_SSE3
                __m128i I = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(src));
#else
                __m128i I = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
#endif

                __m128i cmp = _mm_cmpgt_epi16(I,zeros);

                __m128i Ic = _mm_andnot_si128(cmp,mask);

                I = _mm_or_si128(I,Ic);
                I = _mm_andnot_si128(I,mask);

                _mm_storeu_si128(reinterpret_cast<__m128i*>(it_dst1),I);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(it_dst2),Ic);
#endif
            }

            for(;c<this->_core->_cols; c++,
                src++,
                dst1++,
                dst2++)
            {
                value_type I = *src > 0 ? 0xFFFF : 0x0;
                value_type Ic = ~I;

                I = ~(I | Ic);

                *dst1 = I;
                *dst2 = Ic;
            }
        }
    }

}

template<>
void step1_swap<ushort>::operator()(const tbb::blocked_range2d<std::size_t>& range)const
{


    const tbb::blocked_range<std::size_t>& rows = range.rows();
    const tbb::blocked_range<std::size_t>& tmp = range.cols();

    tbb::blocked_range<std::size_t> cols(tmp.begin() * this->_core->_grain, std::min( tmp.end() * this->_core->_grain, this->_core->_cols));

    const_pointer src = this->_core->_src + rows.begin() * this->_core->_src_step1 + cols.begin();
    pointer dst1 = this->_core->_dst1 + rows.begin() * this->_core->_dst1_step1 + cols.begin();
    pointer dst2 = this->_core->_dst2 + rows.begin() * this->_core->_dst2_step1 + cols.begin();

#if CV_AVX2
    static const std::size_t step = 16;
    static const __m256i mask = _mm256_set1_epi16(0xFFFF);
    static const __m256i zeros = _mm256_setzero_si256();
#else
    static const std::size_t step = 8;
    static const __m128i mask = _mm_set1_epi16(0xFFFF);
    static const __m128i zeros = _mm_setzero_si128();
#endif


    if(this->_core->_is_algn)
    {
        for(std::size_t r=rows.begin(); r<rows.end(); r++,
                                                      src+=this->_core->_src_step1,
                                                      dst1+=this->_core->_dst1_step1,
                                                      dst2+=this->_core->_dst2_step1)
        {

            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            for(std::size_t c=0; c<cols.size(); c+=step,
                                                it_src+=step,
                                                it_dst1+=step,
                                                it_dst2+=step)
            {
#if CV_AVX2
                __m256i I = _mm256_load_si256(reinterpret_cast<const __m256i*>(it_src));

                __m256i cmp = _mm256_cmpgt_epi16(I,zeros);

                __m256i Ic = _mm256_andnot_si256(cmp,mask);

                I = _mm256_or_si256(I,Ic);
                I = _mm256_andnot_si256(I,mask);

                _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst1),I);
                _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst2),Ic);

#else
                __m128i I = _mm_load_si128(reinterpret_cast<const __m128i*>(it_src));

                __m128i cmp = _mm_cmpgt_epi16(I,zeros);

                __m128i Ic = _mm_andnot_si128(cmp,mask);

                I = _mm_or_si128(I,Ic);
                I = _mm_andnot_si128(I,mask);

                _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst1),I);
                _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst2),Ic);

#endif
            }
        }
    }
    else
    {

        const std::size_t stop = cols.size() - (cols.size()%step);



        for(std::size_t r=rows.begin(); r<rows.end(); r++,
                                                      src+=this->_core->_src_step1,
                                                      dst1+=this->_core->_dst1_step1,
                                                      dst2+=this->_core->_dst2_step1)
        {
            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            std::size_t c=0;

            for(;c<stop; c+=step,
                         it_src+=step,
                         it_dst1+=step,
                         it_dst2+=step)
            {
#if CV_AVX2

                __m256i I = _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(it_src));
                __m256i cmp = _mm256_cmpgt_epi16(I,zeros);

                __m256i Ic = _mm256_andnot_si256(cmp,mask);

                I = _mm256_or_si256(I,Ic);
                I = _mm256_andnot_si256(I,mask);

                _mm256_storeu_si256(reinterpret_cast<__m256i*>(it_dst1),I);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(it_dst2),Ic);

#else

#if CV_SSE3
                __m128i I = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(it_src));
#else
                __m128i I = _mm_loadu_si128(reinterpret_cast<const __m128i*>(it_src));
#endif
                __m128i cmp = _mm_cmpgt_epi16(I,zeros);

                __m128i Ic = _mm_andnot_si128(cmp,mask);

                I = _mm_or_si128(I,Ic);
                I = _mm_andnot_si128(I,mask);

                _mm_storeu_si128(reinterpret_cast<__m128i*>(it_dst1),I);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(it_dst2),Ic);
#endif
            }

            for(;c<cols.size();c++,
                               it_src++,
                               it_dst1++,
                               it_dst2++)
            {
                value_type I = *it_src > 0 ? 0xFFFF : 0x0;
                value_type Ic = ~I;

                I = ~(I | Ic);

                *it_dst1 = I;
                *it_dst2 = Ic;
            }
        }
    }

}


template<>
void step1_swap<short>::operator()(const tbb::blocked_range<std::size_t>& tmp)const
{

    tbb::blocked_range<std::size_t> range(tmp.begin() * this->_core->_grain,std::min( tmp.end() * this->_core->_grain, this->_core->_cols));

    const_pointer src = this->_core->_src + range.begin() * this->_core->_src_step1;
    pointer dst1 = this->_core->_dst1 + range.begin() * this->_core->_dst1_step1;
    pointer dst2 = this->_core->_dst2 + range.begin() * this->_core->_dst2_step1;

#if CV_AVX2
    static const std::size_t step = 16;
    static const __m256i mask = _mm256_set1_epi16(0xFFFF);
    static const __m256i zeros = _mm256_setzero_si256();
#else
    const std::size_t step = 8;
    const __m128i mask = _mm_set1_epi16(0xFFFF);
    static const __m128i zeros = _mm_setzero_si128();
#endif

    if(this->_core->_is_algn)
    {
        for(std::size_t r=range.begin();r<range.end();r++,src+=this->_core->_src_step1,dst1+=this->_core->_dst1_step1,dst2+=this->_core->_dst2_step1)
        {
            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            for(std::size_t c=0; c<this->_core->_cols; c+=step,
                                                       it_src+=step,
                                                       it_dst1+=step,
                                                       it_dst2+=step)
            {
#if CV_AVX2
                __m256i I = _mm256_load_si256(reinterpret_cast<const __m256i*>(it_src));

                __m256i cmp = _mm256_cmpgt_epi16(I,zeros);
                __m256i Ic = _mm256_andnot_si256(cmp,mask);

                I = _mm256_or_si256(I,Ic);
                I = _mm256_andnot_si256(I,mask);

                _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst1),I);
                _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst2),Ic);
#else
                __m128i I = _mm_load_si128(reinterpret_cast<const __m128i*>(it_src));

                __m128i cmp = _mm_cmpgt_epi16(I,zeros);

                __m128i Ic = _mm_andnot_si128(cmp,mask);

                I = _mm_or_si128(I,Ic);
                I = _mm_andnot_si128(I,mask);

                _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst1),I);
                _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst2),Ic);
#endif
            }
        }
    }
    else
    {

        const std::size_t stop = this->_core->_cols - (this->_core->_cols%step);

        for(std::size_t r=range.begin(); r<range.end(); r++,
                                                        src+=this->_core->_src_step1,
                                                        dst1+=this->_core->_dst1_step1,
                                                        dst2+=this->_core->_dst2_step1)
        {
            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            std::size_t c=0;

            for(;c<stop; c+=step,
                         it_src+=step,
                         it_dst1+=step,
                         it_dst2+=step)
            {
#if CV_AVX2
                __m256i I = _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(it_src));

                __m256i cmp = _mm256_cmpgt_epi16(I,zeros);

                __m256i Ic = _mm256_andnot_si256(cmp,mask);

                I = _mm256_or_si256(I,Ic);
                I = _mm256_andnot_si256(I,mask);

                _mm256_storeu_si256(reinterpret_cast<__m256i*>(it_dst1),I);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(it_dst2),Ic);
#else

#if CV_SSE3
                __m128i I = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(it_src));
#else
                __m128i I = _mm_loadu_si128(reinterpret_cast<const __m128i*>(it_src));
#endif

                __m128i cmp = _mm_cmpgt_epi16(I,zeros);

                __m128i Ic = _mm_andnot_si128(cmp,mask);

                I = _mm_or_si128(I,Ic);
                I = _mm_andnot_si128(I,mask);

                _mm_storeu_si128(reinterpret_cast<__m128i*>(it_dst1),I);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(it_dst2),Ic);
#endif
            }

            for(;c<this->_core->_cols; c++,
                                       it_src++,
                                       it_dst1++,
                                       it_dst2++)
            {
                value_type I = *it_src > 0 ? 0xFFFF : 0x0;
                value_type Ic = ~I;

                I = ~(I | Ic);

                *it_dst1 = I;
                *it_dst2 = Ic;
            }
        }

    }

}

template<>
void step1_swap<short>::operator()(const tbb::blocked_range2d<std::size_t>& range)const
{


    const tbb::blocked_range<std::size_t>& rows = range.rows();
    const tbb::blocked_range<std::size_t>& tmp = range.cols();

    tbb::blocked_range<std::size_t> cols(tmp.begin() * this->_core->_grain, std::min( tmp.end() * this->_core->_grain, this->_core->_cols));

    const_pointer src = this->_core->_src + rows.begin() * this->_core->_src_step1 + cols.begin();
    pointer dst1 = this->_core->_dst1 + rows.begin() * this->_core->_dst1_step1 + cols.begin();
    pointer dst2 = this->_core->_dst2 + rows.begin() * this->_core->_dst2_step1 + cols.begin();

#if CV_AVX2
    static const std::size_t step = 16;
    static const __m256i mask = _mm256_set1_epi16(0xFFFF);
    static const __m256i zeros = _mm256_setzero_si256();
#else
    static const std::size_t step = 8;
    static const __m128i mask = _mm_set1_epi16(0xFFFF);
    static const __m256i zeros = _mm256_setzero_si256();
#endif


    if(this->_core->_is_algn)
    {
        for(std::size_t r=rows.begin(); r<rows.end(); r++,
                                                      src+=this->_core->_src_step1,
                                                      dst1+=this->_core->_dst1_step1,
                                                      dst2+=this->_core->_dst2_step1)
        {

            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            for(std::size_t c=0; c<cols.size(); c+=step,
                                                it_src+=step,
                                                it_dst1+=step,
                                                it_dst2+=step)
            {
#if CV_AVX2
                __m256i I = _mm256_load_si256(reinterpret_cast<const __m256i*>(it_src));

                __m256i cmp = _mm256_cmpgt_epi16(I,zeros);

                __m256i Ic = _mm256_andnot_si256(cmp,mask);

                I = _mm256_or_si256(I,Ic);
                I = _mm256_andnot_si256(I,mask);

                _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst1),I);
                _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst2),Ic);

#else
                __m128i I = _mm_load_si128(reinterpret_cast<const __m128i*>(it_src));

                __m128i cmp = _mm_cmpgt_epi16(I,zeros);

                __m128i Ic = _mm_andnot_si128(cmp,mask);

                I = _mm_or_si128(I,Ic);
                I = _mm_andnot_si128(I,mask);

                _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst1),I);
                _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst2),Ic);
#endif
            }
        }
    }
    else
    {

        const std::size_t stop = cols.size() - (cols.size()%step);

        for(std::size_t r=rows.begin(); r<rows.end(); r++,
                                                      src+=this->_core->_src_step1,
                                                      dst1+=this->_core->_dst1_step1,
                                                      dst2+=this->_core->_dst2_step1)
        {
            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            std::size_t c=0;

            for(;c<stop; c+=step,
                         it_src+=step,
                         it_dst1+=step,
                         it_dst2+=step)
            {
#if CV_AVX2

                __m256i I = _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(it_src));
                __m256i cmp = _mm256_cmpgt_epi16(I,zeros);

                __m256i Ic = _mm256_andnot_si256(cmp,mask);

                I = _mm256_or_si256(I,Ic);
                I = _mm256_andnot_si256(I,mask);

                _mm256_storeu_si256(reinterpret_cast<__m256i*>(it_dst1),I);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(it_dst2),Ic);

#else

#if CV_SSE3
                __m128i I = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(it_src));
#else
                __m128i I = _mm_loadu_si128(reinterpret_cast<const __m128i*>(it_src));
#endif
                __m128i cmp = _mm_cmpgt_epi16(I,zeros);

                __m128i Ic = _mm_andnot_si128(cmp,mask);

                I = _mm_or_si128(I,Ic);
                I = _mm_andnot_si128(I,mask);

                _mm_storeu_si128(reinterpret_cast<__m128i*>(it_dst1),I);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(it_dst2),Ic);
#endif
            }

            for(;c<cols.size();c++,
                               it_src++,
                               it_dst1++,
                               it_dst2++)
            {
                value_type I = *it_src > 0 ? 0xFFFF : 0x0;
                value_type Ic = ~I;

                I = ~(I | Ic);

                *it_dst1 = I;
                *it_dst2 = Ic;
            }
        }
    }

}


template<>
void step1_swap<int>::operator()(const tbb::blocked_range<std::size_t>& tmp)const
{

    tbb::blocked_range<std::size_t> range(tmp.begin() * this->_core->_grain,std::min( tmp.end() * this->_core->_grain, this->_core->_cols));

    const_pointer src = this->_core->_src + range.begin() * this->_core->_src_step1;
    pointer dst1 = this->_core->_dst1 + range.begin() * this->_core->_dst1_step1;
    pointer dst2 = this->_core->_dst2 + range.begin() * this->_core->_dst2_step1;

#if CV_AVX2
    static const std::size_t step = 8;
    static const __m256i mask = _mm256_set1_epi32(0xFFFFFFFF);
    static const __m256i zeros = _mm256_setzero_si256();
#else
    const std::size_t step = 4;
    const __m128i mask = _mm_set1_epi32(0xFFFFFFFF);
    static const __m128i zeros = _mm_setzero_si128();
#endif

    if(this->_core->_is_algn)
    {
        for(std::size_t r=range.begin();r<range.end();r++,src+=this->_core->_src_step1,dst1+=this->_core->_dst1_step1,dst2+=this->_core->_dst2_step1)
        {
            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            for(std::size_t c=0; c<this->_core->_cols; c+=step,
                                                       it_src+=step,
                                                       it_dst1+=step,
                                                       it_dst2+=step)
            {
#if CV_AVX2
                __m256i I = _mm256_load_si256(reinterpret_cast<const __m256i*>(it_src));

                __m256i cmp = _mm256_cmpgt_epi32(I,zeros);
                __m256i Ic = _mm256_andnot_si256(cmp,mask);

                I = _mm256_or_si256(I,Ic);
                I = _mm256_andnot_si256(I,mask);

                _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst1),I);
                _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst2),Ic);
#else
                __m128i I = _mm_load_si128(reinterpret_cast<const __m128i*>(it_src));

                __m128i cmp = _mm_cmpgt_epi32(I,zeros);

                __m128i Ic = _mm_andnot_si128(cmp,mask);

                I = _mm_or_si128(I,Ic);
                I = _mm_andnot_si128(I,mask);

                _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst1),I);
                _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst2),Ic);
#endif
            }
        }
    }
    else
    {

        const std::size_t stop = this->_core->_cols - (this->_core->_cols%step);

        for(std::size_t r=range.begin(); r<range.end(); r++,
                                                        src+=this->_core->_src_step1,
                                                        dst1+=this->_core->_dst1_step1,
                                                        dst2+=this->_core->_dst2_step1)
        {
            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            std::size_t c=0;

            for(;c<stop; c+=step,
                         it_src+=step,
                         it_dst1+=step,
                         it_dst2+=step)
            {
#if CV_AVX2
                __m256i I = _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(it_src));

                __m256i cmp = _mm256_cmpgt_epi32(I,zeros);

                __m256i Ic = _mm256_andnot_si256(cmp,mask);

                I = _mm256_or_si256(I,Ic);
                I = _mm256_andnot_si256(I,mask);

                _mm256_storeu_si256(reinterpret_cast<__m256i*>(it_dst1),I);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(it_dst2),Ic);
#else

#if CV_SSE3
                __m128i I = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(it_src));
#else
                __m128i I = _mm_loadu_si128(reinterpret_cast<const __m128i*>(it_src));
#endif

                __m128i cmp = _mm_cmpgt_epi32(I,zeros);

                __m128i Ic = _mm_andnot_si128(cmp,mask);

                I = _mm_or_si128(I,Ic);
                I = _mm_andnot_si128(I,mask);

                _mm_storeu_si128(reinterpret_cast<__m128i*>(it_dst1),I);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(it_dst2),Ic);
#endif
            }

            for(;c<this->_core->_cols; c++,
                                       it_src++,
                                       it_dst1++,
                                       it_dst2++)
            {
                value_type I = *it_src > 0 ? 0xFFFFFFFF : 0x0;
                value_type Ic = ~I;

                I = ~(I | Ic);

                *it_dst1 = I;
                *it_dst2 = Ic;
            }
        }

    }

}

template<>
void step1_swap<int>::operator()(const tbb::blocked_range2d<std::size_t>& range)const
{


    const tbb::blocked_range<std::size_t>& rows = range.rows();
    const tbb::blocked_range<std::size_t>& tmp = range.cols();

    tbb::blocked_range<std::size_t> cols(tmp.begin() * this->_core->_grain, std::min( tmp.end() * this->_core->_grain, this->_core->_cols));

    const_pointer src = this->_core->_src + rows.begin() * this->_core->_src_step1 + cols.begin();
    pointer dst1 = this->_core->_dst1 + rows.begin() * this->_core->_dst1_step1 + cols.begin();
    pointer dst2 = this->_core->_dst2 + rows.begin() * this->_core->_dst2_step1 + cols.begin();

#if CV_AVX2
    static const std::size_t step = 8;
    static const __m256i mask = _mm256_set1_epi32(0xFFFFFFFF);
    static const __m256i zeros = _mm256_setzero_si256();
#else
    static const std::size_t step = 4;
    static const __m128i mask = _mm_set1_epi32(0xFFFFFFFF);
    static const __m256i zeros = _mm256_setzero_si256();
#endif


    if(this->_core->_is_algn)
    {
        for(std::size_t r=rows.begin(); r<rows.end(); r++,
                                                      src+=this->_core->_src_step1,
                                                      dst1+=this->_core->_dst1_step1,
                                                      dst2+=this->_core->_dst2_step1)
        {

            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            for(std::size_t c=0; c<cols.size(); c+=step,
                                                it_src+=step,
                                                it_dst1+=step,
                                                it_dst2+=step)
            {
#if CV_AVX2
                __m256i I = _mm256_load_si256(reinterpret_cast<const __m256i*>(it_src));

                __m256i cmp = _mm256_cmpgt_epi32(I,zeros);

                __m256i Ic = _mm256_andnot_si256(cmp,mask);

                I = _mm256_or_si256(I,Ic);
                I = _mm256_andnot_si256(I,mask);

                _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst1),I);
                _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst2),Ic);

#else
                __m128i I = _mm_load_si128(reinterpret_cast<const __m128i*>(it_src));

                __m128i cmp = _mm_cmpgt_epi32(I,zeros);

                __m128i Ic = _mm_andnot_si128(cmp,mask);

                I = _mm_or_si128(I,Ic);
                I = _mm_andnot_si128(I,mask);

                _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst1),I);
                _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst2),Ic);
#endif
            }
        }
    }
    else
    {

        const std::size_t stop = cols.size() - (cols.size()%step);

        for(std::size_t r=rows.begin(); r<rows.end(); r++,
                                                      src+=this->_core->_src_step1,
                                                      dst1+=this->_core->_dst1_step1,
                                                      dst2+=this->_core->_dst2_step1)
        {
            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            std::size_t c=0;

            for(;c<stop; c+=step,
                         it_src+=step,
                         it_dst1+=step,
                         it_dst2+=step)
            {
#if CV_AVX2

                __m256i I = _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(it_src));
                __m256i cmp = _mm256_cmpgt_epi32(I,zeros);

                __m256i Ic = _mm256_andnot_si256(cmp,mask);

                I = _mm256_or_si256(I,Ic);
                I = _mm256_andnot_si256(I,mask);

                _mm256_storeu_si256(reinterpret_cast<__m256i*>(it_dst1),I);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(it_dst2),Ic);

#else

#if CV_SSE3
                __m128i I = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(it_src));
#else
                __m128i I = _mm_loadu_si128(reinterpret_cast<const __m128i*>(it_src));
#endif
                __m128i cmp = _mm_cmpgt_epi32(I,zeros);

                __m128i Ic = _mm_andnot_si128(cmp,mask);

                I = _mm_or_si128(I,Ic);
                I = _mm_andnot_si128(I,mask);

                _mm_storeu_si128(reinterpret_cast<__m128i*>(it_dst1),I);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(it_dst2),Ic);
#endif
            }

            for(;c<cols.size();c++,
                               it_src++,
                               it_dst1++,
                               it_dst2++)
            {
                value_type I = *it_src > 0 ? 0xFFFFFFFF : 0x0;
                value_type Ic = ~I;

                I = ~(I | Ic);

                *it_dst1 = I;
                *it_dst2 = Ic;
            }
        }
    }

}


template<>
void step1_swap<float>::operator()(const tbb::blocked_range<std::size_t>& tmp)const
{

    tbb::blocked_range<std::size_t> range(tmp.begin() * this->_core->_grain,std::min( tmp.end() * this->_core->_grain, this->_core->_cols));

    const_pointer src = this->_core->_src + range.begin() * this->_core->_src_step1;
    pointer dst1 = this->_core->_dst1 + range.begin() * this->_core->_dst1_step1;
    pointer dst2 = this->_core->_dst2 + range.begin() * this->_core->_dst2_step1;

#if CV_AVX2
    static const std::size_t step = 8;
    static const __m256 mask = _mm256_set1_ps(0xFFFFFFFF);
    static const __m256 zeros = _mm256_setzero_ps();
#else
    const std::size_t step = 4;
    const __m128 mask = _mm_set1_ps(0xFFFFFFFF);
    static const __m128 zeros = _mm_setzero_ps();
#endif

    if(this->_core->_is_algn)
    {
        for(std::size_t r=range.begin();r<range.end();r++,src+=this->_core->_src_step1,dst1+=this->_core->_dst1_step1,dst2+=this->_core->_dst2_step1)
        {
            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            for(std::size_t c=0; c<this->_core->_cols; c+=step,
                                                       it_src+=step,
                                                       it_dst1+=step,
                                                       it_dst2+=step)
            {
#if CV_AVX2
                __m256 I = _mm256_load_ps(it_src);

                __m256 cmp = _mm256_cmp_ps(I,zeros,_CMP_GT_OQ);
                __m256 Ic = _mm256_andnot_ps(cmp,mask);

                I = _mm256_or_ps(I,Ic);
                I = _mm256_andnot_ps(I,mask);

                _mm256_stream_ps(it_dst1,I);
                _mm256_stream_ps(it_dst2,Ic);
#else
                __m128 I = _mm_load_ps(reinterpret_cast<const __m128*>(it_src);

                __m128 cmp = _mm_cmpgt_ps(I,zeros);

                __m128 Ic = _mm_andnot_ps(cmp,mask);

                I = _mm_or_ps(I,Ic);
                I = _mm_andnot_ps(I,mask);

                _mm_stream_ps(reinterpret_cast<__m128*>(it_dst1,I);
                _mm_stream_ps(reinterpret_cast<__m128*>(it_dst2,Ic);
#endif
            }
        }
    }
    else
    {

        const std::size_t stop = this->_core->_cols - (this->_core->_cols%step);

        for(std::size_t r=range.begin(); r<range.end(); r++,
                                                        src+=this->_core->_src_step1,
                                                        dst1+=this->_core->_dst1_step1,
                                                        dst2+=this->_core->_dst2_step1)
        {
            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            std::size_t c=0;

            for(;c<stop; c+=step,
                         it_src+=step,
                         it_dst1+=step,
                         it_dst2+=step)
            {
#if CV_AVX2
                __m256 I = _mm256_loadu_ps(it_src);

                __m256 cmp = _mm256_cmp_ps(I,zeros,_CMP_GT_OQ);

                __m256 Ic = _mm256_andnot_ps(cmp,mask);

                I = _mm256_or_ps(I,Ic);
                I = _mm256_andnot_ps(I,mask);

                _mm256_storeu_ps(it_dst1,I);
                _mm256_storeu_ps(it_dst2,Ic);
#else

                __m128 I = _mm_loadu_ps(reinterpret_cast<const __m128*>(it_src);

                __m128 cmp = _mm_cmpgt_ps(I,zeros);

                __m128 Ic = _mm_andnot_ps(cmp,mask);

                I = _mm_or_ps(I,Ic);
                I = _mm_andnot_ps(I,mask);

                _mm_storeu_ps(reinterpret_cast<__m128*>(it_dst1,I);
                _mm_storeu_ps(reinterpret_cast<__m128*>(it_dst2,Ic);
#endif
            }

            for(;c<this->_core->_cols; c++,
                                       it_src++,
                                       it_dst1++,
                                       it_dst2++)
            {
                uint I = *it_src > 0.f ? 0xFFFFFFFF : 0x0;
                uint Ic = ~I;

                I = ~(I | Ic);

                *it_dst1 = I;
                *it_dst2 = Ic;
            }
        }

    }

}

template<>
void step1_swap<float>::operator()(const tbb::blocked_range2d<std::size_t>& range)const
{
    const tbb::blocked_range<std::size_t>& rows = range.rows();
    const tbb::blocked_range<std::size_t>& tmp = range.cols();

    tbb::blocked_range<std::size_t> cols(tmp.begin() * this->_core->_grain, std::min( tmp.end() * this->_core->_grain, this->_core->_cols));

    const_pointer src = this->_core->_src + rows.begin() * this->_core->_src_step1 + cols.begin();
    pointer dst1 = this->_core->_dst1 + rows.begin() * this->_core->_dst1_step1 + cols.begin();
    pointer dst2 = this->_core->_dst2 + rows.begin() * this->_core->_dst2_step1 + cols.begin();

#if CV_AVX2
    static const std::size_t step = 8;
    static const __m256 mask = _mm256_set1_ps(0xFFFFFFFF);
    static const __m256 zeros = _mm256_setzero_ps();
#else
    static const std::size_t step = 4;
    static const __m128 mask = _mm_set1_ps(0xFFFFFFFF);
    static const __m256 zeros = _mm256_setzero_ps();
#endif


    if(this->_core->_is_algn)
    {
        for(std::size_t r=rows.begin(); r<rows.end(); r++,
                                                      src+=this->_core->_src_step1,
                                                      dst1+=this->_core->_dst1_step1,
                                                      dst2+=this->_core->_dst2_step1)
        {

            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            for(std::size_t c=0; c<cols.size(); c+=step,
                                                it_src+=step,
                                                it_dst1+=step,
                                                it_dst2+=step)
            {
#if CV_AVX2
                __m256 I = _mm256_load_ps(it_src);

                __m256 cmp = _mm256_cmp_ps(I,zeros,_CMP_GT_OQ);

                __m256 Ic = _mm256_andnot_ps(cmp,mask);

                I = _mm256_or_ps(I,Ic);
                I = _mm256_andnot_ps(I,mask);

                _mm256_stream_ps(it_dst1,I);
                _mm256_stream_ps(it_dst2,Ic);

#else
                __m128 I = _mm_load_ps(reinterpret_cast<const __m128*>(it_src);

                __m128 cmp = _mm_cmpgt_ps(I,zeros);

                __m128 Ic = _mm_andnot_ps(cmp,mask);

                I = _mm_or_ps(I,Ic);
                I = _mm_andnot_ps(I,mask);

                _mm_stream_ps(reinterpret_cast<__m128*>(it_dst1,I);
                _mm_stream_ps(reinterpret_cast<__m128*>(it_dst2,Ic);
#endif
            }
        }
    }
    else
    {

        const std::size_t stop = cols.size() - (cols.size()%step);

        for(std::size_t r=rows.begin(); r<rows.end(); r++,
                                                      src+=this->_core->_src_step1,
                                                      dst1+=this->_core->_dst1_step1,
                                                      dst2+=this->_core->_dst2_step1)
        {
            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            std::size_t c=0;

            for(;c<stop; c+=step,
                         it_src+=step,
                         it_dst1+=step,
                         it_dst2+=step)
            {
#if CV_AVX2

                __m256 I = _mm256_loadu_ps(it_src);
                __m256 cmp = _mm256_cmp_ps(I,zeros,_CMP_GT_OQ);

                __m256 Ic = _mm256_andnot_ps(cmp,mask);

                I = _mm256_or_ps(I,Ic);
                I = _mm256_andnot_ps(I,mask);

                _mm256_storeu_ps(it_dst1,I);
                _mm256_storeu_ps(it_dst2,Ic);

#else


                __m128 I = _mm_loadu_ps(reinterpret_cast<const __m128*>(it_src);

                __m128 cmp = _mm_cmpgt_ps(I,zeros);

                __m128 Ic = _mm_andnot_ps(cmp,mask);

                I = _mm_or_ps(I,Ic);
                I = _mm_andnot_ps(I,mask);

                _mm_storeu_ps(reinterpret_cast<__m128*>(it_dst1,I);
                _mm_storeu_ps(reinterpret_cast<__m128*>(it_dst2,Ic);
#endif
            }

            for(;c<cols.size();c++,
                               it_src++,
                               it_dst1++,
                               it_dst2++)
            {
                uint I = *it_src > 0.f ? 0xFFFFFFFF : 0x0;
                uint Ic = ~I;

                I = ~(I | Ic);

                *it_dst1 = I;
                *it_dst2 = Ic;
            }
        }
    }

}


template<>
void step1_swap<double>::operator()(const tbb::blocked_range<std::size_t>& tmp)const
{

    tbb::blocked_range<std::size_t> range(tmp.begin() * this->_core->_grain,std::min( tmp.end() * this->_core->_grain, this->_core->_cols));

    const_pointer src = this->_core->_src + range.begin() * this->_core->_src_step1;
    pointer dst1 = this->_core->_dst1 + range.begin() * this->_core->_dst1_step1;
    pointer dst2 = this->_core->_dst2 + range.begin() * this->_core->_dst2_step1;

#if CV_AVX2
    static const std::size_t step = 4;
    static const __m256d mask = _mm256_set1_pd(0xFFFFFFFFFFFFFFFF);
    static const __m256d zeros = _mm256_setzero_pd();
#else
    const std::size_t step = 2;
    const __m128 mask = _mm_set1_pd(0xFFFFFFFFFFFFFFFF);
    static const __m128 zeros = _mm_setzero_pd();
#endif

    if(this->_core->_is_algn)
    {
        for(std::size_t r=range.begin();r<range.end();r++,src+=this->_core->_src_step1,dst1+=this->_core->_dst1_step1,dst2+=this->_core->_dst2_step1)
        {
            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            for(std::size_t c=0; c<this->_core->_cols; c+=step,
                                                       it_src+=step,
                                                       it_dst1+=step,
                                                       it_dst2+=step)
            {
#if CV_AVX2
                __m256d I = _mm256_load_pd(it_src);

                __m256d cmp = _mm256_cmp_pd(I,zeros,_CMP_GT_OQ);
                __m256d Ic = _mm256_andnot_pd(cmp,mask);

                I = _mm256_or_pd(I,Ic);
                I = _mm256_andnot_pd(I,mask);

                _mm256_stream_pd(it_dst1,I);
                _mm256_stream_pd(it_dst2,Ic);
#else
                __m128 I = _mm_load_pd(reinterpret_cast<const __m128*>(it_src);

                __m128 cmp = _mm_cmpgt_pd(I,zeros);

                __m128 Ic = _mm_andnot_pd(cmp,mask);

                I = _mm_or_pd(I,Ic);
                I = _mm_andnot_pd(I,mask);

                _mm_stream_pd(reinterpret_cast<__m128*>(it_dst1,I);
                _mm_stream_pd(reinterpret_cast<__m128*>(it_dst2,Ic);
#endif
            }
        }
    }
    else
    {

        const std::size_t stop = this->_core->_cols - (this->_core->_cols%step);

        for(std::size_t r=range.begin(); r<range.end(); r++,
                                                        src+=this->_core->_src_step1,
                                                        dst1+=this->_core->_dst1_step1,
                                                        dst2+=this->_core->_dst2_step1)
        {
            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            std::size_t c=0;

            for(;c<stop; c+=step,
                         it_src+=step,
                         it_dst1+=step,
                         it_dst2+=step)
            {
#if CV_AVX2
                __m256d I = _mm256_loadu_pd(it_src);

                __m256d cmp = _mm256_cmp_pd(I,zeros,_CMP_GT_OQ);

                __m256d Ic = _mm256_andnot_pd(cmp,mask);

                I = _mm256_or_pd(I,Ic);
                I = _mm256_andnot_pd(I,mask);

                _mm256_storeu_pd(it_dst1,I);
                _mm256_storeu_pd(it_dst2,Ic);
#else

#if CV_SSE3
                __m128 I = _mm_lddqu_pd(reinterpret_cast<const __m128*>(it_src);
#else
                __m128 I = _mm_loadu_pd(reinterpret_cast<const __m128*>(it_src);
#endif

                __m128 cmp = _mm_cmpgt_pd(I,zeros);

                __m128 Ic = _mm_andnot_pd(cmp,mask);

                I = _mm_or_pd(I,Ic);
                I = _mm_andnot_pd(I,mask);

                _mm_storeu_pd(reinterpret_cast<__m128*>(it_dst1,I);
                _mm_storeu_pd(reinterpret_cast<__m128*>(it_dst2,Ic);
#endif
            }

            for(;c<this->_core->_cols; c++,
                                       it_src++,
                                       it_dst1++,
                                       it_dst2++)
            {
                uint I = *it_src > 0. ? 0xFFFFFFFF : 0x0;
                uint Ic = ~I;

                I = ~(I | Ic);

                *it_dst1 = I;
                *it_dst2 = Ic;
            }
        }

    }

}

template<>
void step1_swap<double>::operator()(const tbb::blocked_range2d<std::size_t>& range)const
{
    const tbb::blocked_range<std::size_t>& rows = range.rows();
    const tbb::blocked_range<std::size_t>& tmp = range.cols();

    tbb::blocked_range<std::size_t> cols(tmp.begin() * this->_core->_grain, std::min( tmp.end() * this->_core->_grain, this->_core->_cols));

    const_pointer src = this->_core->_src + rows.begin() * this->_core->_src_step1 + cols.begin();
    pointer dst1 = this->_core->_dst1 + rows.begin() * this->_core->_dst1_step1 + cols.begin();
    pointer dst2 = this->_core->_dst2 + rows.begin() * this->_core->_dst2_step1 + cols.begin();

#if CV_AVX2
    static const std::size_t step = 4;
    static const __m256d mask = _mm256_set1_pd(0xFFFFFFFFFFFFFFFF);
    static const __m256d zeros = _mm256_setzero_pd();
#else
    static const std::size_t step = 2;
    static const __m128 mask = _mm_set1_pd(0xFFFFFFFFFFFFFFFF);
    static const __m256d zeros = _mm256_setzero_pd();
#endif


    if(this->_core->_is_algn)
    {
        for(std::size_t r=rows.begin(); r<rows.end(); r++,
                                                      src+=this->_core->_src_step1,
                                                      dst1+=this->_core->_dst1_step1,
                                                      dst2+=this->_core->_dst2_step1)
        {

            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            for(std::size_t c=0; c<cols.size(); c+=step,
                                                it_src+=step,
                                                it_dst1+=step,
                                                it_dst2+=step)
            {
#if CV_AVX2
                __m256d I = _mm256_load_pd(it_src);

                __m256d cmp = _mm256_cmp_pd(I,zeros,_CMP_GT_OQ);

                __m256d Ic = _mm256_andnot_pd(cmp,mask);

                I = _mm256_or_pd(I,Ic);
                I = _mm256_andnot_pd(I,mask);

                _mm256_stream_pd(it_dst1,I);
                _mm256_stream_pd(it_dst2,Ic);

#else
                __m128 I = _mm_load_pd(reinterpret_cast<const __m128*>(it_src);

                __m128 cmp = _mm_cmpgt_pd(I,zeros);

                __m128 Ic = _mm_andnot_pd(cmp,mask);

                I = _mm_or_pd(I,Ic);
                I = _mm_andnot_pd(I,mask);

                _mm_stream_pd(reinterpret_cast<__m128*>(it_dst1,I);
                _mm_stream_pd(reinterpret_cast<__m128*>(it_dst2,Ic);
#endif
            }
        }
    }
    else
    {

        const std::size_t stop = cols.size() - (cols.size()%step);

        for(std::size_t r=rows.begin(); r<rows.end(); r++,
                                                      src+=this->_core->_src_step1,
                                                      dst1+=this->_core->_dst1_step1,
                                                      dst2+=this->_core->_dst2_step1)
        {
            const_pointer it_src = src;
            pointer it_dst1 = dst1;
            pointer it_dst2 = dst2;

            std::size_t c=0;

            for(;c<stop; c+=step,
                         it_src+=step,
                         it_dst1+=step,
                         it_dst2+=step)
            {
#if CV_AVX2

                __m256d I = _mm256_loadu_pd(it_src);
                __m256d cmp = _mm256_cmp_pd(I,zeros,_CMP_GT_OQ);

                __m256d Ic = _mm256_andnot_pd(cmp,mask);

                I = _mm256_or_pd(I,Ic);
                I = _mm256_andnot_pd(I,mask);

                _mm256_storeu_pd(it_dst1,I);
                _mm256_storeu_pd(it_dst2,Ic);

#else


                __m128 I = _mm_loadu_pd(reinterpret_cast<const __m128*>(it_src);

                __m128 cmp = _mm_cmpgt_pd(I,zeros);

                __m128 Ic = _mm_andnot_pd(cmp,mask);

                I = _mm_or_pd(I,Ic);
                I = _mm_andnot_pd(I,mask);

                _mm_storeu_pd(reinterpret_cast<__m128*>(it_dst1,I);
                _mm_storeu_pd(reinterpret_cast<__m128*>(it_dst2,Ic);
#endif
            }

            for(;c<cols.size();c++,
                               it_src++,
                               it_dst1++,
                               it_dst2++)
            {
                uint I = *it_src > 0. ? 0xFFFFFFFF : 0x0;
                uint Ic = ~I;

                I = ~(I | Ic);

                *it_dst1 = I;
                *it_dst2 = Ic;
            }
        }
    }

}


#else

template<>
void step1_swap<uchar>::operator()(const tbb::blocked_range<std::size_t>& tmp)const
{

    tbb::blocked_range<std::size_t> range(tmp.begin() * this->_core->_grain,std::min( tmp.end() * this->_core->_grain, this->_core->_cols));

    const_pointer src = this->_core->_src + range.begin() * this->_core->_src_step1;
    pointer dst1 = this->_core->_dst1 + range.begin() * this->_core->_dst1_step1;
    pointer dst2 = this->_core->_dst2 + range.begin() * this->_core->_dst2_step1;

    const std::size_t step = 4;

    const std::size_t stop = this->_core->_cols - (this->_core->_cols%step);

    for(std::size_t r=range.begin(); r<range.end(); r++,
        src+=this->_core->_src_step1,
        dst1+=this->_core->_dst1_step1,
        dst2+=this->_core->_dst2_step1)
    {
        const_pointer it_src = src;
        pointer it_dst1 = dst1;
        pointer it_dst2 = dst2;

        std::size_t c=0;

        for(;c<stop; c+=step,
            it_src+=step,
            it_dst1+=step,
            it_dst2+=step)
        {
            value_type I = *it_src > 0 ? 0xFF : 0x0;
            value_type Ic = ~I;


            I = ~(I | Ic);

            *it_dst1 = I;
            *it_dst2 = Ic;


            I = *(it_src+1) > 0 ? 0xFF : 0x0;
            Ic = ~I;

            I = ~(I | Ic);

            *(it_dst1+1) = I;
            *(it_dst2+1) = Ic;


            I = *(it_src+2) > 0 ? 0xFF : 0x0;
            Ic = ~I;

            I = ~(I | Ic);

            *(it_dst1+2) = I;
            *(it_dst2+2) = Ic;


            I = *(it_src+3) > 0 ? 0xFF : 0x0;
            Ic = ~I;

            I = ~(I | Ic);

            *(it_dst1+3) = I;
            *(it_dst2+3) = Ic;

        }

        for(;c<this->_core->_cols; c++,
            it_src++,
            it_dst1++,
            it_dst2++)
        {
            value_type I = *it_src > 0 ? 0xFF : 0x0;
            value_type Ic = ~I;

            I = ~(I | Ic);

            *it_dst1 = I;
            *it_dst2 = Ic;
        }
    }
}

template<>
void step1_swap<uchar>::operator()(const tbb::blocked_range2d<std::size_t>& range)const
{
    const tbb::blocked_range<std::size_t>& rows = range.rows();
    const tbb::blocked_range<std::size_t>& tmp = range.cols();

    tbb::blocked_range<std::size_t> cols(tmp.begin() * this->_core->_grain, std::min( tmp.end() * this->_core->_grain, this->_core->_cols));

    const_pointer src = this->_core->_src + rows.begin() * this->_core->_src_step1 + cols.begin();
    pointer dst1 = this->_core->_dst1 + rows.begin() * this->_core->_dst1_step1 + cols.begin();
    pointer dst2 = this->_core->_dst2 + rows.begin() * this->_core->_dst2_step1 + cols.begin();

    static const std::size_t step = 4;


    const std::size_t stop = cols.size() - (cols.size()%step);


    for(std::size_t r=rows.begin(); r<rows.end(); r++,
        src+=this->_core->_src_step1,
        dst1+=this->_core->_dst1_step1,
        dst2+=this->_core->_dst2_step1)
    {
        const_pointer it_src = src;
        pointer it_dst1 = dst1;
        pointer it_dst2 = dst2;

        std::size_t c=0;

        for(;c<stop; c+=step,
            it_src+=step,
            it_dst1+=step,
            it_dst2+=step)
        {

            value_type I = *it_src > 0 ? 0xFF : 0x0;
            value_type Ic = ~I;

            I = ~(I | Ic);

            *it_dst1 = I;
            *it_dst2 = Ic;


            I = *(it_src+1) > 0 ? 0xFF : 0x0;
            Ic = ~I;

            I = ~(I | Ic);

            *(it_dst1+1) = I;
            *(it_dst2+1) = Ic;


            I = *(it_src+2) > 0 ? 0xFF : 0x0;
            Ic = ~I;

            I = ~(I | Ic);

            *(it_dst1+2) = I;
            *(it_dst2+2) = Ic;


            I = *(it_src+3) > 0 ? 0xFF : 0x0;
            Ic = ~I;

            I = ~(I | Ic);

            *(it_dst1+3) = I;
            *(it_dst2+3) = Ic;

        }


        for(;c<cols.size();c++,
            it_src++,
            it_dst1++,
            it_dst2++)
        {
            value_type I = *it_src > 0 ? 0xFF : 0x0;
            value_type Ic = ~I;

            I = ~(I | Ic);

            *it_dst1 = I;
            *it_dst2 = Ic;
        }
    }
}


template<>
void step1_swap<schar>::operator()(const tbb::blocked_range<std::size_t>& tmp)const
{

    tbb::blocked_range<std::size_t> range(tmp.begin() * this->_core->_grain,std::min( tmp.end() * this->_core->_grain, this->_core->_cols));

    const_pointer src = this->_core->_src + range.begin() * this->_core->_src_step1;
    pointer dst1 = this->_core->_dst1 + range.begin() * this->_core->_dst1_step1;
    pointer dst2 = this->_core->_dst2 + range.begin() * this->_core->_dst2_step1;

    const std::size_t step = 4;

    const std::size_t stop = this->_core->_cols - (this->_core->_cols%step);

    for(std::size_t r=range.begin(); r<range.end(); r++,
        src+=this->_core->_src_step1,
        dst1+=this->_core->_dst1_step1,
        dst2+=this->_core->_dst2_step1)
    {
        const_pointer it_src = src;
        pointer it_dst1 = dst1;
        pointer it_dst2 = dst2;

        std::size_t c=0;

        for(;c<stop; c+=step,
            it_src+=step,
            it_dst1+=step,
            it_dst2+=step)
        {
            value_type I = *it_src > 0 ? 0xFF : 0x0;
            value_type Ic = ~I;


            I = ~(I | Ic);

            *it_dst1 = I;
            *it_dst2 = Ic;


            I = *(it_src+1) > 0 ? 0xFF : 0x0;
            Ic = ~I;

            I = ~(I | Ic);

            *(it_dst1+1) = I;
            *(it_dst2+1) = Ic;


            I = *(it_src+2) > 0 ? 0xFF : 0x0;
            Ic = ~I;

            I = ~(I | Ic);

            *(it_dst1+2) = I;
            *(it_dst2+2) = Ic;


            I = *(it_src+3) > 0 ? 0xFF : 0x0;
            Ic = ~I;

            I = ~(I | Ic);

            *(it_dst1+3) = I;
            *(it_dst2+3) = Ic;

        }

        for(;c<this->_core->_cols; c++,
            it_src++,
            it_dst1++,
            it_dst2++)
        {
            value_type I = *it_src > 0 ? 0xFF : 0x0;
            value_type Ic = ~I;

            I = ~(I | Ic);

            *it_dst1 = I;
            *it_dst2 = Ic;
        }
    }
}

template<>
void step1_swap<schar>::operator()(const tbb::blocked_range2d<std::size_t>& range)const
{
    const tbb::blocked_range<std::size_t>& rows = range.rows();
    const tbb::blocked_range<std::size_t>& tmp = range.cols();

    tbb::blocked_range<std::size_t> cols(tmp.begin() * this->_core->_grain, std::min( tmp.end() * this->_core->_grain, this->_core->_cols));

    const_pointer src = this->_core->_src + rows.begin() * this->_core->_src_step1 + cols.begin();
    pointer dst1 = this->_core->_dst1 + rows.begin() * this->_core->_dst1_step1 + cols.begin();
    pointer dst2 = this->_core->_dst2 + rows.begin() * this->_core->_dst2_step1 + cols.begin();

    static const std::size_t step = 4;


    const std::size_t stop = cols.size() - (cols.size()%step);


    for(std::size_t r=rows.begin(); r<rows.end(); r++,
        src+=this->_core->_src_step1,
        dst1+=this->_core->_dst1_step1,
        dst2+=this->_core->_dst2_step1)
    {
        const_pointer it_src = src;
        pointer it_dst1 = dst1;
        pointer it_dst2 = dst2;

        std::size_t c=0;

        for(;c<stop; c+=step,
            it_src+=step,
            it_dst1+=step,
            it_dst2+=step)
        {

            value_type I = *it_src > 0 ? 0xFF : 0x0;
            value_type Ic = ~I;

            I = ~(I | Ic);

            *it_dst1 = I;
            *it_dst2 = Ic;


            I = *(it_src+1) > 0 ? 0xFF : 0x0;
            Ic = ~I;

            I = ~(I | Ic);

            *(it_dst1+1) = I;
            *(it_dst2+1) = Ic;


            I = *(it_src+2) > 0 ? 0xFF : 0x0;
            Ic = ~I;

            I = ~(I | Ic);

            *(it_dst1+2) = I;
            *(it_dst2+2) = Ic;


            I = *(it_src+3) > 0 ? 0xFF : 0x0;
            Ic = ~I;

            I = ~(I | Ic);

            *(it_dst1+3) = I;
            *(it_dst2+3) = Ic;

        }


        for(;c<cols.size();c++,
            it_src++,
            it_dst1++,
            it_dst2++)
        {
            value_type I = *it_src > 0 ? 0xFF : 0x0;
            value_type Ic = ~I;

            I = ~(I | Ic);

            *it_dst1 = I;
            *it_dst2 = Ic;
        }
    }
}

template<>
void step1_swap<ushort>::operator()(const tbb::blocked_range<std::size_t>& tmp)const
{

    tbb::blocked_range<std::size_t> range(tmp.begin() * this->_core->_grain,std::min( tmp.end() * this->_core->_grain, this->_core->_cols));

    const_pointer src = this->_core->_src + range.begin() * this->_core->_src_step1;
    pointer dst1 = this->_core->_dst1 + range.begin() * this->_core->_dst1_step1;
    pointer dst2 = this->_core->_dst2 + range.begin() * this->_core->_dst2_step1;

    const std::size_t step = 4;

    const std::size_t stop = this->_core->_cols - (this->_core->_cols%step);

    for(std::size_t r=range.begin(); r<range.end(); r++,
        src+=this->_core->_src_step1,
        dst1+=this->_core->_dst1_step1,
        dst2+=this->_core->_dst2_step1)
    {
        const_pointer it_src = src;
        pointer it_dst1 = dst1;
        pointer it_dst2 = dst2;

        std::size_t c=0;

        for(;c<stop; c+=step,
            it_src+=step,
            it_dst1+=step,
            it_dst2+=step)
        {
            value_type I = *it_src > 0 ? 0xFFFF : 0x0;
            value_type Ic = ~I;


            I = ~(I | Ic);

            *it_dst1 = I;
            *it_dst2 = Ic;


            I = *(it_src+1) > 0 ? 0xFFFF : 0x0;
            Ic = ~I;

            I = ~(I | Ic);

            *(it_dst1+1) = I;
            *(it_dst2+1) = Ic;


            I = *(it_src+2) > 0 ? 0xFFFF : 0x0;
            Ic = ~I;

            I = ~(I | Ic);

            *(it_dst1+2) = I;
            *(it_dst2+2) = Ic;


            I = *(it_src+3) > 0 ? 0xFFFF : 0x0;
            Ic = ~I;

            I = ~(I | Ic);

            *(it_dst1+3) = I;
            *(it_dst2+3) = Ic;

        }

        for(;c<this->_core->_cols; c++,
            it_src++,
            it_dst1++,
            it_dst2++)
        {
            value_type I = *it_src > 0 ? 0xFFFF : 0x0;
            value_type Ic = ~I;

            I = ~(I | Ic);

            *it_dst1 = I;
            *it_dst2 = Ic;
        }
    }
}

template<>
void step1_swap<ushort>::operator()(const tbb::blocked_range2d<std::size_t>& range)const
{
    const tbb::blocked_range<std::size_t>& rows = range.rows();
    const tbb::blocked_range<std::size_t>& tmp = range.cols();

    tbb::blocked_range<std::size_t> cols(tmp.begin() * this->_core->_grain, std::min( tmp.end() * this->_core->_grain, this->_core->_cols));

    const_pointer src = this->_core->_src + rows.begin() * this->_core->_src_step1 + cols.begin();
    pointer dst1 = this->_core->_dst1 + rows.begin() * this->_core->_dst1_step1 + cols.begin();
    pointer dst2 = this->_core->_dst2 + rows.begin() * this->_core->_dst2_step1 + cols.begin();

    static const std::size_t step = 4;


    const std::size_t stop = cols.size() - (cols.size()%step);


    for(std::size_t r=rows.begin(); r<rows.end(); r++,
        src+=this->_core->_src_step1,
        dst1+=this->_core->_dst1_step1,
        dst2+=this->_core->_dst2_step1)
    {
        const_pointer it_src = src;
        pointer it_dst1 = dst1;
        pointer it_dst2 = dst2;

        std::size_t c=0;

        for(;c<stop; c+=step,
            it_src+=step,
            it_dst1+=step,
            it_dst2+=step)
        {

            value_type I = *it_src > 0 ? 0xFFFF : 0x0;
            value_type Ic = ~I;

            I = ~(I | Ic);

            *it_dst1 = I;
            *it_dst2 = Ic;


            I = *(it_src+1) > 0 ? 0xFFFF : 0x0;
            Ic = ~I;

            I = ~(I | Ic);

            *(it_dst1+1) = I;
            *(it_dst2+1) = Ic;


            I = *(it_src+2) > 0 ? 0xFFFF : 0x0;
            Ic = ~I;

            I = ~(I | Ic);

            *(it_dst1+2) = I;
            *(it_dst2+2) = Ic;


            I = *(it_src+3) > 0 ? 0xFFFF : 0x0;
            Ic = ~I;

            I = ~(I | Ic);

            *(it_dst1+3) = I;
            *(it_dst2+3) = Ic;

        }


        for(;c<cols.size();c++,
            it_src++,
            it_dst1++,
            it_dst2++)
        {
            value_type I = *it_src > 0 ? 0xFFFF : 0x0;
            value_type Ic = ~I;

            I = ~(I | Ic);

            *it_dst1 = I;
            *it_dst2 = Ic;
        }
    }
}

template<>
void step1_swap<short>::operator()(const tbb::blocked_range<std::size_t>& tmp)const
{

    tbb::blocked_range<std::size_t> range(tmp.begin() * this->_core->_grain,std::min( tmp.end() * this->_core->_grain, this->_core->_cols));

    const_pointer src = this->_core->_src + range.begin() * this->_core->_src_step1;
    pointer dst1 = this->_core->_dst1 + range.begin() * this->_core->_dst1_step1;
    pointer dst2 = this->_core->_dst2 + range.begin() * this->_core->_dst2_step1;

    const std::size_t step = 4;

    const std::size_t stop = this->_core->_cols - (this->_core->_cols%step);

    for(std::size_t r=range.begin(); r<range.end(); r++,
        src+=this->_core->_src_step1,
        dst1+=this->_core->_dst1_step1,
        dst2+=this->_core->_dst2_step1)
    {
        const_pointer it_src = src;
        pointer it_dst1 = dst1;
        pointer it_dst2 = dst2;

        std::size_t c=0;

        for(;c<stop; c+=step,
            it_src+=step,
            it_dst1+=step,
            it_dst2+=step)
        {
            value_type I = *it_src > 0 ? 0xFFFF : 0x0;
            value_type Ic = ~I;


            I = ~(I | Ic);

            *it_dst1 = I;
            *it_dst2 = Ic;


            I = *(it_src+1) > 0 ? 0xFFFF : 0x0;
            Ic = ~I;

            I = ~(I | Ic);

            *(it_dst1+1) = I;
            *(it_dst2+1) = Ic;


            I = *(it_src+2) > 0 ? 0xFFFF : 0x0;
            Ic = ~I;

            I = ~(I | Ic);

            *(it_dst1+2) = I;
            *(it_dst2+2) = Ic;


            I = *(it_src+3) > 0 ? 0xFFFF : 0x0;
            Ic = ~I;

            I = ~(I | Ic);

            *(it_dst1+3) = I;
            *(it_dst2+3) = Ic;

        }

        for(;c<this->_core->_cols; c++,
            it_src++,
            it_dst1++,
            it_dst2++)
        {
            value_type I = *it_src > 0 ? 0xFFFF : 0x0;
            value_type Ic = ~I;

            I = ~(I | Ic);

            *it_dst1 = I;
            *it_dst2 = Ic;
        }
    }
}

template<>
void step1_swap<short>::operator()(const tbb::blocked_range2d<std::size_t>& range)const
{
    const tbb::blocked_range<std::size_t>& rows = range.rows();
    const tbb::blocked_range<std::size_t>& tmp = range.cols();

    tbb::blocked_range<std::size_t> cols(tmp.begin() * this->_core->_grain, std::min( tmp.end() * this->_core->_grain, this->_core->_cols));

    const_pointer src = this->_core->_src + rows.begin() * this->_core->_src_step1 + cols.begin();
    pointer dst1 = this->_core->_dst1 + rows.begin() * this->_core->_dst1_step1 + cols.begin();
    pointer dst2 = this->_core->_dst2 + rows.begin() * this->_core->_dst2_step1 + cols.begin();

    static const std::size_t step = 4;


    const std::size_t stop = cols.size() - (cols.size()%step);


    for(std::size_t r=rows.begin(); r<rows.end(); r++,
        src+=this->_core->_src_step1,
        dst1+=this->_core->_dst1_step1,
        dst2+=this->_core->_dst2_step1)
    {
        const_pointer it_src = src;
        pointer it_dst1 = dst1;
        pointer it_dst2 = dst2;

        std::size_t c=0;

        for(;c<stop; c+=step,
            it_src+=step,
            it_dst1+=step,
            it_dst2+=step)
        {

            value_type I = *it_src > 0 ? 0xFFFF : 0x0;
            value_type Ic = ~I;

            I = ~(I | Ic);

            *it_dst1 = I;
            *it_dst2 = Ic;


            I = *(it_src+1) > 0 ? 0xFFFF : 0x0;
            Ic = ~I;

            I = ~(I | Ic);

            *(it_dst1+1) = I;
            *(it_dst2+1) = Ic;


            I = *(it_src+2) > 0 ? 0xFFFF : 0x0;
            Ic = ~I;

            I = ~(I | Ic);

            *(it_dst1+2) = I;
            *(it_dst2+2) = Ic;


            I = *(it_src+3) > 0 ? 0xFFFF : 0x0;
            Ic = ~I;

            I = ~(I | Ic);

            *(it_dst1+3) = I;
            *(it_dst2+3) = Ic;

        }


        for(;c<cols.size();c++,
            it_src++,
            it_dst1++,
            it_dst2++)
        {
            value_type I = *it_src > 0 ? 0xFFFF : 0x0;
            value_type Ic = ~I;

            I = ~(I | Ic);

            *it_dst1 = I;
            *it_dst2 = Ic;
        }
    }
}


template<>
void step1_swap<int>::operator()(const tbb::blocked_range<std::size_t>& tmp)const
{

    tbb::blocked_range<std::size_t> range(tmp.begin() * this->_core->_grain,std::min( tmp.end() * this->_core->_grain, this->_core->_cols));

    const_pointer src = this->_core->_src + range.begin() * this->_core->_src_step1;
    pointer dst1 = this->_core->_dst1 + range.begin() * this->_core->_dst1_step1;
    pointer dst2 = this->_core->_dst2 + range.begin() * this->_core->_dst2_step1;

    const std::size_t step = 4;

    const std::size_t stop = this->_core->_cols - (this->_core->_cols%step);

    for(std::size_t r=range.begin(); r<range.end(); r++,
        src+=this->_core->_src_step1,
        dst1+=this->_core->_dst1_step1,
        dst2+=this->_core->_dst2_step1)
    {
        const_pointer it_src = src;
        pointer it_dst1 = dst1;
        pointer it_dst2 = dst2;

        std::size_t c=0;

        for(;c<stop; c+=step,
            it_src+=step,
            it_dst1+=step,
            it_dst2+=step)
        {
            value_type I = *it_src > 0 ? 0xFFFFFFFF : 0x0;
            value_type Ic = ~I;


            I = ~(I | Ic);

            *it_dst1 = I;
            *it_dst2 = Ic;


            I = *(it_src+1) > 0 ? 0xFFFFFFFF : 0x0;
            Ic = ~I;

            I = ~(I | Ic);

            *(it_dst1+1) = I;
            *(it_dst2+1) = Ic;


            I = *(it_src+2) > 0 ? 0xFFFFFFFF : 0x0;
            Ic = ~I;

            I = ~(I | Ic);

            *(it_dst1+2) = I;
            *(it_dst2+2) = Ic;


            I = *(it_src+3) > 0 ? 0xFFFFFFFF : 0x0;
            Ic = ~I;

            I = ~(I | Ic);

            *(it_dst1+3) = I;
            *(it_dst2+3) = Ic;

        }

        for(;c<this->_core->_cols; c++,
            it_src++,
            it_dst1++,
            it_dst2++)
        {
            value_type I = *it_src > 0 ? 0xFFFFFFFF : 0x0;
            value_type Ic = ~I;

            I = ~(I | Ic);

            *it_dst1 = I;
            *it_dst2 = Ic;
        }
    }
}

template<>
void step1_swap<int>::operator()(const tbb::blocked_range2d<std::size_t>& range)const
{
    const tbb::blocked_range<std::size_t>& rows = range.rows();
    const tbb::blocked_range<std::size_t>& tmp = range.cols();

    tbb::blocked_range<std::size_t> cols(tmp.begin() * this->_core->_grain, std::min( tmp.end() * this->_core->_grain, this->_core->_cols));

    const_pointer src = this->_core->_src + rows.begin() * this->_core->_src_step1 + cols.begin();
    pointer dst1 = this->_core->_dst1 + rows.begin() * this->_core->_dst1_step1 + cols.begin();
    pointer dst2 = this->_core->_dst2 + rows.begin() * this->_core->_dst2_step1 + cols.begin();

    static const std::size_t step = 4;


    const std::size_t stop = cols.size() - (cols.size()%step);


    for(std::size_t r=rows.begin(); r<rows.end(); r++,
        src+=this->_core->_src_step1,
        dst1+=this->_core->_dst1_step1,
        dst2+=this->_core->_dst2_step1)
    {
        const_pointer it_src = src;
        pointer it_dst1 = dst1;
        pointer it_dst2 = dst2;

        std::size_t c=0;

        for(;c<stop; c+=step,
            it_src+=step,
            it_dst1+=step,
            it_dst2+=step)
        {

            value_type I = *it_src > 0 ? 0xFFFFFFFF : 0x0;
            value_type Ic = ~I;

            I = ~(I | Ic);

            *it_dst1 = I;
            *it_dst2 = Ic;


            I = *(it_src+1) > 0 ? 0xFFFFFFFF : 0x0;
            Ic = ~I;

            I = ~(I | Ic);

            *(it_dst1+1) = I;
            *(it_dst2+1) = Ic;


            I = *(it_src+2) > 0 ? 0xFFFFFFFF : 0x0;
            Ic = ~I;

            I = ~(I | Ic);

            *(it_dst1+2) = I;
            *(it_dst2+2) = Ic;


            I = *(it_src+3) > 0 ? 0xFFFFFFFF : 0x0;
            Ic = ~I;

            I = ~(I | Ic);

            *(it_dst1+3) = I;
            *(it_dst2+3) = Ic;

        }


        for(;c<cols.size();c++,
            it_src++,
            it_dst1++,
            it_dst2++)
        {
            value_type I = *it_src > 0 ? 0xFFFFFFFF : 0x0;
            value_type Ic = ~I;

            I = ~(I | Ic);

            *it_dst1 = I;
            *it_dst2 = Ic;
        }
    }
}


template<>
void step1_swap<float>::operator()(const tbb::blocked_range<std::size_t>& tmp)const
{

    tbb::blocked_range<std::size_t> range(tmp.begin() * this->_core->_grain,std::min( tmp.end() * this->_core->_grain, this->_core->_cols));

    const_pointer src = this->_core->_src + range.begin() * this->_core->_src_step1;
    pointer dst1 = this->_core->_dst1 + range.begin() * this->_core->_dst1_step1;
    pointer dst2 = this->_core->_dst2 + range.begin() * this->_core->_dst2_step1;

    const std::size_t step = 4;

    const std::size_t stop = this->_core->_cols - (this->_core->_cols%step);

    for(std::size_t r=range.begin(); r<range.end(); r++,
        src+=this->_core->_src_step1,
        dst1+=this->_core->_dst1_step1,
        dst2+=this->_core->_dst2_step1)
    {
        const_pointer it_src = src;
        pointer it_dst1 = dst1;
        pointer it_dst2 = dst2;

        std::size_t c=0;

        for(;c<stop; c+=step,
            it_src+=step,
            it_dst1+=step,
            it_dst2+=step)
        {
            uint I = *it_src > 0.f ? 0xFFFFFFFF : 0x0;
            uint Ic = ~I;


            I = ~(I | Ic);

            *it_dst1 = I;
            *it_dst2 = Ic;


            I = *(it_src+1) > 0.f ? 0xFFFFFFFF : 0x0;
            Ic = ~I;

            I = ~(I | Ic);

            *(it_dst1+1) = I;
            *(it_dst2+1) = Ic;


            I = *(it_src+2) > 0.f ? 0xFFFFFFFF : 0x0;
            Ic = ~I;

            I = ~(I | Ic);

            *(it_dst1+2) = I;
            *(it_dst2+2) = Ic;


            I = *(it_src+3) > 0.f ? 0xFFFFFFFF : 0x0;
            Ic = ~I;

            I = ~(I | Ic);

            *(it_dst1+3) = I;
            *(it_dst2+3) = Ic;

        }

        for(;c<this->_core->_cols; c++,
            it_src++,
            it_dst1++,
            it_dst2++)
        {
            uint I = *it_src > 0.f ? 0xFFFFFFFF : 0x0;
            uint Ic = ~I;

            I = ~(I | Ic);

            *it_dst1 = I;
            *it_dst2 = Ic;
        }
    }
}

template<>
void step1_swap<float>::operator()(const tbb::blocked_range2d<std::size_t>& range)const
{
    const tbb::blocked_range<std::size_t>& rows = range.rows();
    const tbb::blocked_range<std::size_t>& tmp = range.cols();

    tbb::blocked_range<std::size_t> cols(tmp.begin() * this->_core->_grain, std::min( tmp.end() * this->_core->_grain, this->_core->_cols));

    const_pointer src = this->_core->_src + rows.begin() * this->_core->_src_step1 + cols.begin();
    pointer dst1 = this->_core->_dst1 + rows.begin() * this->_core->_dst1_step1 + cols.begin();
    pointer dst2 = this->_core->_dst2 + rows.begin() * this->_core->_dst2_step1 + cols.begin();

    static const std::size_t step = 4;


    const std::size_t stop = cols.size() - (cols.size()%step);


    for(std::size_t r=rows.begin(); r<rows.end(); r++,
        src+=this->_core->_src_step1,
        dst1+=this->_core->_dst1_step1,
        dst2+=this->_core->_dst2_step1)
    {
        const_pointer it_src = src;
        pointer it_dst1 = dst1;
        pointer it_dst2 = dst2;

        std::size_t c=0;

        for(;c<stop; c+=step,
            it_src+=step,
            it_dst1+=step,
            it_dst2+=step)
        {

            uint I = *it_src > 0.f ? 0xFFFFFFFF : 0x0;
            uint Ic = ~I;

            I = ~(I | Ic);

            *it_dst1 = I;
            *it_dst2 = Ic;


            I = *(it_src+1) > 0.f ? 0xFFFFFFFF : 0x0;
            Ic = ~I;

            I = ~(I | Ic);

            *(it_dst1+1) = I;
            *(it_dst2+1) = Ic;


            I = *(it_src+2) > 0.f ? 0xFFFFFFFF : 0x0;
            Ic = ~I;

            I = ~(I | Ic);

            *(it_dst1+2) = I;
            *(it_dst2+2) = Ic;


            I = *(it_src+3) > 0.f ? 0xFFFFFFFF : 0x0;
            Ic = ~I;

            I = ~(I | Ic);

            *(it_dst1+3) = I;
            *(it_dst2+3) = Ic;

        }


        for(;c<cols.size();c++,
            it_src++,
            it_dst1++,
            it_dst2++)
        {
            uint I = *it_src > 0.f ? 0xFFFFFFFF : 0x0;
            uint Ic = ~I;

            I = ~(I | Ic);

            *it_dst1 = I;
            *it_dst2 = Ic;
        }
    }
}




template<>
void step1_swap<double>::operator()(const tbb::blocked_range<std::size_t>& tmp)const
{

    tbb::blocked_range<std::size_t> range(tmp.begin() * this->_core->_grain,std::min( tmp.end() * this->_core->_grain, this->_core->_cols));

    const_pointer src = this->_core->_src + range.begin() * this->_core->_src_step1;
    pointer dst1 = this->_core->_dst1 + range.begin() * this->_core->_dst1_step1;
    pointer dst2 = this->_core->_dst2 + range.begin() * this->_core->_dst2_step1;

    const std::size_t step = 4;

    const std::size_t stop = this->_core->_cols - (this->_core->_cols%step);

    for(std::size_t r=range.begin(); r<range.end(); r++,
        src+=this->_core->_src_step1,
        dst1+=this->_core->_dst1_step1,
        dst2+=this->_core->_dst2_step1)
    {
        const_pointer it_src = src;
        pointer it_dst1 = dst1;
        pointer it_dst2 = dst2;

        std::size_t c=0;

        for(;c<stop; c+=step,
            it_src+=step,
            it_dst1+=step,
            it_dst2+=step)
        {
            uint I = *it_src > 0. ? 0xFFFFFFFF : 0x0;
            uint Ic = ~I;


            I = ~(I | Ic);

            *it_dst1 = I;
            *it_dst2 = Ic;


            I = *(it_src+1) > 0. ? 0xFFFFFFFF : 0x0;
            Ic = ~I;

            I = ~(I | Ic);

            *(it_dst1+1) = I;
            *(it_dst2+1) = Ic;


            I = *(it_src+2) > 0. ? 0xFFFFFFFF : 0x0;
            Ic = ~I;

            I = ~(I | Ic);

            *(it_dst1+2) = I;
            *(it_dst2+2) = Ic;


            I = *(it_src+3) > 0. ? 0xFFFFFFFF : 0x0;
            Ic = ~I;

            I = ~(I | Ic);

            *(it_dst1+3) = I;
            *(it_dst2+3) = Ic;

        }

        for(;c<this->_core->_cols; c++,
            it_src++,
            it_dst1++,
            it_dst2++)
        {
            uint I = *it_src > 0. ? 0xFFFFFFFF : 0x0;
            uint Ic = ~I;

            I = ~(I | Ic);

            *it_dst1 = I;
            *it_dst2 = Ic;
        }
    }
}

template<>
void step1_swap<double>::operator()(const tbb::blocked_range2d<std::size_t>& range)const
{
    const tbb::blocked_range<std::size_t>& rows = range.rows();
    const tbb::blocked_range<std::size_t>& tmp = range.cols();

    tbb::blocked_range<std::size_t> cols(tmp.begin() * this->_core->_grain, std::min( tmp.end() * this->_core->_grain, this->_core->_cols));

    const_pointer src = this->_core->_src + rows.begin() * this->_core->_src_step1 + cols.begin();
    pointer dst1 = this->_core->_dst1 + rows.begin() * this->_core->_dst1_step1 + cols.begin();
    pointer dst2 = this->_core->_dst2 + rows.begin() * this->_core->_dst2_step1 + cols.begin();

    static const std::size_t step = 4;


    const std::size_t stop = cols.size() - (cols.size()%step);


    for(std::size_t r=rows.begin(); r<rows.end(); r++,
        src+=this->_core->_src_step1,
        dst1+=this->_core->_dst1_step1,
        dst2+=this->_core->_dst2_step1)
    {
        const_pointer it_src = src;
        pointer it_dst1 = dst1;
        pointer it_dst2 = dst2;

        std::size_t c=0;

        for(;c<stop; c+=step,
            it_src+=step,
            it_dst1+=step,
            it_dst2+=step)
        {

            uint I = *it_src > 0. ? 0xFFFFFFFF : 0x0;
            uint Ic = ~I;

            I = ~(I | Ic);

            *it_dst1 = I;
            *it_dst2 = Ic;


            I = *(it_src+1) > 0. ? 0xFFFFFFFF : 0x0;
            Ic = ~I;

            I = ~(I | Ic);

            *(it_dst1+1) = I;
            *(it_dst2+1) = Ic;


            I = *(it_src+2) > 0. ? 0xFFFFFFFF : 0x0;
            Ic = ~I;

            I = ~(I | Ic);

            *(it_dst1+2) = I;
            *(it_dst2+2) = Ic;


            I = *(it_src+3) > 0. ? 0xFFFFFFFF : 0x0;
            Ic = ~I;

            I = ~(I | Ic);

            *(it_dst1+3) = I;
            *(it_dst2+3) = Ic;

        }


        for(;c<cols.size();c++,
            it_src++,
            it_dst1++,
            it_dst2++)
        {
            uint I = *it_src > 0. ? 0xFFFFFFFF : 0x0;
            uint Ic = ~I;

            I = ~(I | Ic);

            *it_dst1 = I;
            *it_dst2 = Ic;
        }
    }
}

#endif

template<class _Ty>
class step2
{
public:

    typedef typename std::iterator_traits<_Ty*>::value_type value_type;
    typedef typename std::iterator_traits<_Ty*>::pointer pointer;
    typedef typename std::iterator_traits<const _Ty*>::pointer const_pointer;

private:

    struct core_t
    {

        const_pointer _src1;
        const std::size_t _src1_step1;

        const_pointer _src2;
        const std::size_t _src2_step1;

        pointer _dst;
        const std::size_t _dst_step1;

        const std::size_t _cols;
        const std::size_t _grain;

        const bool _is_inline;

#if CV_SSE2
        const bool _is_algn;
#if CV_AVX2
        static const int algn = 32;
        static const int step = 32/sizeof(_Ty);
#else
        static const int algn = 16;
        static const int step = 16/sizeof(_Ty);
#endif

#endif

        inline core_t(cv::Mat_<value_type>& src1,
               cv::Mat_<value_type>& src2,
               cv::Mat_<value_type>& dst,
               const std::size_t& grain):
            _src1(src1[0]),
            _src1_step1(src1.isContinuous() ? 1 : src1.step1()),
            _src2(src2[0]),
            _src2_step1(src2.isContinuous() ? 1 : src2.step1()),
            _dst(dst[0]),
            _dst_step1(dst.isContinuous() ? 1 : dst.step1()),
            _cols(src1.isContinuous() ? src1.total() : src1.cols),
            _grain(grain),
            _is_inline(src1.isContinuous() && src2.isContinuous() && dst.isContinuous())
  #if CV_SSE2
            ,_is_algn(((src1.step%algn)==0) &&
                      ((src2.step%algn)==0) &&
                      ((dst.step%algn)==0) &&
                      ((src1.cols%step)==0) &&
                      ((src2.cols%step)==0) &&
                      ((dst.cols%step)==0) &&
                      src1.isContinuous() &&
                      src2.isContinuous() &&
                      dst.isContinuous() )
  #endif
        {}

        ~core_t() = default;
    };

    std::shared_ptr<core_t> _core;

public:

    inline step2(cv::Mat_<value_type>& src1,cv::Mat_<value_type>& src2,cv::Mat_<value_type>& dst,const std::size_t& grain):
        _core(new core_t(src1,src2,dst,grain))
    {}

    inline step2(const step2& obj):
        _core(obj._core)
    {}

    inline step2(step2&& obj):
        _core(std::move(obj._core))
    {}

    ~step2() = default;

    void operator()(const tbb::blocked_range<std::size_t>& tmp)const;

    void operator()(const tbb::blocked_range2d<std::size_t>& ranges)const;

};

#if CV_SSE2




template<class _Ty>
void step2<_Ty>::operator()(const tbb::blocked_range<std::size_t>& tmp)const
{
    const tbb::blocked_range<std::size_t> range(tmp.begin() * this->_core->_grain, std::min(tmp.end() * this->_core->_grain, this->_core->_cols));

    mtx.lock();
    std::cout<<"CHECK STOP STEP2 "<<range.begin()<<" "<<range.end()<<" "<<this->_core->_cols<<" "<<this->_core->_grain<<" "<<this->_core->_is_algn<<" "<<this->_core->_is_inline<<" "<<range.begin()<<" "<<range.end()<<std::endl;
    mtx.unlock();


    const_pointer src1 = this->_core->_src1 + range.begin() * this->_core->_src1_step1;
    const_pointer src2 = this->_core->_src2 + range.begin() * this->_core->_src2_step1;
    pointer dst = this->_core->_dst + range.begin() * this->_core->_dst_step1;

    static const std::size_t step = this->_core->step;
#if CV_AVX2
    const __m256i mask = _mm256_set1_epi32(typename std::make_unsigned<value_type>::type(-1));
#else
    const __m128i mask = _mm_set1_epi32(typename std::make_unsigned<value_type>::type(-1));
#endif

    if(this->_core->_is_algn)
    {
        if(this->_core->_is_inline)
        {
            for(std::size_t c=range.begin();c<range.end();c+=step,src1+=step,src2+=step,dst+=step)
            {
#if CV_AVX2
                    __m256i s1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src1));
                    __m256i s2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src2));

                    s1 = _mm256_or_si256(s1,s2);

                    __m256i h = _mm256_andnot_si256(s1,mask);

                    h = _mm256_or_si256(s2,h);

                    _mm256_stream_si256(reinterpret_cast<__m256i*>(dst),h);
#else
                    __m128i s1 = _mm_load_si128(reinterpret_cast<const __m128i*>(src1));
                    __m128i s2 = _mm_load_si128(reinterpret_cast<const __m128i*>(src2));

                    s1 = _mm_or_si128(s1,s2);

                    __m128i h = _mm_andnot_si128(s1,mask);

                    h = _mm_or_si128(s2,h);

                    _mm_stream_si128(reinterpret_cast<__m128i*>(dst),h);
#endif
            }
        }
        else
        {
            for(std::size_t r=range.begin();r<range.end();r++,src1+=this->_core->_src1_step1,src2+=this->_core->_src2_step1,dst+=this->_core->_dst_step1)
            {

                const_pointer it_src1 = src1;
                const_pointer it_src2 = src2;
                pointer it_dst = dst;

                for(std::size_t c=0;c<this->_core->_cols;c+=step,it_src1+=step,it_src2+=step,it_dst+=step)
                {
#if CV_AVX2
                    __m256i s1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(it_src1));
                    __m256i s2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(it_src2));

                    s1 = _mm256_or_si256(s1,s2);

                    __m256i h = _mm256_andnot_si256(s1,mask);

                    h = _mm256_or_si256(s2,h);

                    _mm256_stream_si256(reinterpret_cast<__m256i*>(it_dst),h);
#else
                    __m128i s1 = _mm_load_si128(reinterpret_cast<const __m128i*>(it_src1));
                    __m128i s2 = _mm_load_si128(reinterpret_cast<const __m128i*>(it_src2));

                    s1 = _mm_or_si128(s1,s2);

                    __m128i h = _mm_andnot_si128(s1,mask);

                    h = _mm_or_si128(s2,h);

                    _mm_stream_si128(reinterpret_cast<__m128i*>(it_dst),h);
#endif
                }

            }
        }

    }
    else
    {
        const std::size_t stop = this->_core->_cols - (this->_core->_cols%step);


        for(std::size_t r=range.begin();r<range.end(); r++,
                                                       src1+=this->_core->_src1_step1,
                                                       src2+=this->_core->_src2_step1,
                                                       dst+=this->_core->_dst_step1)
        {

            const_pointer it_src1 = src1;
            const_pointer it_src2 = src2;
            pointer it_dst = dst;

            std::size_t c=0;

            for(;c<stop;c+=step,it_src1+=step,it_src2+=step,it_dst+=step)
            {
#if CV_AVX2
                __m256i s1 = _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(it_src1));
                __m256i s2 = _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(it_src2));

                s1 = _mm256_or_si256(s1,s2);

                __m256i h = _mm256_andnot_si256(s1,mask);

                h = _mm256_or_si256(s2,h);

                _mm256_storeu_si256(reinterpret_cast<__m256i*>(it_dst),h);
#else

#if CV_SSE3
                __m128i s1 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(it_src1));
                __m128i s2 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(it_src2));
#else
                __m128i s1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(it_src1));
                __m128i s2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(it_src2));
#endif

                s1 = _mm_or_si128(s1,s2);

                __m128i h = _mm_andnot_si128(s1,mask);

                h = _mm_or_si128(s2,h);

                _mm_storeu_si128(reinterpret_cast<__m128i*>(it_dst),h);
#endif
            }

            for(;c<this->_core->_cols;c++,it_src1++,it_src2++,it_dst++)
            {
                value_type F = *it_src1;
                value_type Ic = *it_src2;

                F |= Ic;

                *it_dst = ((~F) | Ic);
            }

        }

    }


}


template<>
void step2<float>::operator()(const tbb::blocked_range<std::size_t>& tmp)const
{

    const tbb::blocked_range<std::size_t> range(tmp.begin() * this->_core->_grain,std::min(tmp.end() * this->_core->_grain,this->_core->_cols));

    const_pointer src1 = this->_core->_src1 + range.begin() * this->_core->_src1_step1;
    const_pointer src2 = this->_core->_src2 + range.begin() * this->_core->_src2_step1;
    pointer dst = this->_core->_dst + range.begin() * this->_core->_dst_step1;

    static const std::size_t step = this->_core->step;
#if CV_AVX2
    static const __m256 mask = _mm256_set1_ps(0xFFFFFFFF);
#else
    static const __m128 mask = _mm_set1_ps(0xFFFFFFFF);
#endif

    if(this->_core->_is_algn)
    {

        for(std::size_t r=range.begin();r<range.end();r++,src1+=this->_core->_src1_step1,src2+=this->_core->_src2_step1,dst+=this->_core->_dst_step1)
        {

            const_pointer it_src1 = src1;
            const_pointer it_src2 = src2;
            pointer it_dst = dst;

            for(std::size_t c=0;c<this->_core->_cols;c+=step,it_src1+=step,it_src2+=step,it_dst+=step)
            {
#if CV_AVX2
                __m256 s1 = _mm256_load_ps(it_src1);
                __m256 s2 = _mm256_load_ps(it_src2);

                s1 = _mm256_or_ps(s1,s2);

                __m256 h = _mm256_andnot_ps(s1,mask);

                h = _mm256_or_ps(s2,h);

                _mm256_stream_ps(it_dst,h);
#else
                __m128 s1 = _mm_load_ps(it_src1);
                __m128 s2 = _mm_load_ps(it_src2);

                s1 = _mm_or_ps(s1,s2);

                __m128 h = _mm_andnot_ps(s1,mask);

                h = _mm_or_ps(s2,h);

                _mm_stream_ps(it_dst,h);
#endif
            }

        }

    }
    else
    {
        const std::size_t stop = this->_core->_cols - (this->_core->_cols%step);

        for(std::size_t r=range.begin();r<range.end(); r++,
                                                       src1+=this->_core->_src1_step1,
                                                       src2+=this->_core->_src2_step1,
                                                       dst+=this->_core->_dst_step1)
        {

            const_pointer it_src1 = src1;
            const_pointer it_src2 = src2;
            pointer it_dst = dst;

            std::size_t c=0;

            for(;c<stop;c+=step,it_src1+=step,it_src2+=step,it_dst+=step)
            {
#if CV_AVX2
                __m256 s1 = _mm256_loadu_ps(it_src1);
                __m256 s2 = _mm256_loadu_ps(it_src2);

                s1 = _mm256_or_ps(s1,s2);

                __m256 h = _mm256_andnot_ps(s1,mask);

                h = _mm256_or_ps(s2,h);

                _mm256_storeu_ps(it_dst,h);
#else
                __m128 s1 = _mm_loadu_ps(it_src1);
                __m128 s2 = _mm_loadu_ps(it_src2);

                s1 = _mm_or_ps(s1,s2);

                __m128 h = _mm_andnot_ps(s1,mask);

                h = _mm_or_ps(s2,h);

                _mm_storeu_ps(it_dst,h);
#endif
            }

            for(;c<this->_core->_cols;c++,it_src1++,it_src2++,it_dst++)
            {
                uint F = *it_src1;
                uint Ic = *it_src2;

                F |= Ic;

                *it_dst = ((~F) | Ic);
            }

        }

    }


}


template<>
void step2<double>::operator()(const tbb::blocked_range<std::size_t>& tmp)const
{

    const tbb::blocked_range<std::size_t> range(tmp.begin() * this->_core->_grain,std::min(tmp.end() * this->_core->_grain,this->_core->_cols));

    const_pointer src1 = this->_core->_src1 + range.begin() * this->_core->_src1_step1;
    const_pointer src2 = this->_core->_src2 + range.begin() * this->_core->_src2_step1;
    pointer dst = this->_core->_dst + range.begin() * this->_core->_dst_step1;

    static const std::size_t step = this->_core->step;
#if CV_AVX2
    static const __m256d mask = _mm256_set1_pd(0xFFFFFFFFFFFFFFFF);
#else
    static const __m128d mask = _mm_set1_pd(0xFFFFFFFFFFFFFFFF);
#endif

    if(this->_core->_is_algn)
    {

        for(std::size_t r=range.begin();r<range.end();r++,src1+=this->_core->_src1_step1,src2+=this->_core->_src2_step1,dst+=this->_core->_dst_step1)
        {

            const_pointer it_src1 = src1;
            const_pointer it_src2 = src2;
            pointer it_dst = dst;

            for(std::size_t c=0;c<this->_core->_cols;c+=step,it_src1+=step,it_src2+=step,it_dst+=step)
            {
#if CV_AVX2
                __m256d s1 = _mm256_load_pd(it_src1);
                __m256d s2 = _mm256_load_pd(it_src2);

                s1 = _mm256_or_pd(s1,s2);

                __m256d h = _mm256_andnot_pd(s1,mask);

                h = _mm256_or_pd(s2,h);

                _mm256_stream_pd(it_dst,h);
#else
                __m128d s1 = _mm_load_pd(it_src1);
                __m128d s2 = _mm_load_pd(it_src2);

                s1 = _mm_or_pd(s1,s2);

                __m128d h = _mm_andnot_pd(s1,mask);

                h = _mm_or_pd(s2,h);

                _mm_stream_pd(it_dst,h);
#endif
            }

        }

    }
    else
    {
        const std::size_t stop = this->_core->_cols - (this->_core->_cols%step);

        for(std::size_t r=range.begin();r<range.end(); r++,
                                                       src1+=this->_core->_src1_step1,
                                                       src2+=this->_core->_src2_step1,
                                                       dst+=this->_core->_dst_step1)
        {

            const_pointer it_src1 = src1;
            const_pointer it_src2 = src2;
            pointer it_dst = dst;

            std::size_t c=0;

            for(;c<stop;c+=step,it_src1+=step,it_src2+=step,it_dst+=step)
            {
#if CV_AVX2
                __m256d s1 = _mm256_loadu_pd(it_src1);
                __m256d s2 = _mm256_loadu_pd(it_src2);

                s1 = _mm256_or_pd(s1,s2);

                __m256d h = _mm256_andnot_pd(s1,mask);

                h = _mm256_or_pd(s2,h);

                _mm256_storeu_pd(it_dst,h);
#else

                __m128d s1 = _mm_loadu_pd(it_src1);
                __m128d s2 = _mm_loadu_pd(it_src2);

                s1 = _mm_or_pd(s1,s2);

                __m128d h = _mm_andnot_pd(s1,mask);

                h = _mm_or_pd(s2,h);

                _mm_storeu_pd(it_dst,h);
#endif
            }

            for(;c<this->_core->_cols;c++,it_src1++,it_src2++,it_dst++)
            {
                uint F = *it_src1;
                uint Ic = *it_src2;

                F |= Ic;

                *it_dst = ((~F) | Ic);
            }

        }

    }


}



#else
template<>
void step2<uchar>::operator()(const tbb::blocked_range<std::size_t>& tmp)const
{

    const tbb::blocked_range<std::size_t> range(tmp.begin() * this->_core->_grain,std::min(tmp.end() * this->_core->_grain,this->_core->_cols));

    const_pointer src1 = this->_core->_src1 + range.begin() * this->_core->_src1_step1;
    const_pointer src2 = this->_core->_src2 + range.begin() * this->_core->_src2_step1;
    pointer dst = this->_core->_dst + range.begin() * this->_core->_dst_step1;

    const std::size_t step = 4;

    const std::size_t stop = this->_core->_cols - (this->_core->_cols%step);

    for(std::size_t r=range.begin();r<range.end(); r++,
        src1+=this->_core->_src1_step1,
        src2+=this->_core->_src2_step1,
        dst+=this->_core->_dst_step1)
    {

        const_pointer it_src1 = src1;
        const_pointer it_src2 = src2;
        pointer it_dst = dst;

        std::size_t c=0;

        for(;c<stop;c+=step,it_src1+=step,it_src2+=step,it_dst+=step)
        {

            value_type F = *it_src1;
            value_type Ic = *it_src2;

            F |= Ic;

            *it_dst = ((~F) | Ic);


            F = *(it_src1+1);
            Ic = *(it_src2+1);

            F |= Ic;

            *(it_dst+1) = ((~F) | Ic);


            F = *(it_src1+2);
            Ic = *(it_src2+2);

            F |= Ic;

            *(it_dst+2) = ((~F) | Ic);


            F = *(it_src1+3);
            Ic = *(it_src2+3);

            F |= Ic;

            *(it_dst+3) = ((~F) | Ic);
        }

        for(;c<this->_core->_cols;c++,it_src1++,it_src2++,it_dst++)
        {
            value_type F = *it_src1;
            value_type Ic = *it_src2;

            F |= Ic;

            *it_dst = ((~F) | Ic);
        }

    }



}

template<>
void step2<uchar>::operator()(const tbb::blocked_range2d<std::size_t>& ranges)const
{

    const tbb::blocked_range<std::size_t>& rows = ranges.rows();
    const tbb::blocked_range<std::size_t>& tmp = ranges.cols();

    const tbb::blocked_range<std::size_t> cols(tmp.begin() * this->_core->_grain,std::min(tmp.end() * this->_core->_grain,this->_core->_cols));

    const_pointer src1 = this->_core->_src1 + rows.begin() * this->_core->_src1_step1 + cols.begin();
    const_pointer src2 = this->_core->_src2 + rows.begin() * this->_core->_src2_step1 + cols.begin();

    pointer dst = this->_core->_dst + rows.begin() * this->_core->_dst_step1 + cols.begin();

    const std::size_t step = 4;



    const std::size_t stop = cols.size() - (cols.size()%step);

    for(std::size_t r=rows.begin(); r<rows.end(); r++,
        src1+=this->_core->_src1_step1,
        src2+=this->_core->_src2_step1,
        dst+=this->_core->_dst_step1)
    {

        const_pointer it_src1 = src1;
        const_pointer it_src2 = src2;
        pointer it_dst = dst;

        std::size_t c=0;

        for(;c<stop; c+=step,
            it_src1+=step,
            it_src2+=step,
            it_dst+=step)
        {


            value_type F = *it_src1;
            value_type Ic = *it_src2;

            F |= Ic;

            *it_dst = ((~F) | Ic);


            F = *(it_src1+1);
            Ic = *(it_src2+1);

            F |= Ic;

            *(it_dst+1) = ((~F) | Ic);


            F = *(it_src1+2);
            Ic = *(it_src2+2);

            F |= Ic;

            *(it_dst+2) = ((~F) | Ic);


            F = *(it_src1+3);
            Ic = *(it_src2+3);

            F |= Ic;

            *(it_dst+3) = ((~F) | Ic);

        }

        for(;c<cols.size(); c++,
            it_src1++,
            it_src2++,
            it_dst++)
        {
            value_type F = *it_src1;
            value_type Ic = *it_src2;

            F |= Ic;

            *it_dst = ((~F) | Ic);
        }

    }
}


template<>
void step2<schar>::operator()(const tbb::blocked_range<std::size_t>& tmp)const
{

    const tbb::blocked_range<std::size_t> range(tmp.begin() * this->_core->_grain,std::min(tmp.end() * this->_core->_grain,this->_core->_cols));

    const_pointer src1 = this->_core->_src1 + range.begin() * this->_core->_src1_step1;
    const_pointer src2 = this->_core->_src2 + range.begin() * this->_core->_src2_step1;
    pointer dst = this->_core->_dst + range.begin() * this->_core->_dst_step1;

    const std::size_t step = 4;

    const std::size_t stop = this->_core->_cols - (this->_core->_cols%step);

    for(std::size_t r=range.begin();r<range.end(); r++,
        src1+=this->_core->_src1_step1,
        src2+=this->_core->_src2_step1,
        dst+=this->_core->_dst_step1)
    {

        const_pointer it_src1 = src1;
        const_pointer it_src2 = src2;
        pointer it_dst = dst;

        std::size_t c=0;

        for(;c<stop;c+=step,it_src1+=step,it_src2+=step,it_dst+=step)
        {

            value_type F = *it_src1;
            value_type Ic = *it_src2;

            F |= Ic;

            *it_dst = ((~F) | Ic);


            F = *(it_src1+1);
            Ic = *(it_src2+1);

            F |= Ic;

            *(it_dst+1) = ((~F) | Ic);


            F = *(it_src1+2);
            Ic = *(it_src2+2);

            F |= Ic;

            *(it_dst+2) = ((~F) | Ic);


            F = *(it_src1+3);
            Ic = *(it_src2+3);

            F |= Ic;

            *(it_dst+3) = ((~F) | Ic);
        }

        for(;c<this->_core->_cols;c++,it_src1++,it_src2++,it_dst++)
        {
            value_type F = *it_src1;
            value_type Ic = *it_src2;

            F |= Ic;

            *it_dst = ((~F) | Ic);
        }

    }



}

template<>
void step2<schar>::operator()(const tbb::blocked_range2d<std::size_t>& ranges)const
{

    const tbb::blocked_range<std::size_t>& rows = ranges.rows();
    const tbb::blocked_range<std::size_t>& tmp = ranges.cols();

    const tbb::blocked_range<std::size_t> cols(tmp.begin() * this->_core->_grain,std::min(tmp.end() * this->_core->_grain,this->_core->_cols));

    const_pointer src1 = this->_core->_src1 + rows.begin() * this->_core->_src1_step1 + cols.begin();
    const_pointer src2 = this->_core->_src2 + rows.begin() * this->_core->_src2_step1 + cols.begin();

    pointer dst = this->_core->_dst + rows.begin() * this->_core->_dst_step1 + cols.begin();

    const std::size_t step = 4;



    const std::size_t stop = cols.size() - (cols.size()%step);

    for(std::size_t r=rows.begin(); r<rows.end(); r++,
        src1+=this->_core->_src1_step1,
        src2+=this->_core->_src2_step1,
        dst+=this->_core->_dst_step1)
    {

        const_pointer it_src1 = src1;
        const_pointer it_src2 = src2;
        pointer it_dst = dst;

        std::size_t c=0;

        for(;c<stop; c+=step,
            it_src1+=step,
            it_src2+=step,
            it_dst+=step)
        {


            value_type F = *it_src1;
            value_type Ic = *it_src2;

            F |= Ic;

            *it_dst = ((~F) | Ic);


            F = *(it_src1+1);
            Ic = *(it_src2+1);

            F |= Ic;

            *(it_dst+1) = ((~F) | Ic);


            F = *(it_src1+2);
            Ic = *(it_src2+2);

            F |= Ic;

            *(it_dst+2) = ((~F) | Ic);


            F = *(it_src1+3);
            Ic = *(it_src2+3);

            F |= Ic;

            *(it_dst+3) = ((~F) | Ic);

        }

        for(;c<cols.size(); c++,
            it_src1++,
            it_src2++,
            it_dst++)
        {
            value_type F = *it_src1;
            value_type Ic = *it_src2;

            F |= Ic;

            *it_dst = ((~F) | Ic);
        }

    }
}

template<>
void step2<ushort>::operator()(const tbb::blocked_range<std::size_t>& tmp)const
{

    const tbb::blocked_range<std::size_t> range(tmp.begin() * this->_core->_grain,std::min(tmp.end() * this->_core->_grain,this->_core->_cols));

    const_pointer src1 = this->_core->_src1 + range.begin() * this->_core->_src1_step1;
    const_pointer src2 = this->_core->_src2 + range.begin() * this->_core->_src2_step1;
    pointer dst = this->_core->_dst + range.begin() * this->_core->_dst_step1;

    const std::size_t step = 4;

    const std::size_t stop = this->_core->_cols - (this->_core->_cols%step);

    for(std::size_t r=range.begin();r<range.end(); r++,
        src1+=this->_core->_src1_step1,
        src2+=this->_core->_src2_step1,
        dst+=this->_core->_dst_step1)
    {

        const_pointer it_src1 = src1;
        const_pointer it_src2 = src2;
        pointer it_dst = dst;

        std::size_t c=0;

        for(;c<stop;c+=step,it_src1+=step,it_src2+=step,it_dst+=step)
        {

            value_type F = *it_src1;
            value_type Ic = *it_src2;

            F |= Ic;

            *it_dst = ((~F) | Ic);


            F = *(it_src1+1);
            Ic = *(it_src2+1);

            F |= Ic;

            *(it_dst+1) = ((~F) | Ic);


            F = *(it_src1+2);
            Ic = *(it_src2+2);

            F |= Ic;

            *(it_dst+2) = ((~F) | Ic);


            F = *(it_src1+3);
            Ic = *(it_src2+3);

            F |= Ic;

            *(it_dst+3) = ((~F) | Ic);
        }

        for(;c<this->_core->_cols;c++,it_src1++,it_src2++,it_dst++)
        {
            value_type F = *it_src1;
            value_type Ic = *it_src2;

            F |= Ic;

            *it_dst = ((~F) | Ic);
        }

    }



}

template<>
void step2<ushort>::operator()(const tbb::blocked_range2d<std::size_t>& ranges)const
{

    const tbb::blocked_range<std::size_t>& rows = ranges.rows();
    const tbb::blocked_range<std::size_t>& tmp = ranges.cols();

    const tbb::blocked_range<std::size_t> cols(tmp.begin() * this->_core->_grain,std::min(tmp.end() * this->_core->_grain,this->_core->_cols));

    const_pointer src1 = this->_core->_src1 + rows.begin() * this->_core->_src1_step1 + cols.begin();
    const_pointer src2 = this->_core->_src2 + rows.begin() * this->_core->_src2_step1 + cols.begin();

    pointer dst = this->_core->_dst + rows.begin() * this->_core->_dst_step1 + cols.begin();

    const std::size_t step = 4;



    const std::size_t stop = cols.size() - (cols.size()%step);

    for(std::size_t r=rows.begin(); r<rows.end(); r++,
        src1+=this->_core->_src1_step1,
        src2+=this->_core->_src2_step1,
        dst+=this->_core->_dst_step1)
    {

        const_pointer it_src1 = src1;
        const_pointer it_src2 = src2;
        pointer it_dst = dst;

        std::size_t c=0;

        for(;c<stop; c+=step,
            it_src1+=step,
            it_src2+=step,
            it_dst+=step)
        {


            value_type F = *it_src1;
            value_type Ic = *it_src2;

            F |= Ic;

            *it_dst = ((~F) | Ic);


            F = *(it_src1+1);
            Ic = *(it_src2+1);

            F |= Ic;

            *(it_dst+1) = ((~F) | Ic);


            F = *(it_src1+2);
            Ic = *(it_src2+2);

            F |= Ic;

            *(it_dst+2) = ((~F) | Ic);


            F = *(it_src1+3);
            Ic = *(it_src2+3);

            F |= Ic;

            *(it_dst+3) = ((~F) | Ic);

        }

        for(;c<cols.size(); c++,
            it_src1++,
            it_src2++,
            it_dst++)
        {
            value_type F = *it_src1;
            value_type Ic = *it_src2;

            F |= Ic;

            *it_dst = ((~F) | Ic);
        }

    }
}

template<>
void step2<int>::operator()(const tbb::blocked_range<std::size_t>& tmp)const
{

    const tbb::blocked_range<std::size_t> range(tmp.begin() * this->_core->_grain,std::min(tmp.end() * this->_core->_grain,this->_core->_cols));

    const_pointer src1 = this->_core->_src1 + range.begin() * this->_core->_src1_step1;
    const_pointer src2 = this->_core->_src2 + range.begin() * this->_core->_src2_step1;
    pointer dst = this->_core->_dst + range.begin() * this->_core->_dst_step1;

    const std::size_t step = 4;

    const std::size_t stop = this->_core->_cols - (this->_core->_cols%step);

    for(std::size_t r=range.begin();r<range.end(); r++,
        src1+=this->_core->_src1_step1,
        src2+=this->_core->_src2_step1,
        dst+=this->_core->_dst_step1)
    {

        const_pointer it_src1 = src1;
        const_pointer it_src2 = src2;
        pointer it_dst = dst;

        std::size_t c=0;

        for(;c<stop;c+=step,it_src1+=step,it_src2+=step,it_dst+=step)
        {

            value_type F = *it_src1;
            value_type Ic = *it_src2;

            F |= Ic;

            *it_dst = ((~F) | Ic);


            F = *(it_src1+1);
            Ic = *(it_src2+1);

            F |= Ic;

            *(it_dst+1) = ((~F) | Ic);


            F = *(it_src1+2);
            Ic = *(it_src2+2);

            F |= Ic;

            *(it_dst+2) = ((~F) | Ic);


            F = *(it_src1+3);
            Ic = *(it_src2+3);

            F |= Ic;

            *(it_dst+3) = ((~F) | Ic);
        }

        for(;c<this->_core->_cols;c++,it_src1++,it_src2++,it_dst++)
        {
            value_type F = *it_src1;
            value_type Ic = *it_src2;

            F |= Ic;

            *it_dst = ((~F) | Ic);
        }

    }



}

template<>
void step2<int>::operator()(const tbb::blocked_range2d<std::size_t>& ranges)const
{

    const tbb::blocked_range<std::size_t>& rows = ranges.rows();
    const tbb::blocked_range<std::size_t>& tmp = ranges.cols();

    const tbb::blocked_range<std::size_t> cols(tmp.begin() * this->_core->_grain,std::min(tmp.end() * this->_core->_grain,this->_core->_cols));

    const_pointer src1 = this->_core->_src1 + rows.begin() * this->_core->_src1_step1 + cols.begin();
    const_pointer src2 = this->_core->_src2 + rows.begin() * this->_core->_src2_step1 + cols.begin();

    pointer dst = this->_core->_dst + rows.begin() * this->_core->_dst_step1 + cols.begin();

    const std::size_t step = 4;



    const std::size_t stop = cols.size() - (cols.size()%step);

    for(std::size_t r=rows.begin(); r<rows.end(); r++,
        src1+=this->_core->_src1_step1,
        src2+=this->_core->_src2_step1,
        dst+=this->_core->_dst_step1)
    {

        const_pointer it_src1 = src1;
        const_pointer it_src2 = src2;
        pointer it_dst = dst;

        std::size_t c=0;

        for(;c<stop; c+=step,
            it_src1+=step,
            it_src2+=step,
            it_dst+=step)
        {


            value_type F = *it_src1;
            value_type Ic = *it_src2;

            F |= Ic;

            *it_dst = ((~F) | Ic);


            F = *(it_src1+1);
            Ic = *(it_src2+1);

            F |= Ic;

            *(it_dst+1) = ((~F) | Ic);


            F = *(it_src1+2);
            Ic = *(it_src2+2);

            F |= Ic;

            *(it_dst+2) = ((~F) | Ic);


            F = *(it_src1+3);
            Ic = *(it_src2+3);

            F |= Ic;

            *(it_dst+3) = ((~F) | Ic);

        }

        for(;c<cols.size(); c++,
            it_src1++,
            it_src2++,
            it_dst++)
        {
            value_type F = *it_src1;
            value_type Ic = *it_src2;

            F |= Ic;

            *it_dst = ((~F) | Ic);
        }

    }
}

template<>
void step2<float>::operator()(const tbb::blocked_range<std::size_t>& tmp)const
{

    const tbb::blocked_range<std::size_t> range(tmp.begin() * this->_core->_grain,std::min(tmp.end() * this->_core->_grain,this->_core->_cols));

    const_pointer src1 = this->_core->_src1 + range.begin() * this->_core->_src1_step1;
    const_pointer src2 = this->_core->_src2 + range.begin() * this->_core->_src2_step1;
    pointer dst = this->_core->_dst + range.begin() * this->_core->_dst_step1;

    const std::size_t step = 4;

    const std::size_t stop = this->_core->_cols - (this->_core->_cols%step);

    for(std::size_t r=range.begin();r<range.end(); r++,
        src1+=this->_core->_src1_step1,
        src2+=this->_core->_src2_step1,
        dst+=this->_core->_dst_step1)
    {

        const_pointer it_src1 = src1;
        const_pointer it_src2 = src2;
        pointer it_dst = dst;

        std::size_t c=0;

        for(;c<stop;c+=step,it_src1+=step,it_src2+=step,it_dst+=step)
        {

            uint F = cv::saturate_cast<uint>(*it_src1);
            uint Ic = cv::saturate_cast<uint>(*it_src2);

            F |= Ic;

            *it_dst = ((~F) | Ic);


            F = cv::saturate_cast<uint>(*(it_src1+1));
            Ic = cv::saturate_cast<uint>(*(it_src2+1));

            F |= Ic;

            *(it_dst+1) = ((~F) | Ic);


            F = cv::saturate_cast<uint>(*(it_src1+2));
            Ic = cv::saturate_cast<uint>(*(it_src2+2));

            F |= Ic;

            *(it_dst+2) = ((~F) | Ic);


            F = cv::saturate_cast<uint>(*(it_src1+3));
            Ic = cv::saturate_cast<uint>(*(it_src2+3));

            F |= Ic;

            *(it_dst+3) = ((~F) | Ic);
        }

        for(;c<this->_core->_cols;c++,it_src1++,it_src2++,it_dst++)
        {
            uint F = cv::saturate_cast<uint>(*it_src1);
            uint Ic = cv::saturate_cast<uint>(*it_src2);

            F |= Ic;

            *it_dst = ((~F) | Ic);
        }

    }



}

template<>
void step2<float>::operator()(const tbb::blocked_range2d<std::size_t>& ranges)const
{

    const tbb::blocked_range<std::size_t>& rows = ranges.rows();
    const tbb::blocked_range<std::size_t>& tmp = ranges.cols();

    const tbb::blocked_range<std::size_t> cols(tmp.begin() * this->_core->_grain,std::min(tmp.end() * this->_core->_grain,this->_core->_cols));

    const_pointer src1 = this->_core->_src1 + rows.begin() * this->_core->_src1_step1 + cols.begin();
    const_pointer src2 = this->_core->_src2 + rows.begin() * this->_core->_src2_step1 + cols.begin();

    pointer dst = this->_core->_dst + rows.begin() * this->_core->_dst_step1 + cols.begin();

    const std::size_t step = 4;



    const std::size_t stop = cols.size() - (cols.size()%step);

    for(std::size_t r=rows.begin(); r<rows.end(); r++,
        src1+=this->_core->_src1_step1,
        src2+=this->_core->_src2_step1,
        dst+=this->_core->_dst_step1)
    {

        const_pointer it_src1 = src1;
        const_pointer it_src2 = src2;
        pointer it_dst = dst;

        std::size_t c=0;

        for(;c<stop; c+=step,
            it_src1+=step,
            it_src2+=step,
            it_dst+=step)
        {


            uint F = cv::saturate_cast<uint>(*it_src1);
            uint Ic = cv::saturate_cast<uint>(*it_src2);

            F |= Ic;

            *it_dst = ((~F) | Ic);


            F = cv::saturate_cast<uint>(*(it_src1+1));
            Ic = cv::saturate_cast<uint>(*(it_src2+1));

            F |= Ic;

            *(it_dst+1) = ((~F) | Ic);


            F = cv::saturate_cast<uint>(*(it_src1+2));
            Ic = cv::saturate_cast<uint>(*(it_src2+2));

            F |= Ic;

            *(it_dst+2) = ((~F) | Ic);


            F = cv::saturate_cast<uint>(*(it_src1+3));
            Ic = cv::saturate_cast<uint>(*(it_src2+3));

            F |= Ic;

            *(it_dst+3) = ((~F) | Ic);

        }

        for(;c<cols.size(); c++,
            it_src1++,
            it_src2++,
            it_dst++)
        {
            uint F = cv::saturate_cast<uint>(*it_src1);
            uint Ic = cv::saturate_cast<uint>(*it_src2);

            F |= Ic;

            *it_dst = ((~F) | Ic);
        }

    }
}



template<>
void step2<double>::operator()(const tbb::blocked_range<std::size_t>& tmp)const
{

    const tbb::blocked_range<std::size_t> range(tmp.begin() * this->_core->_grain,std::min(tmp.end() * this->_core->_grain,this->_core->_cols));

    const_pointer src1 = this->_core->_src1 + range.begin() * this->_core->_src1_step1;
    const_pointer src2 = this->_core->_src2 + range.begin() * this->_core->_src2_step1;
    pointer dst = this->_core->_dst + range.begin() * this->_core->_dst_step1;

    const std::size_t step = 4;

    const std::size_t stop = this->_core->_cols - (this->_core->_cols%step);

    for(std::size_t r=range.begin();r<range.end(); r++,
        src1+=this->_core->_src1_step1,
        src2+=this->_core->_src2_step1,
        dst+=this->_core->_dst_step1)
    {

        const_pointer it_src1 = src1;
        const_pointer it_src2 = src2;
        pointer it_dst = dst;

        std::size_t c=0;

        for(;c<stop;c+=step,it_src1+=step,it_src2+=step,it_dst+=step)
        {

            uint F = cv::saturate_cast<uint>(*it_src1);
            uint Ic = cv::saturate_cast<uint>(*it_src2);

            F |= Ic;

            *it_dst = ((~F) | Ic);


            F = cv::saturate_cast<uint>(*(it_src1+1));
            Ic = cv::saturate_cast<uint>(*(it_src2+1));

            F |= Ic;

            *(it_dst+1) = ((~F) | Ic);


            F = cv::saturate_cast<uint>(*(it_src1+2));
            Ic = cv::saturate_cast<uint>(*(it_src2+2));

            F |= Ic;

            *(it_dst+2) = ((~F) | Ic);


            F = cv::saturate_cast<uint>(*(it_src1+3));
            Ic = cv::saturate_cast<uint>(*(it_src2+3));

            F |= Ic;

            *(it_dst+3) = ((~F) | Ic);
        }

        for(;c<this->_core->_cols;c++,it_src1++,it_src2++,it_dst++)
        {
            uint F = cv::saturate_cast<uint>(*it_src1);
            uint Ic = cv::saturate_cast<uint>(*it_src2);

            F |= Ic;

            *it_dst = ((~F) | Ic);
        }

    }



}

template<>
void step2<double>::operator()(const tbb::blocked_range2d<std::size_t>& ranges)const
{

    const tbb::blocked_range<std::size_t>& rows = ranges.rows();
    const tbb::blocked_range<std::size_t>& tmp = ranges.cols();

    const tbb::blocked_range<std::size_t> cols(tmp.begin() * this->_core->_grain,std::min(tmp.end() * this->_core->_grain,this->_core->_cols));

    const_pointer src1 = this->_core->_src1 + rows.begin() * this->_core->_src1_step1 + cols.begin();
    const_pointer src2 = this->_core->_src2 + rows.begin() * this->_core->_src2_step1 + cols.begin();

    pointer dst = this->_core->_dst + rows.begin() * this->_core->_dst_step1 + cols.begin();

    const std::size_t step = 4;



    const std::size_t stop = cols.size() - (cols.size()%step);

    for(std::size_t r=rows.begin(); r<rows.end(); r++,
        src1+=this->_core->_src1_step1,
        src2+=this->_core->_src2_step1,
        dst+=this->_core->_dst_step1)
    {

        const_pointer it_src1 = src1;
        const_pointer it_src2 = src2;
        pointer it_dst = dst;

        std::size_t c=0;

        for(;c<stop; c+=step,
            it_src1+=step,
            it_src2+=step,
            it_dst+=step)
        {


            uint F = cv::saturate_cast<uint>(*it_src1);
            uint Ic = cv::saturate_cast<uint>(*it_src2);

            F |= Ic;

            *it_dst = ((~F) | Ic);


            F = cv::saturate_cast<uint>(*(it_src1+1));
            Ic = cv::saturate_cast<uint>(*(it_src2+1));

            F |= Ic;

            *(it_dst+1) = ((~F) | Ic);


            F = cv::saturate_cast<uint>(*(it_src1+2));
            Ic = cv::saturate_cast<uint>(*(it_src2+2));

            F |= Ic;

            *(it_dst+2) = ((~F) | Ic);


            F = cv::saturate_cast<uint>(*(it_src1+3));
            Ic = cv::saturate_cast<uint>(*(it_src2+3));

            F |= Ic;

            *(it_dst+3) = ((~F) | Ic);

        }

        for(;c<cols.size(); c++,
            it_src1++,
            it_src2++,
            it_dst++)
        {
            uint F = cv::saturate_cast<uint>(*it_src1);
            uint Ic = cv::saturate_cast<uint>(*it_src2);

            F |= Ic;

            *it_dst = ((~F) | Ic);
        }

    }
}


#endif

#if CV_SSE2


template<class _Ty>
void worker_(const cv::Mat_<_Ty>& _src,cv::Mat_<_Ty>& dst,const cv::Mat1b& struct_elem,const int& ivt)
{
    static const cv::Mat1b se_cross = (cv::Mat1b(3,3)<<0,1,0,1,1,1,0,1,0);

    cv::Mat1b se;

    if(struct_elem.empty())
        se = se_cross;
    else
        se = struct_elem;


//    cv::Mat_<_Ty> I = _src;
//    cv::Mat_<_Ty>& dst = _dst;

#if CV_AVX
    static const std::size_t algn = 32/sizeof(_Ty);
#elif CV_SSE
    static const std::size_t algn = 16/sizeof(_Ty);
#else
    static const std::size_t algn = 1;
#endif

    cv::Mat_<_Ty> I(_src.rows,cv::alignSize(_src.cols,algn),0.);

    if(_src.isContinuous())
        std::memcpy(I.data,_src.data,_src.total()*_src.elemSize1());
    else
    {
        if(_src.total() > 1e6)
            tbb::parallel_for(0,I.rows,
                              [&I,&_src](const std::size_t& i)->void
            {
                std::memcpy(I.ptr(i),_src.ptr(i),_src.cols*_src.elemSize1());
            });
        else
            std::memcpy(I.data,_src.data,_src.cols*sizeof(_Ty));
    }



    cv::Mat_<_Ty> Ic = cv::Mat_<_Ty>::zeros(I.size());
    cv::Mat_<_Ty> F = cv::Mat_<_Ty>::zeros(I.size());
//    cv::Mat_<_Ty> H = cv::Mat1b::zeros(I.size());
    cv::Mat_<_Ty> tmp = cv::Mat_<_Ty>::zeros(F.size());


    std::size_t grain(0);
    std::size_t cols(0);


#if CV_AVX2
    const std::size_t step = 32/sizeof(_Ty);
#elif CV_SSE2
    const std::size_t step = 16/sizeof(_Ty);
#endif

//    cv::bitwise_not(I,Ic);,
    //                         std::swap(Ic,roi_s);

//    cv::bitwise_or(Ic,I,F);


        grain = I.total() > 0x186A0 ? 0x19000 : 0x400;

        cols = std::ceil(static_cast<float>(F.total())/static_cast<float>(grain));

        if((ivt & AS_IS) == AS_IS)
        {
            step1_as_is<_Ty> body(I, Ic, F, grain);

            tbb::parallel_for( tbb::blocked_range<std::size_t>( 0, cols), body);
        }
        else
        {
            step1_swap<_Ty> body(I, Ic, F, grain);

            tbb::parallel_for( tbb::blocked_range<std::size_t>( 0, cols), body);
        }

        cv::dilate(F,F,se);

//        cv::bitwise_or(F,Ic,F);
//        cv::bitwise_not(F,H);
//        cv::bitwise_or(H,Ic,dst);


        step2<_Ty> body(F,Ic,tmp,grain);

        tbb::parallel_for(tbb::blocked_range<std::size_t>(0,cols),body);


        if(tmp.cols != dst.cols)
        {
            if(tmp.total() > 1e6)
                tbb::parallel_for(0,tmp.rows,[&tmp,&dst](const int& i)->void
                {
                   std::memcpy(dst.ptr(i),tmp.ptr(i),dst.cols*sizeof(_Ty));
                });
            else
                std::memcpy(dst.data,tmp.data,dst.total()*sizeof(_Ty));
        }



}

template<class _Ty>
void worker(cv::InputArray& _src,cv::OutputArray& _dst,cv::InputArray& struct_elem,const int& ivt)
{

    cv::Mat_<_Ty> tmp = _dst.getMat();


    worker_<_Ty>(_src.getMat(),tmp,struct_elem.getMat(),ivt);

}



#else

template<class _Ty>
void worker(cv::InputArray& _src,cv::OutputArray& _dst,cv::InputArray& struct_elem,const int& ivt)
{
    static const cv::Mat1b se_cross = (cv::Mat1b(3,3)<<0,1,0,1,1,1,0,1,0);

    cv::Mat1b se;

    if(struct_elem.empty())
        se = se_cross;
    else
        se = struct_elem.getMat();


    cv::Mat_<_Ty> I = _src.getMat();
    cv::Mat_<_Ty> dst = _dst.getMat();

    cv::Mat_<_Ty> Ic = cv::Mat1b::zeros(I.size());
    cv::Mat_<_Ty> F = cv::Mat1b::zeros(dst.size());

    std::size_t grain(0);
    std::size_t cols(0);

        if(I.isContinuous())
        {
            grain = I.total() > 0x186A0 ? 0x186A0 : 0x400;

            cols = std::ceil(static_cast<float>(F.cols)/static_cast<float>(grain));

            if((ivt & AS_IS) == AS_IS)
            {
                step1_as_is<_Ty> body(I, Ic, F, grain);

                tbb::parallel_for( tbb::blocked_range<std::size_t>( 0, cols), body);
            }
            else
            {
                step1_swap<_Ty> body(I, Ic, F, grain);

                tbb::parallel_for( tbb::blocked_range<std::size_t>( 0, cols), body);
            }

        }
        else
        {
            grain = I.cols > 0x400 ? 0x400 : 0x32;

            cols = std::ceil(static_cast<float>(F.cols)/static_cast<float>(grain));

            if((ivt & AS_IS) == AS_IS)
            {
                step1_as_is<_Ty> body(I, Ic, F, grain);

                tbb::parallel_for( tbb::blocked_range2d<std::size_t>( 0, I.rows, 0,cols ), body);
            }
            else
            {
                step1_swap<_Ty> body(I, Ic, F, grain);

                tbb::parallel_for( tbb::blocked_range2d<std::size_t>( 0, I.rows, 0, cols ), body);
            }
        }

        cv::dilate(F,F,se);

        step2<_Ty> body(F,Ic,dst,grain);


        if(dst.isContinuous())
            tbb::parallel_for(tbb::blocked_range<std::size_t>(0,cols),body);
        else
            tbb::parallel_for(tbb::blocked_range2d<std::size_t>(0, dst.rows, 0, cols),body);
}

#endif

}


void morphological_hole_filling(cv::InputArray _src, cv::OutputArray _dst, cv::InputArray struct_elem,const int& ivt)
{
    CV_DbgAssert((_src.type()==_src.depth()) && (!_dst.fixedType() ||  (_dst.empty() || _dst.type() == _dst.depth()) ) && (struct_elem.empty() || struct_elem.type() == CV_8UC1) );

    if(_dst.empty() || (_dst.size() != _src.size()) )
    {
        _dst.createSameSize(_src,_src.depth());
        _dst.setTo(0.);
    }

    typedef void(*function_type)(cv::InputArray&,cv::OutputArray&,cv::InputArray&,const int&);

    static const function_type funcs[] = {worker<uchar>,
                                          worker<schar>,
                                          worker<ushort>,
                                         worker<short>,
                                         worker<int>,
                                         worker<float>,
                                         worker<double>};

    function_type fun = funcs[_src.depth()];

    fun(_src,_dst,struct_elem,ivt);


}


namespace hal
{

void morphological_hole_filling_8u(const cv::Mat1b& _src,const cv::Mat1b& _dst,const cv::Mat1b& struct_elem,const int& ivt)
{
//    worker<uchar>(_src,_)
}
void morphological_hole_filling_8s(const cv::Mat_<schar>& _src,const cv::Mat_<schar>& _dst,const cv::Mat1b& struct_elem,const int& ivt);

void morphological_hole_filling_16u(const cv::Mat1w& _src,const cv::Mat1w& _dst,const cv::Mat1b& struct_elem,const int& ivt);
void morphological_hole_filling_16s(const cv::Mat1s& _src,const cv::Mat1s& _dst,const cv::Mat1b& struct_elem,const int& ivt);

void morphological_hole_filling_32s(const cv::Mat1i& _src,const cv::Mat1i& _dst,const cv::Mat1b& struct_elem,const int& ivt);
void morphological_hole_filling_32f(const cv::Mat1f& _src,const cv::Mat1f& _dst,const cv::Mat1b& struct_elem,const int& ivt);

void morphological_hole_filling_64f(const cv::Mat1d& _src,const cv::Mat1d& _dst,const cv::Mat1b& struct_elem,const int& ivt);

}

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

//       dst=cv::Mat::zeros(_src.size(),_src.type());

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

//             if(contours.size() < 20)
//             {
                 for(std::size_t i=0;i<contours.size();i++)
                 {

                     cv::Rect roi = cv::boundingRect(contours.at(i));

                     if( (roi.area() > area) && (roi.area() < upper_bound))
                     {

//                         std::cout<<"THERE"<<std::endl<<roi.area()<<" "<<area<<" "<<upper_bound<<std::endl;

                         cv::drawContours(tmp,contours,i,color,cv::FILLED,cv::LINE_AA,hierarchy,0);




                         cv::Mat1b roi_s = src(roi);
                         cv::Mat1b roi_d = tmp(roi);

//                         cv::Mat1b Ic;
//                         cv::Mat1b F;
//                         cv::Mat1b H;

//                         support::show("check",roi_s);

//                         roi_s = roi_s > 0;

//                         cv::bitwise_not(roi_s,roi_s);

//                         static const cv::Mat1b se = (cv::Mat1b(3,3)<<0,1,0,1,1,1,0,1,0);


//                         cv::bitwise_not(roi_s,Ic);
////                         std::swap(Ic,roi_s);

//                         cv::bitwise_or(Ic,roi_s,F);

//                         cv::dilate(F,F,se);

//                         cv::bitwise_or(F,Ic,F);
//                         cv::bitwise_not(F,H);
//                         cv::bitwise_or(H,Ic,roi_d);

//                         cv::bitwise_not(roi_s,roi_s);

                         cv::Mat1w roi_ws;
                         roi_s.convertTo(roi_ws,CV_16U);
                         cv::Mat1w roi_wd(roi_d.size(),ushort(0));

                         morphological_hole_filling(roi_ws,roi_wd,cv::noArray(),SWAP_SRC);

                         if(!roi_wd.empty())
                             roi_wd.convertTo(roi_d,CV_8U);


//#if CV_ENABLE_UNROLLED
//                         for(int r=0,c=0;r<roi_s.rows;r++,c=0)
//                         {
//                             const uchar* s = roi_s[r];
//                             uchar* d = roi_d[r];

//                             for(c=0;c<roi_s.cols-4;c+=4)
//                             {
//                                 uchar p1 = *(s+c);
//                                 uchar p2 = *(s+c+1);

//                                 if(p1==0)
//                                     *(d+c) = 0;

//                                 if(p2==0)
//                                     *(d+c+1) = 0;

//                                 p1 = *(s+c+2);
//                                 p2 = *(s+c+3);

//                                 if(p1==0)
//                                     *(d+c+2) = 0;

//                                 if(p2==0)
//                                     *(d+c+3) = 0;
//                             }
//                             for(;c<roi_s.cols;c++)
//                                 if(*(s+c)==0)
//                                     *(d+c) = 0;
//                         }
//#else
//                         for(int r=0,c=0;r<roi_s.rows;r++,c=0)
//                             for(int c=0;c<roi_s.cols;c++)
//                                 if(roi_s(r,c)==0)
//                                     roi_d(r,c) = 0;
//#endif

                     }
                     else
                         cv::drawContours(tmp,contours,i,color,cv::FILLED,cv::LINE_8,hierarchy,0);

                 }

                 dst = tmp;
//             }
//             else
//             {
//                 ctrs_proc body(contours,hierarchy,color,src,area,upper_bound);

//                 cv::parallel_for_(cv::Range(0,contours.size()),body,0x10);

//                 dst = body;
//             }
    }

    dst.copyTo(_dst);

}

}

