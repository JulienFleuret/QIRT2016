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
#include <memory>

#include <tbb/parallel_for.h>

#include <iostream>

namespace support
{

namespace
{

template<class _Ty>
class generate_colours_
{
private:

    struct core_t
    {
        cv::Mat_<_Ty>& _pixels;
        cv::Mat3b& _colours;
        cv::RNG _rng;

        inline core_t(cv::Mat_<_Ty>& pixels,cv::Mat3b& colours):
            _pixels(pixels),
            _colours(colours),
            _rng(std::time(nullptr))
        {}

        ~core_t() = default;
    };

    std::shared_ptr<core_t> _core;

public:

    inline generate_colours_(cv::Mat_<_Ty>& pixels,cv::Mat3b& colours):
        _core(new core_t(pixels,colours))
    {}

    inline generate_colours_(const generate_colours_& obj):
        _core(obj._core)
    {}


    ~generate_colours_() = default;

    void operator()(const tbb::blocked_range<int>& range)const
    {

        cv::RNG_MT19937 rng((unsigned)this->_core->_rng);

        std::shared_ptr<core_t> core = this->_core;

        for(int r=range.begin();r<range.end();r++)
        {
            cv::Vec3b pixel(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255));

            core->_colours(r) = pixel;
        }
    }

};


class apply_colours_
{
private:

    struct core_t
    {
        cv::Mat3b& _pixels;
        cv::Mat1i& _map;
        cv::Mat3b& _dst;

        mutable cv::Mutex _mtx;

        inline core_t(cv::Mat3b& pixels,cv::Mat1i& map,cv::Mat3b& dst):
            _pixels(pixels),
            _map(map),
            _dst(dst)
        {}

        ~core_t() = default;
    };

    std::shared_ptr<core_t> _core;

public:

    inline apply_colours_(cv::Mat3b& colours,cv::Mat1i& map,cv::Mat3b& dst):
        _core(new core_t(colours,map,dst))
    {}

    inline apply_colours_(const apply_colours_& obj):
        _core(obj._core)
    {}

    ~apply_colours_() = default;

    inline void operator()(const tbb::blocked_range<std::size_t>& range)const
    {
        std::shared_ptr<core_t> core = this->_core;

        for(std::size_t idx = range.begin();idx < range.end();idx++)
            core->_dst(idx) = core->_pixels(core->_map(idx)-1);

    }

};

template<class _Ty>
void worker(cv::InputArray& _src,cv::OutputArray& _dst)
{
    cv::Mat_<_Ty> src = _src.getMat();

    cv::Mat3b dst(src.size(),cv::Vec3b(0,0,0));

    cv::Mat_<_Ty> pixels;
    cv::Mat1i map;

    getColours(src,pixels,map);

    cv::Mat3b colours(pixels.size(),cv::Vec3b(0,0,0));

    // 1) Generate as many colours as pixels.

    generate_colours_<_Ty> body_generate(pixels,colours);

    tbb::parallel_for(tbb::blocked_range<int>(0,pixels.rows,0x5),body_generate);

    // 2) Apply the colours to the map.

    double min(0.);
    double max(0.);

    cv::minMaxLoc(map,&min,&max);

    apply_colours_ body_apply_colours_(colours,map,dst);

    tbb::parallel_for(tbb::blocked_range<std::size_t>(0,src.total(),0x400),body_apply_colours_);

    dst.copyTo(_dst);
}


}

void apply_random_colours(cv::InputArray _src, cv::OutputArray _dst)
{
    CV_DbgAssert(_src.isMat() && _dst.isMat() && _src.depth() == _src.type());

    if(_src.empty())
    {
        _dst.create(_src.size(),_src.type());
        _dst.setTo(cv::Scalar::all(128));
    }


    typedef void (*function_type)(cv::InputArray&,cv::OutputArray&);

    static const function_type funcs[] = {
                                           worker<uchar>,
                                           worker<schar>,
                                           worker<ushort>,
                                           worker<short>,
                                           worker<int>,
                                           worker<float>,
                                           worker<double>
                                          };

    function_type fun = funcs[_src.depth()];

    fun(_src,_dst);
}

}

