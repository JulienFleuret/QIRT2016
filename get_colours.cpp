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

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <set>
#include <list>
#include <functional>
#include <memory>
#include <mutex>

#include <iostream>




namespace support
{

namespace
{

template<class _Ty>
class body_colours_
{
private:

    std::list<_Ty> _pixels;


public:

    typedef typename cv::Mat_<_Ty>::iterator iterator;
    typedef typename cv::Mat_<_Ty>::const_iterator const_iterator;


    body_colours_() = default;

    body_colours_(const body_colours_&,tbb::split) {}


    ~body_colours_() = default;

    void join(body_colours_& obj)
    {
        if(this->_pixels.empty())
            this->_pixels.assign(obj._pixels.begin(),obj._pixels.end());
        else
        {
            std::list<_Ty> buf;

            for(typename std::list<_Ty>::const_iterator it_obj = obj._pixels.begin();it_obj != obj._pixels.end();it_obj++)
            {
                bool ok(true);

                for(typename std::list<_Ty>::const_iterator it_self = this->_pixels.begin();it_self != this->_pixels.end();it_self++)
                    if(*it_obj == *it_self)
                    {
                        ok = false;
                        break;
                    }

                if(ok)
                    buf.push_back(*it_obj);
            }

            this->_pixels.insert(this->_pixels.end(),buf.begin(),buf.end());
        }

        obj._pixels.clear();
    }

    void operator()(const tbb::blocked_range<const_iterator>& range)
    {
        for(const_iterator it = range.begin();it != range.end();it++)
        {

            if(this->_pixels.empty())
                this->_pixels.push_back(*it);
            else
            {
                bool ok(true);

                for(typename std::list<_Ty>::const_iterator it_lst = this->_pixels.begin();it_lst != this->_pixels.end();it_lst++)
                    if(*it_lst == *it)
                    {
                        ok = false;
                        break;
                    }

                if(ok)
                    this->_pixels.push_back(*it);
            }
        }

    }

    inline void get(cv::Mat_<_Ty>& rng)
    {
        if(rng.rows != this->_pixels.size())
            rng.create(this->_pixels.size(),1);

        std::copy(this->_pixels.begin(),this->_pixels.end(),rng.begin());

    }

};


template<class _Ty,class _Oty = int>
class body_map_
{
private:

    struct core_t
    {
        cv::Mat_<_Ty> _img;
        cv::Mat_<_Ty> _pixels;
        cv::Mat_<_Oty> _map;

        inline core_t(cv::Mat_<_Ty>& img,cv::Mat_<_Ty>& pixels):
            _img(img),
            _pixels(pixels),
            _map(img.size(),int(0))
        {}

        ~core_t() = default;
    };

    std::shared_ptr<core_t> _core;

    typedef typename cv::Mat_<_Oty>::iterator map_iterator;

public:

    typedef typename cv::Mat_<_Ty>::iterator iterator;
    typedef typename cv::Mat_<_Ty>::const_iterator const_iterator;

    inline body_map_(cv::Mat_<_Ty>& img,cv::Mat_<_Ty>& pixels):
        _core(new core_t(img,pixels))
    {}

    inline body_map_(const body_map_& obj):
        _core(obj._core)
    {}

    ~body_map_() = default;

    void operator()(const tbb::blocked_range<std::size_t>& range)const
    {
        std::shared_ptr<core_t> core = this->_core;

        for(std::size_t idx = range.begin();idx <range.end();idx++)
        {
            _Ty current_pixel = core->_img(idx);

            std::size_t cnt(1);

            for(const_iterator it = core->_pixels.begin();it != core->_pixels.end();it++,cnt++)
                if(current_pixel == *it)
                    core->_map(idx) = cnt;
        }
    }

    void get(cv::Mat_<_Oty>& map)
    {
        map = this->_core->_map;
    }
};

template<class _Ty>
void worker(cv::InputArray& _src,cv::OutputArray& _dst,cv::OutputArray& _map)
{

    cv::Mat_<_Ty> img = _src.getMat();
    cv::Mat_<_Ty> colours;

    body_colours_<_Ty> colours_body;

    tbb::blocked_range<typename cv::Mat_<_Ty>::const_iterator> colours_range(img.begin(),img.end(),BLOCK_SIZE);

    tbb::parallel_reduce(colours_range,colours_body);

    colours_body.get(colours);

    if(_dst.needed())
        colours.copyTo(_dst);

    //

    if(_map.needed())
    {

        if((_map.fixedType() && _map.depth() == CV_32S) || _map.empty() || ((_map.size() != _src.size()) || (_map.depth() != CV_8U) || (_map.depth() != CV_32S) || (_map.depth() != _map.type()) ))
        {
            cv::Mat1i map;

            body_map_<_Ty> map_body(img,colours);

            tbb::parallel_for(tbb::blocked_range<std::size_t>(0,img.total(),BLOCK_SIZE),map_body);

            map_body.get(map);

            map.copyTo(_map);
        }
        else
        {
            cv::Mat1b map;

            body_map_<_Ty,uchar> map_body(img,colours);

            tbb::parallel_for(tbb::blocked_range<std::size_t>(0,img.total(),BLOCK_SIZE),map_body);

            map_body.get(map);

            map.copyTo(_map);
        }
    }

}

}

void getColours(cv::InputArray _src, cv::OutputArray _dst,cv::OutputArray _map)
{
    CV_DbgAssert(_src.isMat() && (_dst.isMat() || !_dst.needed()) );


    if(_src.empty())
        return;

    typedef void (*function_type)(cv::InputArray&,cv::OutputArray&,cv::OutputArray&);

    static const function_type funcs[7][4] =
    {
        {worker<uchar>,worker<cv::Vec2b>,worker<cv::Vec3b>,worker<cv::Vec4b>},
        {worker<schar>,worker<cv::Vec<schar,2>>,worker<cv::Vec<schar,3>>,worker<cv::Vec<schar,4>>},
        {worker<ushort>,worker<cv::Vec2w>,worker<cv::Vec3w>,worker<cv::Vec4w>},
        {worker<short>,worker<cv::Vec2s>,worker<cv::Vec3s>,worker<cv::Vec4s>},
        {worker<int>,worker<cv::Vec2i>,worker<cv::Vec3i>,worker<cv::Vec4i>},
        {worker<float>,worker<cv::Vec2f>,worker<cv::Vec3f>,worker<cv::Vec4f>},
        {worker<double>,worker<cv::Vec2d>,worker<cv::Vec3d>,worker<cv::Vec4d>}
    };

    function_type fun = funcs[_src.depth()][_src.channels()-1];

    fun(_src,_dst,_map);
}

}



