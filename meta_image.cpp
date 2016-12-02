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


#include "experiment.h"

namespace experiment
{

struct meta_image::core_t
{
    cv::Mat _grid;
    std::vector<cv::Mat> _blocks;

    cv::Size _sz_blocks;

    core_t(const int& rows,const int& cols,const int& type):
        _grid(),
        _blocks(),
        _sz_blocks(cols,rows)
    {
        this->create_grid(type);
    }

    core_t(const int& g_r,const int& g_c,const int& cols,const int& rows,const int& type):
        _grid(),
        _blocks(),
        _sz_blocks(cols,rows)
    {
        this->create_grid(g_r,g_c,cols,rows,type);
    }

    ~core_t() = default;

    void create_grid(const int& type)
    {
        this->_grid.create(cv::alignSize(this->_sz_blocks.height*10,1024),cv::alignSize(this->_sz_blocks.width*10,1024),type);
        this->_grid.setTo(0.);

        this->_blocks.reserve(100);
        this->_blocks.resize(100);

        for(int r=0,i=0;r<10;r++)
            for(int c=0;c<10;c++,i++)
            {
                cv::Point pt(c*this->_sz_blocks.width,r*this->_sz_blocks.height);
                cv::Rect roi(pt,this->_sz_blocks);

                this->_blocks.at(i) = this->_grid(roi);
            }
    }

    void create_grid(const int& r_grid,const int& c_grid,const int& r_block,const int& c_block,const int& type)
    {
        int rg(0);
        int cg(0);


        const int ref_row_grid = cv::alignSize(10*r_block,1024);
        const int ref_col_grid = cv::alignSize(10*c_block,1024);

        rg = r_grid < ref_row_grid ? ref_row_grid : cv::alignSize(r_grid,1024);
        cg = c_grid < ref_col_grid ? ref_col_grid : cv::alignSize(c_grid,1024);

        this->_grid.create(rg,cg,type);
        this->_grid.setTo(0.);

        this->_blocks.reserve(100);
        this->_blocks.resize(100);

        for(int r=0,i=0;r<10;r++)
            for(int c=0;c<10;c++,i++)
            {
                cv::Point pt(c*this->_sz_blocks.width,r*this->_sz_blocks.height);
                cv::Rect roi(pt,this->_sz_blocks);

                this->_blocks.at(i) = this->_grid(roi);
            }
    }
};

meta_image::meta_image(const int &r, const int &c, const int &type):
    _core(new core_t(r,c,type))
{}

meta_image::meta_image(const int &r_grid, const int &c_grid, const int &r_block, const int &c_block, const int &type):
    _core(new core_t(r_grid,c_grid,r_block,c_block,type))
{}



bool meta_image::empty()const
{
    return this->_core->_grid.empty();
}

cv::Size meta_image::size()const
{
    return this->_core->_grid.size();
}

std::size_t meta_image::rows()const
{
    return this->_core->_grid.rows;
}

std::size_t meta_image::cols()const
{
    return this->_core->_grid.cols;
}

std::size_t meta_image::total()const
{
    return this->_core->_grid.total();
}

int meta_image::depth()const
{
    return this->_core->_grid.depth();
}

int meta_image::type()const
{
    return this->_core->_grid.type();
}




typename meta_image::image_iterator meta_image::begin_image()
{
    return this->_core->_blocks.begin();
}

typename meta_image::image_iterator meta_image::end_image()
{
    return this->_core->_blocks.end();
}

typename meta_image::const_image_iterator meta_image::begin_image()const
{
    return this->_core->_blocks.begin();
}

typename meta_image::const_image_iterator meta_image::end_image()const
{
    return this->_core->_blocks.end();
}



cv::Size meta_image::image_size()const
{
    return this->_core->_sz_blocks;
}


cv::Mat& meta_image::at_image(const int& idx)
{
    return this->_core->_blocks.at(idx);
}

const cv::Mat& meta_image::at_image(const int& idx)const
{
    return this->_core->_blocks.at(idx);
}


cv::Mat& meta_image::at_image(const int& r,const int& c)
{
    return this->_core->_blocks.at(r*10+c);
}

const cv::Mat& meta_image::at_image(const int &r, const int &c)const
{
    return this->_core->_blocks.at(r*10+c);
}


meta_image::operator cv::Mat()const
{
    return this->_core->_grid;
}

meta_image::operator const cv::Mat()const
{
    return this->_core->_grid;
}

meta_image::operator cv::InputArray()const
{
    return cv::InputArray(this->_core->_grid);
}

meta_image::operator cv::OutputArray()const
{
    return cv::OutputArray(this->_core->_grid);
}

meta_image::operator cv::InputOutputArray()const
{
    return cv::InputOutputArray(this->_core->_grid);
}

void meta_image::release()
{
    this->_core->_grid.release();
    this->_core->_blocks.clear();
    this->_core->_sz_blocks.height = -1;
}

void meta_image::reset_to_zero()
{
    this->_core->_grid.setTo(0.);
}


void meta_image::create(const std::size_t& r, const std::size_t& c, const int &type)
{
    if((r == this->rows()) && (c== this->cols()) && (type == this->type()))
        return;

    this->_core.reset(new core_t(r,c,type));
}

void meta_image::create_images(const std::size_t& r,const std::size_t& c,const std::size_t& r_img,const std::size_t& c_img,const int& type)
{
    if((r == this->rows()) && (c== this->cols()) && (r_img == static_cast<std::size_t>(this->_core->_sz_blocks.height)) && (c_img == static_cast<std::size_t>(this->_core->_sz_blocks.width)) && (type == this->type()))
        return;

    this->_core.reset(new core_t(r,c,r_img,c_img,type));
}


template<>
uchar& meta_image::at<uchar>(const int& x)
{
    return this->_core->_grid.at<uchar>(x);
}

template<>
const uchar& meta_image::at<uchar>(const int& x)const
{
    return this->_core->_grid.at<uchar>(x);
}


template<>
uchar& meta_image::at<uchar>(const int& r,const int& c)
{
    return this->_core->_grid.at<uchar>(r,c);
}

template<>
const uchar& meta_image::at<uchar>(const int& r,const int& c)const
{
    return this->_core->_grid.at<uchar>(r,c);
}


template<>
uchar* meta_image::ptr<uchar>(const int &idx)
{
    return this->_core->_grid.ptr<uchar>(idx);
}

template<>
const uchar* meta_image::ptr<uchar>(const int &idx) const
{
    return this->_core->_grid.ptr<uchar>(idx);
}


template<>
cv::MatIterator_<uchar> meta_image::begin<uchar>()
{
    return this->_core->_grid.begin<uchar>();
}

template<>
cv::MatIterator_<uchar> meta_image::end<uchar>()
{
    return this->_core->_grid.end<uchar>();
}

template<>
cv::MatConstIterator_<uchar> meta_image::begin<uchar>()const
{
    return this->_core->_grid.begin<uchar>();
}

template<>
cv::MatConstIterator_<uchar> meta_image::end<uchar>()const
{
    return this->_core->_grid.end<uchar>();
}






template<>
schar& meta_image::at<schar>(const int& x)
{
    return this->_core->_grid.at<schar>(x);
}

template<>
const schar& meta_image::at<schar>(const int& x)const
{
    return this->_core->_grid.at<schar>(x);
}


template<>
schar& meta_image::at<schar>(const int& r,const int& c)
{
    return this->_core->_grid.at<schar>(r,c);
}

template<>
const schar& meta_image::at<schar>(const int& r,const int& c)const
{
    return this->_core->_grid.at<schar>(r,c);
}


template<>
schar* meta_image::ptr<schar>(const int &idx)
{
    return this->_core->_grid.ptr<schar>(idx);
}

template<>
const schar* meta_image::ptr<schar>(const int &idx) const
{
    return this->_core->_grid.ptr<schar>(idx);
}


template<>
cv::MatIterator_<schar> meta_image::begin<schar>()
{
    return this->_core->_grid.begin<schar>();
}

template<>
cv::MatIterator_<schar> meta_image::end<schar>()
{
    return this->_core->_grid.end<schar>();
}

template<>
cv::MatConstIterator_<schar> meta_image::begin<schar>()const
{
    return this->_core->_grid.begin<schar>();
}

template<>
cv::MatConstIterator_<schar> meta_image::end<schar>()const
{
    return this->_core->_grid.end<schar>();
}






template<>
ushort& meta_image::at<ushort>(const int& x)
{
    return this->_core->_grid.at<ushort>(x);
}

template<>
const ushort& meta_image::at<ushort>(const int& x)const
{
    return this->_core->_grid.at<ushort>(x);
}


template<>
ushort& meta_image::at<ushort>(const int& r,const int& c)
{
    return this->_core->_grid.at<ushort>(r,c);
}

template<>
const ushort& meta_image::at<ushort>(const int& r,const int& c)const
{
    return this->_core->_grid.at<ushort>(r,c);
}


template<>
ushort* meta_image::ptr<ushort>(const int &idx)
{
    return this->_core->_grid.ptr<ushort>(idx);
}

template<>
const ushort* meta_image::ptr<ushort>(const int &idx) const
{
    return this->_core->_grid.ptr<ushort>(idx);
}


template<>
cv::MatIterator_<ushort> meta_image::begin<ushort>()
{
    return this->_core->_grid.begin<ushort>();
}

template<>
cv::MatIterator_<ushort> meta_image::end<ushort>()
{
    return this->_core->_grid.end<ushort>();
}

template<>
cv::MatConstIterator_<ushort> meta_image::begin<ushort>()const
{
    return this->_core->_grid.begin<ushort>();
}

template<>
cv::MatConstIterator_<ushort> meta_image::end<ushort>()const
{
    return this->_core->_grid.end<ushort>();
}






template<>
short& meta_image::at<short>(const int& x)
{
    return this->_core->_grid.at<short>(x);
}

template<>
const short& meta_image::at<short>(const int& x)const
{
    return this->_core->_grid.at<short>(x);
}


template<>
short& meta_image::at<short>(const int& r,const int& c)
{
    return this->_core->_grid.at<short>(r,c);
}

template<>
const short& meta_image::at<short>(const int& r,const int& c)const
{
    return this->_core->_grid.at<short>(r,c);
}


template<>
short* meta_image::ptr<short>(const int &idx)
{
    return this->_core->_grid.ptr<short>(idx);
}

template<>
const short* meta_image::ptr<short>(const int &idx) const
{
    return this->_core->_grid.ptr<short>(idx);
}


template<>
cv::MatIterator_<short> meta_image::begin<short>()
{
    return this->_core->_grid.begin<short>();
}

template<>
cv::MatIterator_<short> meta_image::end<short>()
{
    return this->_core->_grid.end<short>();
}

template<>
cv::MatConstIterator_<short> meta_image::begin<short>()const
{
    return this->_core->_grid.begin<short>();
}

template<>
cv::MatConstIterator_<short> meta_image::end<short>()const
{
    return this->_core->_grid.end<short>();
}








template<>
int& meta_image::at<int>(const int& x)
{
    return this->_core->_grid.at<int>(x);
}

template<>
const int& meta_image::at<int>(const int& x)const
{
    return this->_core->_grid.at<int>(x);
}


template<>
int& meta_image::at<int>(const int& r,const int& c)
{
    return this->_core->_grid.at<int>(r,c);
}

template<>
const int& meta_image::at<int>(const int& r,const int& c)const
{
    return this->_core->_grid.at<int>(r,c);
}


template<>
int* meta_image::ptr<int>(const int &idx)
{
    return this->_core->_grid.ptr<int>(idx);
}

template<>
const int* meta_image::ptr<int>(const int &idx) const
{
    return this->_core->_grid.ptr<int>(idx);
}


template<>
cv::MatIterator_<int> meta_image::begin<int>()
{
    return this->_core->_grid.begin<int>();
}

template<>
cv::MatIterator_<int> meta_image::end<int>()
{
    return this->_core->_grid.end<int>();
}

template<>
cv::MatConstIterator_<int> meta_image::begin<int>()const
{
    return this->_core->_grid.begin<int>();
}

template<>
cv::MatConstIterator_<int> meta_image::end<int>()const
{
    return this->_core->_grid.end<int>();
}






template<>
float& meta_image::at<float>(const int& x)
{
    return this->_core->_grid.at<float>(x);
}

template<>
const float& meta_image::at<float>(const int& x)const
{
    return this->_core->_grid.at<float>(x);
}


template<>
float& meta_image::at<float>(const int& r,const int& c)
{
    return this->_core->_grid.at<float>(r,c);
}

template<>
const float& meta_image::at<float>(const int& r,const int& c)const
{
    return this->_core->_grid.at<float>(r,c);
}


template<>
float* meta_image::ptr<float>(const int &idx)
{
    return this->_core->_grid.ptr<float>(idx);
}

template<>
const float* meta_image::ptr<float>(const int &idx) const
{
    return this->_core->_grid.ptr<float>(idx);
}


template<>
cv::MatIterator_<float> meta_image::begin<float>()
{
    return this->_core->_grid.begin<float>();
}

template<>
cv::MatIterator_<float> meta_image::end<float>()
{
    return this->_core->_grid.end<float>();
}

template<>
cv::MatConstIterator_<float> meta_image::begin<float>()const
{
    return this->_core->_grid.begin<float>();
}

template<>
cv::MatConstIterator_<float> meta_image::end<float>()const
{
    return this->_core->_grid.end<float>();
}






template<>
double& meta_image::at<double>(const int& x)
{
    return this->_core->_grid.at<double>(x);
}

template<>
const double& meta_image::at<double>(const int& x)const
{
    return this->_core->_grid.at<double>(x);
}


template<>
double& meta_image::at<double>(const int& r,const int& c)
{
    return this->_core->_grid.at<double>(r,c);
}

template<>
const double& meta_image::at<double>(const int& r,const int& c)const
{
    return this->_core->_grid.at<double>(r,c);
}


template<>
double* meta_image::ptr<double>(const int &idx)
{
    return this->_core->_grid.ptr<double>(idx);
}

template<>
const double* meta_image::ptr<double>(const int &idx) const
{
    return this->_core->_grid.ptr<double>(idx);
}


template<>
cv::MatIterator_<double> meta_image::begin<double>()
{
    return this->_core->_grid.begin<double>();
}

template<>
cv::MatIterator_<double> meta_image::end<double>()
{
    return this->_core->_grid.end<double>();
}

template<>
cv::MatConstIterator_<double> meta_image::begin<double>()const
{
    return this->_core->_grid.begin<double>();
}

template<>
cv::MatConstIterator_<double> meta_image::end<double>()const
{
    return this->_core->_grid.end<double>();
}



}
