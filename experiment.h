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



#ifndef EXPERIMENT
#define EXPERIMENT

#include <opencv2/core.hpp>

#if __GNUG__ >= 5
#include <experimental/filesystem>
#else
#include <boost/filesystem.hpp>
#endif

#include <map>
#include <string>
#include <vector>



//namespace cv
//{
//class String;
//}

#if __GNUG__ >= 5
namespace sef = std::experimental::filesystem;
namespace fs = sef;
#else
namespace bfs = boost::filesystem;
namespace fs = bfs;
#endif


namespace experiment
{


enum existing_folder_flags
{
    CREATE_FOLDER_NEAR = 0x2,
    ERASE_FOLDER = 0x4

};

void prepration(const fs::path& input_folder,const std::vector<std::string>& ext,std::vector<fs::path>& input_files,std::vector<fs::path>& output_files,const int & options = CREATE_FOLDER_NEAR);

void prepration(const fs::path& input_folder,const std::vector<std::string>& ext,const fs::path& output_folder,std::vector<fs::path>& input_files,std::vector<fs::path>& output_files,const int & options = CREATE_FOLDER_NEAR);

void create_output_dir(const fs::path& src,fs::path& dst,const std::string& ext = std::string("_out"));

void create_output_filename(const fs::path& src,fs::path& dst,const std::string& ext = std::string("_out"));

void create_output_filenames(const std::vector<fs::path>& src,std::vector<fs::path>& dst,const std::string& ext = std::string("_out"));

void split(const cv::String& str,std::vector<std::string>& exts);


class meta_image
{

private:

    struct core_t;

    std::shared_ptr<core_t> _core;

public:

    typedef typename std::vector<cv::Mat>::iterator image_iterator;
    typedef typename std::vector<cv::Mat>::const_iterator const_image_iterator;

    meta_image() = default;

    meta_image(const int& r,const int& c,const int& type);

    inline meta_image(const cv::Size& sz,const int& type):
        meta_image(sz.height,sz.width,type)
    {}



    meta_image(const int& r_grid,const int& c_grid,const int& r_block,const int& c_block,const int& type);

    inline meta_image(const int& r_grid,const int& c_grid,const cv::Size& sz,const int& type):
        meta_image(r_grid,c_grid,sz.height,sz.width,type)
    {}

    inline meta_image(const cv::Size &sz, const int &r_block,const int& c_block,const int& type):
        meta_image(sz.height,sz.width,r_block,c_block,type)
    {}

    inline meta_image(const cv::Size& sz_grid,const cv::Size& sz_block,const int& type):
        meta_image(sz_grid.height,sz_grid.width,sz_block.height,sz_block.width,type)
    {}


    inline meta_image(const meta_image& obj):
        _core(obj._core)
    {}

    inline meta_image(meta_image&& obj):
        _core(obj._core)
    {}


    ~meta_image() = default;


    inline meta_image& operator=(const meta_image& obj)
    {
        if(std::addressof(obj) != this)
            this->_core = obj._core;

        return (*this);
    }

    inline meta_image& operator =(meta_image&& obj)
    {
        if(std::addressof(obj) != this)
            this->_core = obj._core;

        return (*this);
    }

    bool empty()const;

    cv::Size size()const;

    std::size_t rows()const;
    std::size_t cols()const;

    inline std::size_t width()const{ return this->cols();}
    inline std::size_t height()const{ return this->rows();}

    std::size_t total()const;

    int depth()const;
    int type()const;

    // accessors

    template<class _Ty>
    _Ty& at(const int& x);

    template<class _Ty>
    const _Ty& at(const int& x)const;


    template<class _Ty>
    _Ty& at(const int& r,const int& c);

    template<class _Ty>
    const _Ty& at(const int& r,const int& c)const;


    uchar* ptr(const int& idx = 0);
    const uchar* ptr(const int& idx = 0)const;

    template<class _Ty>
    _Ty* ptr(const int &idx);

    template<class _Ty>
    const _Ty* ptr(const int &idx) const;


    template<class _Ty>
    cv::MatIterator_<_Ty> begin();

    template<class _Ty>
    cv::MatIterator_<_Ty> end();

    template<class _Ty>
    cv::MatConstIterator_<_Ty> begin()const;

    template<class _Ty>
    cv::MatConstIterator_<_Ty> end()const;

    //

    image_iterator begin_image();
    image_iterator end_image();

    const_image_iterator begin_image()const;
    const_image_iterator end_image()const;



    cv::Size image_size()const;


    cv::Mat& at_image(const int& idx);
    const cv::Mat& at_image(const int& idx)const;


    cv::Mat& at_image(const int& r,const int& c);
    const cv::Mat& at_image(const int &r, const int &c)const;


    operator cv::Mat()const;
    operator const cv::Mat()const;

    operator cv::InputArray()const;
    operator cv::OutputArray()const;
    operator cv::InputOutputArray()const;

    void release();
    void reset_to_zero();

    void create(const std::size_t &r, const std::size_t &c, const int &type);

    inline void create(const cv::Size& sz,const int& type)
    {
        this->create(sz.height,sz.width,type);
    }

    void create_images(const std::size_t& r,const std::size_t& c,const std::size_t& r_img,const std::size_t& c_img,const int& type);

    inline void create_images(const std::size_t& r,const std::size_t& c,const cv::Size& sz,const int& type)
    {
        this->create_images(r,c,sz,type);
    }

    inline void create_images(const cv::Size& sz,const std::size_t& r,const std::size_t& c,const int& type)
    {
        this->create_images(sz.height,sz.width,r,c,type);
    }

    inline void create_images(const cv::Size &sz_grid, const cv::Size& sz_images,const int& type)
    {
        this->create_images(sz_grid.height,sz_grid.width,sz_images.height,sz_images.width,type);
    }

};



template<>
uchar& meta_image::at<uchar>(const int& x);

template<>
const uchar& meta_image::at<uchar>(const int& x)const;


template<>
uchar& meta_image::at<uchar>(const int& r,const int& c);

template<>
const uchar& meta_image::at<uchar>(const int& r,const int& c)const;


template<>
uchar* meta_image::ptr<uchar>(const int &idx);

template<>
const uchar* meta_image::ptr<uchar>(const int &idx) const;


template<>
cv::MatIterator_<uchar> meta_image::begin<uchar>();

template<>
cv::MatIterator_<uchar> meta_image::end<uchar>();

template<>
cv::MatConstIterator_<uchar> meta_image::begin<uchar>()const;

template<>
cv::MatConstIterator_<uchar> meta_image::end<uchar>()const;






template<>
schar& meta_image::at<schar>(const int& x);

template<>
const schar& meta_image::at<schar>(const int& x)const;


template<>
schar& meta_image::at<schar>(const int& r,const int& c);

template<>
const schar& meta_image::at<schar>(const int& r,const int& c)const;


template<>
schar* meta_image::ptr<schar>(const int &idx);

template<>
const schar* meta_image::ptr<schar>(const int &idx) const;


template<>
cv::MatIterator_<schar> meta_image::begin<schar>();

template<>
cv::MatIterator_<schar> meta_image::end<schar>();

template<>
cv::MatConstIterator_<schar> meta_image::begin<schar>()const;

template<>
cv::MatConstIterator_<schar> meta_image::end<schar>()const;






template<>
ushort& meta_image::at<ushort>(const int& x);

template<>
const ushort& meta_image::at<ushort>(const int& x)const;


template<>
ushort& meta_image::at<ushort>(const int& r,const int& c);

template<>
const ushort& meta_image::at<ushort>(const int& r,const int& c)const;


template<>
ushort* meta_image::ptr<ushort>(const int &idx);

template<>
const ushort* meta_image::ptr<ushort>(const int &idx) const;


template<>
cv::MatIterator_<ushort> meta_image::begin<ushort>();

template<>
cv::MatIterator_<ushort> meta_image::end<ushort>();

template<>
cv::MatConstIterator_<ushort> meta_image::begin<ushort>()const;

template<>
cv::MatConstIterator_<ushort> meta_image::end<ushort>()const;






template<>
short& meta_image::at<short>(const int& x);

template<>
const short& meta_image::at<short>(const int& x)const;


template<>
short& meta_image::at<short>(const int& r,const int& c);

template<>
const short& meta_image::at<short>(const int& r,const int& c)const;


template<>
short* meta_image::ptr<short>(const int &idx);

template<>
const short* meta_image::ptr<short>(const int &idx) const;


template<>
cv::MatIterator_<short> meta_image::begin<short>();

template<>
cv::MatIterator_<short> meta_image::end<short>();

template<>
cv::MatConstIterator_<short> meta_image::begin<short>()const;

template<>
cv::MatConstIterator_<short> meta_image::end<short>()const;








template<>
int& meta_image::at<int>(const int& x);

template<>
const int& meta_image::at<int>(const int& x)const;


template<>
int& meta_image::at<int>(const int& r,const int& c);

template<>
const int& meta_image::at<int>(const int& r,const int& c)const;


template<>
int* meta_image::ptr<int>(const int &idx);

template<>
const int* meta_image::ptr<int>(const int &idx) const;


template<>
cv::MatIterator_<int> meta_image::begin<int>();

template<>
cv::MatIterator_<int> meta_image::end<int>();

template<>
cv::MatConstIterator_<int> meta_image::begin<int>()const;

template<>
cv::MatConstIterator_<int> meta_image::end<int>()const;






template<>
float& meta_image::at<float>(const int& x);

template<>
const float& meta_image::at<float>(const int& x)const;


template<>
float& meta_image::at<float>(const int& r,const int& c);

template<>
const float& meta_image::at<float>(const int& r,const int& c)const;


template<>
float* meta_image::ptr<float>(const int &idx);

template<>
const float* meta_image::ptr<float>(const int &idx) const;


template<>
cv::MatIterator_<float> meta_image::begin<float>();

template<>
cv::MatIterator_<float> meta_image::end<float>();

template<>
cv::MatConstIterator_<float> meta_image::begin<float>()const;

template<>
cv::MatConstIterator_<float> meta_image::end<float>()const;






template<>
double& meta_image::at<double>(const int& x);

template<>
const double& meta_image::at<double>(const int& x)const;


template<>
double& meta_image::at<double>(const int& r,const int& c);

template<>
const double& meta_image::at<double>(const int& r,const int& c)const;


template<>
double* meta_image::ptr<double>(const int &idx);

template<>
const double* meta_image::ptr<double>(const int &idx) const;


template<>
cv::MatIterator_<double> meta_image::begin<double>();

template<>
cv::MatIterator_<double> meta_image::end<double>();

template<>
cv::MatConstIterator_<double> meta_image::begin<double>()const;

template<>
cv::MatConstIterator_<double> meta_image::end<double>()const;


}

#endif // EXPERIMENT

