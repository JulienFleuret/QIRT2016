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

#include "regions.h"

namespace segmentation
{


namespace
{

template<class _Ty>
class map_8u_ : public cv::ParallelLoopBody
{
private:
    cv::Mat_<_Ty> _src;
    std::vector<std::vector<cv::Point> >& _points;
    cv::Mat3b _map;
    cv::Mat3b _nmap;
    std::vector<cv::Mat> _cns;

public:

    typedef _Ty value_type;

    inline map_8u_(cv::Mat& src,std::vector<std::vector<cv::Point> >& points,cv::Mat3b& map,cv::Mat3b& nmap,std::vector<cv::Mat>& cns):
        _src(src),
        _points(points),
        _map(map),
        _nmap(nmap),
        _cns(cns)
    {}

    ~map_8u_() = default;

    void operator()(const cv::Range& range)const
    {
        for(int r=range.start;r<range.end;r++)
        {
            std::vector<cv::Point> pts = std::move(this->_points.at(r));

            float s(0.f);
            int p(2);

            // Process the fake mean value in order to determine weather the regions associated to the points is a part of the forground, background, or false_positive.
            for(typename std::vector<cv::Point>::const_iterator it = pts.begin();it != pts.end();it++)
            {
                if(this->_map(*it) == cv::Vec3b(255,0,0))
                {
                    s+=1;
                    continue;
                }

                if(this->_map(*it) == cv::Vec3b(0,255,0))
                {
                    s+=2;
                    continue;
                }

                s+=3;
            }

            s/=pts.size();

            p = std::round(s);

            cv::Mat_<_Ty> cn = this->_cns.at(p-1);
            cv::Mat3b nmap = this->_nmap;

            cv::Vec3b pix = p == 3 ? cv::Vec3b(0,0,255) : p == 2 ? cv::Vec3b(0,255,0) : cv::Vec3b(255,0,0);
            cv::Vec3b black(0,0,0);


            for(typename std::vector<cv::Point>::const_iterator it = pts.begin();it != pts.end();it++)
            {
                cn(*it) = this->_src(*it);
                if(nmap(*it) == black)
                    nmap(*it) = pix;
            }
        }
    }
};

template<class _Ty>
void worker_map_8u(cv::Mat& src,cv::Mat& fd,cv::Mat& bd,cv::Mat& fp,cv::Mat& map,cv::Mat& nmap,std::vector<std::vector<cv::Point> >& pts)
{
    std::vector<cv::Mat> tmp = {fd,bd,fp};

    cv::Mat3b tmp2 = map;
    cv::Mat3b tmp3 = nmap;


    map_8u_<_Ty> body(src,pts,tmp2,tmp3,tmp);

    cv::parallel_for_(cv::Range(0,pts.size()),body,0x20);
}

void map_8u(cv::Mat src,cv::Mat& fd,cv::Mat& bd,cv::Mat& fp,cv::Mat& map,cv::Mat& nmap,std::vector<std::vector<cv::Point> >& pts)
{
    typedef void(*function_type)(cv::Mat&,cv::Mat&,cv::Mat&,cv::Mat&,cv::Mat&,cv::Mat&,std::vector<std::vector<cv::Point> >&);

    static function_type funcs[] = {worker_map_8u<uchar>,
                                    worker_map_8u<schar>,
                                    worker_map_8u<ushort>,
                                    worker_map_8u<short>,
                                    worker_map_8u<int>};

    function_type fun = funcs[src.depth()];

    fun(src,fd,bd,fp, map, nmap,pts);
}

// If the input is an matrix of integers. The range of such map must be between 1 (background) and 3 (false_positive)

template<class _Ty>
class map_32s_ : public cv::ParallelLoopBody
{
private:
    cv::Mat_<_Ty> _src;
    std::vector<std::vector<cv::Point> >& _points;
    cv::Mat1i _map;
    cv::Mat3b _nmap;
    std::vector<cv::Mat> _cns;

public:

    typedef _Ty value_type;

    inline map_32s_(cv::Mat& src,std::vector<std::vector<cv::Point> >& points,cv::Mat1i& map,cv::Mat3b& nmap,std::vector<cv::Mat>& cns):
        _src(src),
        _points(points),
        _map(map),
        _nmap(nmap),
        _cns(cns)
    {}

    ~map_32s_() = default;

    void operator()(const cv::Range& range)const
    {
        for(int r=range.start;r<range.end;r++)
        {
            std::vector<cv::Point> pts = std::move(this->_points.at(r));

            float s(0.f);
            int p(2);

            // Process the fake mean value in order to determine weather the regions associated to the points is a part of the forground, background, or false_positive.
            for(typename std::vector<cv::Point>::const_iterator it = pts.begin();it != pts.end();it++)
                s+= this->_map(*it);

            s/=pts.size();

            p = std::round(s);

            cv::Mat_<_Ty> cn = this->_cns.at(p-1);
            cv::Mat3b nmap = this->_nmap;

            cv::Vec3b pix = p == 3 ? cv::Vec3b(0,0,255) : p == 2 ? cv::Vec3b(0,255,0) : cv::Vec3b(255,0,0);
            cv::Vec3b black(0,0,0);


            for(typename std::vector<cv::Point>::const_iterator it = pts.begin();it != pts.end();it++)
            {
                cn(*it) = this->_src(*it);
                if(nmap(*it) == black)
                    nmap(*it) = pix;
            }
        }
    }
};

template<class _Ty>
void worker_map_32s(cv::Mat& src,cv::Mat& fd,cv::Mat& bd,cv::Mat& fp,cv::Mat& map,cv::Mat& nmap,std::vector<std::vector<cv::Point> >& pts)
{
    std::vector<cv::Mat> tmp = {fd,bd,fp};

    cv::Mat1i tmp2 = map;
    cv::Mat3b tmp3 = nmap;


    map_32s_<_Ty> body(src,pts,tmp2,tmp3,tmp);

    cv::parallel_for_(cv::Range(0,pts.size()),body,0x20);
}

void map_32s(cv::Mat src,cv::Mat& fd,cv::Mat& bd,cv::Mat& fp,cv::Mat& map,cv::Mat& nmap,std::vector<std::vector<cv::Point> >& pts)
{
    typedef void(*function_type)(cv::Mat&,cv::Mat&,cv::Mat&,cv::Mat&,cv::Mat&,cv::Mat&,std::vector<std::vector<cv::Point> >&);

    static function_type funcs[] = {worker_map_32s<uchar>,
                                    worker_map_32s<schar>,
                                    worker_map_32s<ushort>,
                                    worker_map_32s<short>,
                                    worker_map_32s<int>};

    function_type fun = funcs[src.depth()];

    fun(src,fd,bd,fp, map, nmap,pts);
}

}

void smooth(cv::InputArray _src, cv::InputArray _map,cv::OutputArray _foreground,cv::OutputArray _background,cv::OutputArray _false_positive,cv::OutputArray new_map)
{
//    std::cout<<"CHECK INSIDE "<<_src.depth()<<" "<<_map.depth()<<" "<<_map.channels()<<""<<std::endl;

    CV_DbgAssert(   (_src.depth() < CV_32F) && (_src.depth() == _src.type()) &&
                   ((_map.type() == CV_32SC1) || (_map.type() == CV_8UC3)) &&
                  (!_foreground.needed() || ( _foreground.empty() || ((_foreground.depth() == _src.depth()) && (_foreground.depth() == _foreground.type()))) ) &&
                  (!_background.needed() || ( _background.empty() || ((_background.depth() == _src.depth()) && (_background.depth() == _background.type()))) )  &&
                  (!_false_positive.needed() || ( (_false_positive.empty() || ((_false_positive.depth() == _src.depth()) && (_false_positive.depth() == _false_positive.type())))) )
                  );

    std::vector<std::vector<cv::Point> > points;



    // Give a smooth region clustering.
    regions::fast_scan(_map,cv::noArray(),cv::noArray(),points);


    cv::Mat nmap(_map.size(),_map.type());
    nmap.setTo(cv::Scalar::all(0.));

    cv::Mat fd = cv::Mat::zeros(_src.size(),_src.type());
    cv::Mat bd = cv::Mat::zeros(_src.size(),_src.type());
    cv::Mat fp = cv::Mat::zeros(_src.size(),_src.type());
    cv::Mat map = _map.getMat();



    // Process the output map image.
    if(_map.depth() == CV_8U)
        map_8u(_src.getMat(),fd,bd,fp,map,nmap,points);
    else
        map_32s(_src.getMat(),fd,bd,fp,map,nmap,points);



    if(_foreground.needed())
        fd.copyTo(_foreground);

    if(_background.needed())
        bd.copyTo(_background);

    if(_false_positive.needed())
        fp.copyTo(_false_positive);

    if(new_map.needed())
        nmap.copyTo(new_map);
}


}
