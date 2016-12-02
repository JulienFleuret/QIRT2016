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


#include "regions.h"
#include <opencv2/core/utility.hpp>

#include <opencv2/imgproc.hpp>

#include <list>
#include <mutex>
#include <memory>


#include <functional>

#include <iostream>



namespace regions
{

namespace
{

template<class _Oty,class _Ity>
inline _Oty cvt_(const _Ity& v)
{
    return static_cast<_Oty>(v);
}

template<class _Oty,class _Ty,int m>
inline _Oty cvt_(const cv::Vec<_Ty,m>& v)
{
    _Oty ret;

    std::transform(v.val,v.val+m,ret.val,[](const _Ty& l){ return static_cast<typename _Oty::value_type>(l);});

    return ret;
}

// The behaviour use to be close as std::Divides but this class use two templates rather than one.
// Otherwise an overload of std::Divides would have been preferable.

template<class _Lty,class _Rty>
struct Divides
{
#if __cplusplus <= 201103L
 const _Lty operator()(const _Lty& l,const _Rty& r)const
#else
 constexpr _Lty operator()(const _Lty& l,const _Rty& r)const
#endif
 {
     return l/r;
 }
};

template<class _Ty,int m,class _Rty>
struct Divides<cv::Vec<_Ty,m>,_Rty>
{
#if __cplusplus <= 201103L
 const _Rty operator()(const cv::Vec<_Ty,m>& l,const _Rty& r)const
#else
 constexpr cv::Vec<_Ty,m> operator()(const cv::Vec<_Ty,m>& l,const _Rty& r)const
#endif
 {
     cv::Vec<_Ty,m> ret;

     for(int i=0;i<m;i++)
         ret(i) = l(i)/r;

     return ret;
 }
};

// Worker structure.
// Every methods and contructors is call at least one time.
template<class _Ty>
struct cluster_t_
{
    _Ty mean;
    _Ty sum;
    int n;
    std::list<cv::Point> points;


    template<class _Oty>
    inline cluster_t_(const _Oty& I,const int& r,const int& c,const int& nc):
        mean(cvt_<_Ty>(I)),
        sum(I),
        n(nc),
        points(1,cv::Point(c,r))
    {}


    inline cluster_t_(cluster_t_&& obj):
                mean(obj.mean),
                sum(obj.sum),
                n(obj.n),
                points(std::move(obj.points))
    {}

    inline cluster_t_(cluster_t_& left,cluster_t_& right,const int& n):
        mean(0.),
        sum(left.sum+right.sum),
        n(n),
        points(std::move(left.points))
    {
        this->points.merge(right.points,[](const cv::Point& l,const cv::Point& r){return (l.x < r.x) && (l.y < r.y);});
//        this->mean = this->sum / this->points.size();

        Divides<_Ty,std::size_t> div;

        this->mean = div(this->sum,this->points.size());

        left.clear();
        right.clear();
    }

    ~cluster_t_() = default;

    inline cluster_t_& operator=(const cluster_t_& obj)
    {

        if(std::addressof(obj) != this)
        {
            this->mean = obj.mean;
            this->sum = obj.sum;
            this->n = obj.n;
            this->points = obj.points;

        }

        return (*this);
    }

    inline void clear()
    {
        this->sum = 0;
        this->points.clear();
        this->mean = 0;
    }

    template<class _Oty>
    inline void push(const _Oty& I,const int& r,const int& c)
    {
        this->sum += I;
        this->points.push_back(cv::Point(c,r));

        Divides<_Ty,std::size_t> div;

        div(this->mean,this->points.size());
    }

    void merge(cluster_t_& obj)
    {
        this->sum += obj.sum;
        this->points.insert(this->points.end(),obj.points.begin(),obj.points.end());

        Divides<_Ty,std::size_t> div;

        this->mean = div(this->mean,this->points.size());

        obj.clear();
    }

    inline bool exist()const{ return !this->points.empty();}

    inline operator std::vector<cv::Point>()const
    {
        return std::vector<cv::Point>(this->points.begin(),this->points.end());
    }

};

// The use of opencv parallel interface is made for convinience.
// This interface is very usefull because it's the same interface whatever the library on the background (TBB,OMP,pthread,...).
// But unfortunatly the face the container cv::Range only accept 32 bits signed integers is a huge limitation and is not adapted in every situation.
// Another other limitations of opencv's parallel interface are the fact it's only 1D (there is no cv::Range2d).
// A nive to have when opencv is compile with TBB use to be a compatibilty tbb's blocked_range class.
class parallel_map_update_ : public cv::ParallelLoopBody
{
private:

    cv::Mat1i& map;
    typename std::list<cv::Point>::const_iterator pts_begin;
    const int n;

public:

    inline parallel_map_update_(cv::Mat1i& m,typename std::list<cv::Point>::const_iterator begin,const int& id):
        map(m),
        pts_begin(begin),
        n(id)
    {}

    virtual ~parallel_map_update_() = default;

    inline virtual void operator()(const cv::Range& range)const
    {
        typename std::list<cv::Point>::const_iterator it = this->pts_begin;

        std::advance(it,range.start);

        for(int r = range.start;r<range.end;r++,it++)
            this->map(*it) = this->n;
    }

};

// This mean : update img for the cluster obj i.e. set in img every pixel contains in obj to the obj's cluster id.
template<class _Ty>
inline cv::Mat1i& operator<<(cv::Mat1i& img,const cluster_t_<_Ty>& obj)
{
    if(obj.points.size() < 2e5)
        for(typename std::list<cv::Point>::const_iterator it = obj.points.begin();it != obj.points.end();it++)
            img(*it) = obj.n;
    else
    {
        parallel_map_update_ body(img,obj.points.cbegin(),obj.n);

        cv::parallel_for_(cv::Range(0,obj.points.size()),body,0x400);
    }

    return img;
}

template<class _Sty,class _Mty>
inline bool condition_function(const _Sty& src,const _Mty& mean,const cv::Scalar& thresh)
{
    return std::abs(src - mean) <= thresh(0);
}

template<class _Ty,class _Oty,int m>
inline bool condition_function(const cv::Vec<_Ty,m>& src,const cv::Vec<_Oty,m>& mean,const cv::Scalar& thresh)
{
    bool ret(true);

    for(int i=0;i<m;i++)
        ret = ret && (std::abs(src(i) - mean(i)) <= thresh(i));

    return ret;
}


template<class _Ty,class _Oty>
void worker(cv::InputArray& _src,cv::OutputArray& _dst,cv::OutputArray& _stats,cv::OutputArrayOfArrays& _pts,const cv::Scalar& thresh,const int& delta)
{

    typedef cluster_t_<_Oty> cluster_t;

    cv::Mat_<_Ty> src = _src.getMat();
    cv::Mat1i map = cv::Mat1i::zeros(src.size());
    std::vector<cluster_t> clusters;

    //Step1. Let the upper left pixel as the first cluster.
    clusters.push_back(cluster_t(src(0,0),0,0,0));


    int n(0);

    for(int c=1;c<src.cols;c++)
    {
        // Step 2.
        // If |C_j - mean(C_i)| <= threshold then the candidate C_j is merge in the cluster C_i.
        if(condition_function(src(0,c),clusters.at(map(0,c-1)).mean,thresh))
        {
            map(0,c) = map(0,c-1);
            clusters.at(map(0,c)).push(src(0,c),0,c);
        }
        else // Else a new cluster is create.
        {
            map(0,c) = ++n;
            clusters.push_back(cluster_t(src(0,c),0,c,n));
        }

        //Step3. Repeat Step 2 until all the pixels in the first row have been scanned.
    }

    for(int r=1;r<src.rows;r++)
    {

        // Step4. In the next row compare the first pixel with the cluster C_u
        // which is in the upside of it. And determine if it can be merged into
        // the same cluster.

        int c(0);

        if(condition_function(src(r,c),clusters.at(map(r-1,c)).mean,thresh))
        {
            map(r,c) = map(r-1,c);
            clusters.at(map(r,c)).push(src(r,c),r,c);
        }
        else // Else a new cluster is create.
        {

            map(r,c) = ++n;
            clusters.push_back(cluster_t(src(r,c),r,c,n));
        }



        for(c=1;c<src.cols;c++)
        {

//           const bool cdt_upper_row = (std::abs(cvt_<_Oty>(src(r,c)) - clusters.at(map(r-1,c)).mean) <= thresh); // check the upper row.
//           const bool cdt_previous_col = (std::abs(cvt_<_Oty>(src(r,c)) - clusters.at(map(r,c-1)).mean) <= thresh); // check left column

            const bool cdt_upper_row = condition_function(src(r,c),clusters.at(map(r-1,c)).mean,thresh); // check the upper row.
            const bool cdt_previous_col = condition_function(src(r,c),clusters.at(map(r,c-1)).mean,thresh); // check left column

            if( cdt_upper_row && cdt_previous_col && (map(r-1,c) != map(r,c-1)) )
            {

                // (1) merge C_j weather into C_u or C_l
                map(r,c) = map(r-1,c);
                clusters.at(map(r,c)).push(src(r,c),r,c);

                // (2) merge C_u and C_l into C_n.
                // (3) recompute the mean of C_n.

                map(r,c) = ++n;

                clusters.push_back(cluster_t(clusters.at(map(r-1,c)),clusters.at(map(r,c-1)),n));

                map<<clusters.back();

                continue;
            }

            if(cdt_upper_row)
            {
                // merge C_j with C_u and update the mean of C_u.
                map(r,c) = map(r-1,c);
                clusters.at(map(r,c)).push(src(r,c),r,c);

                continue;
            }

            if(cdt_previous_col)
            {
                map(r,c) = map(r,c-1);
                clusters.at(map(r,c)).push(src(r,c),r,c);

                continue;
            }

                // Otherwise create a new cluster.

                map(r,c) = ++n;
                clusters.push_back(cluster_t(src(r,c),r,c,n));
        }

    }




//    //    Step7. Remove small clusters and assign them to the adjacent cluster.


        std::for_each(clusters.begin(),clusters.end(),[&map,&clusters,&delta](cluster_t& obj)
        {
            if(!obj.exist() || (obj.points.size() >= (std::size_t)delta))
                return;

            cv::Point min = *std::min_element(obj.points.begin(),obj.points.end(),[](const cv::Point& l,const cv::Point& r){return (l.x < r.x) && (l.y < r.y);});

            const int x_beg = (min.x - 1) < 0 ? 0 : min.x - 1;
            const int y_beg = (min.y - 1) < 0 ? 0 : min.y - 1;

            const int x_end = (min.x + 1) >= map.cols ? map.cols - 1 : min.x + 1;
            const int y_end = (min.y + 1) >= map.rows ? map.rows + 1 : min.y + 1;

            int idx(-1);

            for(int x = x_beg;x<=x_end;x++)
                for(int y = y_beg;y<=y_end;y++)
                {
                    if((x == min.x) && (y == min.y))
                        continue;

                    if(map(y,x) != obj.n)
                    {
                        idx = map(y,x);
                        break;
                    }
                }

            if(idx < 0)
            {
                bool stop(false);
                for(typename std::list<cv::Point>::const_iterator it = obj.points.begin();it != obj.points.end();it++)
                {
                    const int x_beg = (it->x - 1) < 0 ? 0 : it->x - 1;
                    const int y_beg = (it->y - 1) < 0 ? 0 : it->y - 1;

                    const int x_end = (it->x + 1) >= map.cols ? map.cols - 1 : it->x + 1;
                    const int y_end = (it->y + 1) >= map.rows ? map.rows + 1 : it->y + 1;

                    for(int x = x_beg;x<=x_end;x++)
                        for(int y = y_beg;y<=y_end;y++)
                        {
                            if((x == it->x) && (y == it->y))
                                continue;

                            if(map(y,x) != obj.n)
                            {
                                idx = map(y,x);
                                stop = true;
                                break;
                            }
                        }

                    if(stop)
                        break;
                }
            }


                clusters.at(idx).merge(obj);

                map<<clusters.at(idx);

        });


        std::vector<cluster_t> final_clusters;

        final_clusters.reserve(clusters.size());

        for(std::size_t i(0);i<clusters.size();i++)
        {
            if(clusters.at(i).exist())
            {
                final_clusters.push_back(std::move(clusters.at(i)));
                final_clusters.back().n = final_clusters.size();
                map<<final_clusters.back();
            }
        }

    clusters = std::move(final_clusters);

    if(_dst.needed())
        map.copyTo(_dst);

    if(_stats.needed())
    {
        if(!_stats.fixedType())
        {
            _stats.create(clusters.size(),5,CV_32SC1,-1,true);


            for(std::size_t i=0;i<clusters.size();i++)
            {
                cv::Mat1i tmp = _stats.getMat(i);
                cluster_t& clus = clusters.at(i);

                cv::Rect roi = cv::boundingRect((std::vector<cv::Point>)clus);

                tmp(0) = roi.x;
                tmp(1) = roi.y;
                tmp(2) = roi.width;
                tmp(3) = roi.height;
                tmp(4) = static_cast<int>(clusters.at(i).points.size());
            }
        }
        else
        {
            cv::Mat1i stats = cv::Mat1i::zeros(clusters.size(),5);

            for(std::size_t i=0;i<clusters.size();i++)
            {
                cv::Mat1i tmp = stats.row(i);
                cluster_t& clus = clusters.at(i);

                cv::Rect roi = cv::boundingRect((std::vector<cv::Point>)clus);

                tmp(0) = roi.x;
                tmp(1) = roi.y;
                tmp(2) = roi.width;
                tmp(3) = roi.height;
                tmp(4) = static_cast<int>(clusters.at(i).points.size());
            }

            stats.copyTo(_stats);
        }
    }

    if(_pts.needed())
    {
        _pts.create(clusters.size(),1,0,-1,true);

        std::size_t i(0);

        for(typename std::vector<cluster_t>::const_iterator it = clusters.begin();it != clusters.end();it++,i++)
        {
            _pts.create(it->points.size(),1,CV_32SC2,i,true);
            cv::Mat2i pts = _pts.getMat(i);

            std::size_t j(0);

            for(typename std::list<cv::Point>::const_iterator it_pts = it->points.begin();it_pts != it->points.end();it_pts++,++j)
                pts(j) = cv::Vec2i(it_pts->x,it_pts->y);
        }
    }
}



}


void fast_scan(cv::InputArray _src, cv::OutputArray _dst,cv::OutputArray _stats,cv::OutputArrayOfArrays _pts, const cv::Scalar &thresh,const int& delta)
{
    CV_DbgAssert(_src.isMat() && (_src.channels()<=4) && ((_dst.needed() && _dst.isMat()) || !_dst.needed()));


    typedef void(*function_type)(cv::InputArray&,cv::OutputArray&,cv::OutputArray&,cv::OutputArrayOfArrays&,const cv::Scalar&,const int&);

    static const function_type funcs[7][4] =
    {
        {worker<uchar,float>,worker<cv::Vec2b,cv::Vec2f>,worker<cv::Vec3b,cv::Vec3f>,worker<cv::Vec4b,cv::Vec4f>},
        {worker<schar,float>,worker<cv::Vec<schar,2>,cv::Vec2f>,worker<cv::Vec<schar,3>,cv::Vec3f>,worker<cv::Vec<schar,4>,cv::Vec4f>},
        {worker<ushort,float>,worker<cv::Vec2w,cv::Vec2f>,worker<cv::Vec3w,cv::Vec3f>,worker<cv::Vec4w,cv::Vec4f>},
        {worker<short,float>,worker<cv::Vec2s,cv::Vec2f>,worker<cv::Vec3s,cv::Vec3f>,worker<cv::Vec4s,cv::Vec4f>},
        {worker<int,double>,worker<cv::Vec2i,cv::Vec2d>,worker<cv::Vec3i,cv::Vec3d>,worker<cv::Vec4i,cv::Vec4d>},
        {worker<float,double>,worker<cv::Vec2f,cv::Vec2d>,worker<cv::Vec3f,cv::Vec3d>,worker<cv::Vec4f,cv::Vec4d>},
        {worker<double,double>,worker<cv::Vec2d,cv::Vec2d>,worker<cv::Vec3d,cv::Vec3d>,worker<cv::Vec4d,cv::Vec4d>}
    };

    function_type fun = funcs[_src.depth()][_src.channels()-1];

    fun(_src,_dst,_stats,_pts,thresh,delta);
}

void fast_scan(cv::InputArray _src, cv::OutputArray _dst, const cv::Scalar &thresh, const int &delta)
{
    fast_scan(_src,_dst,cv::noArray(),cv::noArray(),thresh,delta);
}

}

