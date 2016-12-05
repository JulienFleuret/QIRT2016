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


#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui.hpp>

#include <iterator>

#include <chrono>

#include "support.h"

#include "fs.h"
#include "experiment.h"

#include "policy.h"

#include "autothresh.h"

#include <fstream>
#include <list>



#include "segmentation.h"
#include "regions.h"

#include <tbb/blocked_range2d.h>
#include <tbb/parallel_reduce.h>

namespace
{
    void write_tags(const cv::Mat3b& map, const cv::String &output_fn,const bool& is_online);
}

int main(int argc,char* argv[])
{


    const cv::String keys =
            "{help h usage ?      |                      |   print this message}"
            "{i input image       |        <none>        |   path to the folder containing the images to analyse}"
            "{o output filename   |        <none>        |   path to folder where to store the analyzed images}"
            "{e extentions        | .tiff .jpg .png .bmp |   supported image format}"
            "{online              |         true         |   execution procedure}"
            ;



    cv::CommandLineParser parser(argc,argv,keys);

    parser.about("A Real Time Animal Detection And Segmentation Algorithm For IRT Images In Indoor Environments");

    if(parser.has("help"))
    {
        parser.printMessage();
        return EXIT_SUCCESS;
    }

    bool is_online = parser.get<bool>("online");

//    is_online = false;

//    std::cout<<"IS_ONLINE "<<is_online<<std::endl;

    cv::String input = parser.get<cv::String>("input") ;    

    //TMP
//    if(input.empty())
//        input = IMAGE_TIFF4;

//    std::cout<<"CHECK INPUT "<<input<<std::endl;

//    // TMP
//    if(input.empty())
//        input = "/home/administrateur/lib_dir/opencv_dir/opencv_311/opencv_extra/testdata/cv/cvtcolor";


    bool is_image(false);

    if(input.empty())
    {
        std::cerr<<"An input directory or image must be set."<<std::endl;
        return EXIT_SUCCESS;
    }
    else
    {
        fs::path ipth = (std::string)input;



        if(!fs::exists(ipth))
        {
            std::cerr<<"The path set as input directory doesn't exist."<<std::endl;
            return EXIT_SUCCESS;
        }

        if(fs::is_regular_file(ipth) || fs::is_symlink(ipth))
        {
            if(fs::is_symlink(ipth))
            {
                fs::path tmp = fs::read_symlink(ipth);

                if(tmp.empty())
                {
                    std::cerr<<"The filename set as input directory refer to a broken simlink."<<std::endl;
                    return EXIT_SUCCESS;
                }

                // Surprizingly if tmp refer to a folder fs::is_regular_file(tmp) is false.
                is_image = fs::is_regular_file(tmp);
//                std::cout<<"CHECK HERE "<<is_image<<" "<<tmp<<" "<<fs::is_regular_file(tmp)<<std::endl;

            }
            else
                is_image = true;
        }
        else
        {
            if(!fs::is_directory(ipth))
            {
                if(fs::is_symlink(ipth))
                {
                    ipth = fs::read_symlink(ipth);

                    if(ipth.empty())
                    {
                        std::cerr<<"The path set as input directory refer to a broken simlink."<<std::endl;
                        return EXIT_SUCCESS;
                    }

                    if(!fs::is_directory(ipth))
                    {
                        std::cerr<<"The path set as input directory refer to a simlink. But this simlink doesn't refer to a directory"<<std::endl;
                        return EXIT_SUCCESS;
                    }
                }

                std::cerr<<"The path set as input directory doesn't refer to a directory."<<std::endl;
                return EXIT_SUCCESS;
            }
        }
    }


    cv::String output = parser.get<cv::String>("output");
    cv::String extentions = parser.get<cv::String>("extentions");



    if(output.empty() && is_image)
        output = parser.getPathToApplication();

    std::vector<std::string> exts;

    experiment::split(extentions,exts);


//    std::cout<<"CHECK "<<is_image<<" "<<input<<" "<<fs::is_regular_file(fs::path((std::string)input))<<" "<<fs::is_symlink(fs::path((std::string)input))<<std::endl;

    if(is_image)
    {

//        fs::path img = (std::string)input

        std::string iext = fs::path((std::string)input).extension().string();

        std::size_t cnt(0);

        for(typename std::vector<std::string>::const_iterator it = exts.begin();it != exts.end();it++)

            if(iext == *it)
                cnt++;

        if(cnt == 0)
        {
            std::cerr<<"The format of the input file doesn't correspond to one supported"<<std::endl;
            return EXIT_SUCCESS;
        }

        fs::path output;
        fs::path bkgd_fn;
        fs::path frgd_fn;
        fs::path fp_fn;
        fs::path ctrs_fn;
        fs::path maps_fn;
//        fs::path ymls_fn;


        if(parser.has("output"))
            output = (std::string)parser.get<cv::String>("output");

        std::string online_name = is_online ? "_online" : "_offline";

        experiment::create_output_filename((std::string)input,output);
        experiment::create_output_filename(output.string(),bkgd_fn,online_name+"_bkgd");
        experiment::create_output_filename(output.string(),frgd_fn,online_name+"_frgd");
        experiment::create_output_filename(output.string(),fp_fn,online_name+"_fp");
        experiment::create_output_filename(output.string(),ctrs_fn,online_name+"_ctrs");
        experiment::create_output_filename(output.string(),maps_fn,online_name+"_map");
//        experiment::create_output_filename(output.string(),ymls_fn,"");


//        std::cout<<"CHECK OUTPUT "<<input<<" "<<output<<" "<<bkgd_fn<<std::endl;

        cv::Mat in = cv::imread(input,cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
        cv::Mat frgd;
        cv::Mat bckgd;
        cv::Mat ctrs;
        cv::Mat fp;
        cv::Mat map;

        cv::Vec2d min_max;

//        std::cout<<"IS ONLINE "<<is_online<<" "<<in.type()<<" "<<(int)('\x80')<<std::endl;

        if(is_online)
        {
            support::timer.start();
            min_max = segmentation::get_structure_of_interest(in,bckgd,frgd,fp,ctrs,map);
            support::timer.stop();
        }
        else //Offline
        {
            support::timer.start();
            min_max = segmentation::get_structure_of_interest_refined(in,bckgd,frgd,fp,ctrs,map);
            support::timer.stop();
        }


        support::timer.print_us(std::cout);

//        cv::Mat3b tmp = map;

//        write_tags(map,output.string(),is_online);

//        cv::imwrite(bkgd_fn.string(),bckgd);
//        cv::imwrite(frgd_fn.string(),frgd);
//        cv::imwrite(fp_fn.string(),fp);
//        cv::imwrite(ctrs_fn.string(),ctrs);
//        cv::imwrite(maps_fn.string(),map);


        support::show("source",in);
        support::show(bkgd_fn.string(),bckgd,min_max);
        support::show(frgd_fn.string(),frgd);
        support::show(fp_fn.string(),fp);
        support::show(ctrs_fn.string(),ctrs);
        support::show(maps_fn.string(),map);

        cv::waitKey(-1);
    }

    if(!is_image)
    {
        fs::path i_img = (std::string)input;
        fs::path o_frgd = (std::string)output;
        fs::path o_bkgd = (std::string)output;
        fs::path o_fp = (std::string)output;
        fs::path o_ctrs = (std::string)output;
        fs::path o_maps = (std::string)output;
        fs::path o_yml = (std::string)output;

        // Create the output directories
        experiment::create_output_dir(i_img,o_frgd,"_frgd");
        experiment::create_output_dir(i_img,o_bkgd,"_bkgd");
        experiment::create_output_dir(i_img,o_ctrs,"_ctrs");
        experiment::create_output_dir(i_img,o_fp,"_fp");
        experiment::create_output_dir(i_img,o_maps,"_maps");
        experiment::create_output_dir(i_img,o_yml,"_yml");



//        std::cout<<std::endl<<"CHECK OUTPUT "<<i_img<<std::endl<<o_frgd<<std::endl<<o_bkgd<<std::endl<<o_ctrs<<std::endl<<std::endl;


        std::vector<fs::path> ifs;
        std::vector<fs::path> bkgd;
        std::vector<fs::path> frgd;
        std::vector<fs::path> fps;
        std::vector<fs::path> ctrs;
        std::vector<fs::path> maps;
        std::vector<fs::path> ymls;

        // Read the content of the input folder.
        exp_fs::content(i_img,ifs,exts);
        // Keep only the files with a valid extension. By default the valid extensions are : .bmp, .pnf, .tiff, .jpg.
        exp_fs::get_files(ifs,ifs);



        bkgd.reserve(ifs.size());
        bkgd.resize(bkgd.capacity(),o_bkgd);

        frgd.reserve(ifs.size());
        frgd.resize(frgd.capacity(),o_frgd);

        fps.reserve(ifs.size());
        fps.resize(fps.capacity(),o_fp);

        ctrs.reserve(ifs.size());
        ctrs.resize(ctrs.capacity(),o_ctrs);

        maps.reserve(ifs.size());
        maps.resize(maps.capacity(),o_maps);

        ymls.reserve(ifs.size());
        ymls.resize(ymls.capacity());

        std::string online_name = is_online ? "_online" : "_offline";

        // Prepare the output filenames.
        experiment::create_output_filenames(ifs,bkgd,online_name+"_bkgd");
        experiment::create_output_filenames(ifs,frgd,online_name+"_frgd");
        experiment::create_output_filenames(ifs,fps,online_name+"_fp");
        experiment::create_output_filenames(ifs,ctrs,online_name+"_ctrs");
        experiment::create_output_filenames(ifs,maps,online_name+"_map");
        experiment::create_output_filenames(ifs,ymls,online_name+"_ymls");

        std::cout<<"is_online "<<is_online<<std::endl;

        std::size_t cnt(0);

        for(std::size_t i=0;i<ifs.size();i++)
        {

            if(fs::is_directory(ifs.at(i)))
                continue;

            cv::Mat in = cv::imread(ifs.at(i).string(),cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
            cv::Mat fd;
            cv::Mat bd;
            cv::Mat ct;
            cv::Mat fp;
            cv::Mat map;

            cv::Vec2d min_max;



            if(is_online)
            {
                support::timer.start();
                min_max = segmentation::get_structure_of_interest(in,bd,fd,fp,ct,map);
                support::timer.stop();
            }
            else //Offline
            {
                support::timer.start();
                min_max = segmentation::get_structure_of_interest_refined(in,bd,fd,fp,ct,map);
                support::timer.stop();
            }

            cnt+=support::timer.diff_us();

            support::timer.print_us(std::cout);


//            write_tags(map,ymls.at(i).string(),is_online);

//            cv::imwrite(bkgd.at(i).string(),bd);
//            cv::imwrite(frgd.at(i).string(),fd);
//            cv::imwrite(fps.at(i).string(),fp);
//            cv::imwrite(ctrs.at(i).string(),ct);
//            cv::imwrite(maps.at(i).string(),map);

        }

        std::cout<<"average processing time : "<<(cnt/float(ifs.size()))<<" us"<<std::endl;

    }



    std::cout << "Hello World!" << std::endl;
    return EXIT_SUCCESS;
}


namespace
{

class wt_
{
private:

    cv::Mat3b _map;
    std::vector<std::list<cv::Point> > _pts;

    typedef std::move_iterator<typename std::list<cv::Point>::iterator> move_iterator;
    typedef std::move_iterator<typename std::list<cv::Point>::const_iterator> const_move_iterator;

public:

    inline wt_(cv::Mat3b& map):
        _map(map)
    {
        this->_pts.reserve(3);
        this->_pts.resize(3);
    }

    inline wt_(const wt_& obj,tbb::split):
        _map(obj._map)
    {
        this->_pts.reserve(3);
        this->_pts.resize(3);
    }

    ~wt_() = default;

    inline void join(wt_& obj)
    {
        for(typename std::vector<std::list<cv::Point> >::iterator it = this->_pts.begin(), it_obj = obj._pts.begin(); it != this->_pts.end();it++,it_obj++)
            it->insert(it->end(),move_iterator(it_obj->begin()),move_iterator(it_obj->end()));
    }

    void operator()(const tbb::blocked_range2d<std::size_t>& range)
    {
        const tbb::blocked_range<std::size_t>& rows = range.rows();
        const tbb::blocked_range<std::size_t>& cols = range.cols();

        for(std::size_t r=rows.begin();r<rows.end();r++)
            for(std::size_t c=cols.begin();c<cols.end();c++)
            {
                if(this->_map(r,c) == cv::Vec3b(255,0,0))
                {
                    this->_pts.at(0).push_back(cv::Point(c,r));
                    continue;
                }

                if(this->_map(r,c) == cv::Vec3b(0,255,0))
                {
                    this->_pts.at(1).push_back(cv::Point(c,r));
                    continue;
                }


                this->_pts.at(2).push_back(cv::Point(c,r));

            }
    }

    void operator()(const tbb::blocked_range<std::size_t>& range)
    {


        for(std::size_t r=range.begin();r<range.end();r++)
            for(std::size_t c=0;c<this->_map.cols;c++)
            {
                if(this->_map(r,c) == cv::Vec3b(255,0,0))
                {
                    this->_pts.at(0).push_back(cv::Point(c,r));
                    continue;
                }

                if(this->_map(r,c) == cv::Vec3b(0,255,0))
                {
                    this->_pts.at(1).push_back(cv::Point(c,r));
                    continue;
                }


                this->_pts.at(2).push_back(cv::Point(c,r));
            }
    }

    operator std::vector<std::vector<cv::Point> >()const
    {
        std::vector<std::vector<cv::Point> > ret;

        ret.reserve(3);

        for(typename std::vector<std::list<cv::Point> >::const_iterator it = this->_pts.begin();it != this->_pts.end();it++)
        {
            std::vector<cv::Point> tmp(const_move_iterator(it->begin()),const_move_iterator(it->end()));

            ret.push_back(std::move(tmp));
        }

        return ret;
    }

};

    void write_tags(const cv::Mat3b& map,const cv::String& output_fn,const bool& is_online)
    {

//        // 1) Find the points in each regions (foreground, background, false_positive).
//        wt_ body(map);

//        if(map.cols > 0x400)
//            tbb::parallel_reduce(tbb::blocked_range2d<std::size_t>(0,map.rows,0x10,0,map.cols,0x400),body);
//        else
//            tbb::parallel_reduce(tbb::blocked_range<std::size_t>(0,map.rows,0x10),body);

//        std::vector<std::vector<cv::Point> > pts = body;

        std::vector<std::vector<cv::Point> > pts;

        regions::fast_scan(map,cv::noArray(),cv::noArray(),pts,cv::Scalar::all(25),1);

        std::vector<cv::String> names(pts.size());
        int ids[3]={0};

        cv::Vec3b ref_pix[3] = {cv::Vec3b(255,0,0),cv::Vec3b(0,255,0),cv::Vec3b(0,0,255)};

        std::size_t idx=0;
        for(typename std::vector<std::vector<cv::Point> >::const_iterator it = pts.begin();it != pts.end();it++,idx++)
        {
            cv::Vec3b pix = map(it->front());

            if(pix == ref_pix[0])
            {
                names.at(idx) = cv::format("foreground_%d",*ids);
                (*ids)++;
            }

            if(pix == ref_pix[1])
            {
                names.at(idx) = cv::format("background_%d",*(ids+1));
                (*(ids+1))++;
            }

            if(pix == ref_pix[2])
            {
                names.at(idx) = cv::format("false_positive_%d",(*(ids+2)));
                (*(ids+2))++;
            }
        }


        // 2) Modify the output filename in order to write the
        std::string yml_fn = output_fn;

        std::string extention = output_fn.find(".bmp") != cv::String::npos ? ".bmp" : output_fn.find(".png") != cv::String::npos ? ".png" : output_fn.find(".jpg") != cv::String::npos ? ".jpg" : ".tiff" ;

        std::string name = is_online ? "_online.yml" : "_offline.yml";

        yml_fn.replace(yml_fn.find(extention),extention.size(),name);


        // 3) Write the datas.
        cv::FileStorage fs(yml_fn,cv::FileStorage::WRITE);

//        std::vector<cv::String> names = {"foreground","background","false_positive"};

        fs<<"nb_regions"<<static_cast<int>(pts.size());
        fs<<"names"<<"[";
        for(std::size_t i=0;i<names.size();i++)
            fs<<"{:"<<cv::format("region_%d",i)<<names.at(i)<<"}";
        fs<<"]";


        std::size_t i=0;

        for(typename std::vector<std::vector<cv::Point> >::const_iterator it = pts.begin();it != pts.end();it++,i++)
            fs<<names.at(i)<<cv::Mat(*it);


    }
}
