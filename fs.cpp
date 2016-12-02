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

#include "fs.h"

#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>

#include <mutex>

#include <list>

namespace exp_fs
{

namespace
{

void content_(const fs::path& path,std::list<fs::path>& files)
{
    std::list<fs::path> tmp;

    for(fs::directory_iterator it = fs::directory_iterator(path);it != fs::directory_iterator();it++)
    {
        if(fs::is_regular_file(*it))
            tmp.push_back(*it);

        if(fs::is_symlink(*it))
        {
            fs::path pth = *it;

            fs::path chk = fs::read_symlink(pth);

            if(!chk.empty())
                tmp.push_back(pth);
        }

        if(fs::is_directory(*it))
        {
            tmp.push_back(*it);

            content_(*it,tmp);
        }
    }

    files.insert(files.end(),tmp.begin(),tmp.end());
}

}

void content(const fs::path &path, std::vector<fs::path> &files, const std::vector<std::string> &extentions)
{

    DbgAssert(fs::exists(path) && fs::is_directory(path));

    std::list<fs::path> tmp;

    for(fs::directory_iterator it = fs::directory_iterator(path);it != fs::directory_iterator();it++)
    {

        if(fs::is_directory(*it))
        {

            fs::path pth = *it;

            tmp.push_back(pth);

            content_(pth,tmp);
        }
        else
        {
            if(fs::is_regular_file(*it))
            {
                fs::path pth = *it;

                tmp.push_back(pth);
            }

            //If the symlink is not broken it's added to the files list.
            if(fs::is_symlink(*it))
            {
                fs::path chk = fs::read_symlink(*it);

                if(!chk.empty())
                {
                    fs::path pth = *it;

                    tmp.push_back(pth);
                }
            }
        }

    }

    if(!extentions.empty())
    {
        std::list<fs::path> tmp2;

        for(typename std::list<fs::path>::iterator it = tmp.begin();it != tmp.end();it++)
        {
            if(fs::is_directory(*it))

                tmp2.push_back(*it);
            else
            {

                std::string fn = it->string();

                std::size_t cnt(0);

                for(typename std::vector<std::string>::const_iterator it = extentions.begin();it != extentions.end();it++)
                {
                    if(fn.find(*it) != std::string::npos)
                        cnt++;
                }

                if(cnt != 0)
                    tmp2.push_back(fn);
            }
        }

        tmp = std::move(tmp2);
    }


    if(!tmp.empty())
        files.assign(tmp.begin(),tmp.end());

}

void get_files(const std::vector<fs::path> &in, std::vector<fs::path> &out)
{

    typedef std::lock_guard<std::mutex> lock_t;

      std::list<fs::path> tmp;

      std::mutex mtx;

      tbb::parallel_for_each(in.begin(),in.end(),
                             [&](const fs::path& pth)
      {
        if(fs::is_regular_file(pth))
        {
            lock_t lck(mtx);
            tmp.push_back(pth);
        }

        if(fs::is_symlink(pth))
        {
            fs::path chk = fs::read_symlink(pth);

            if(!chk.empty())
            {
                lock_t lck(mtx);
                tmp.push_back(pth);
            }
        }

      });

      if(!tmp.empty())
          out.assign(tmp.begin(),tmp.end());
}


void get_folders(const std::vector<fs::path> &in, std::vector<fs::path> &out)
{

    typedef std::lock_guard<std::mutex> lock_t;

      std::list<fs::path> tmp;

      std::mutex mtx;

      tbb::parallel_for_each(in.begin(),in.end(),
                             [&](const fs::path& pth)
      {
        if(fs::is_directory(pth))
        {
            lock_t lck(mtx);
            tmp.push_back(pth);
        }


      });

      if(!tmp.empty())
          out.assign(tmp.begin(),tmp.end());
}

}

