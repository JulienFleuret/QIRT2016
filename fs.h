
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

#ifndef FS
#define FS

#if __GNUC__ >= 5
#include <experimental/filesystem>
#include <cassert>
#else
#include <boost/filesystem.hpp>
#endif

#if __GNUG__ >= 5
#ifdef _DEBUG
#define DbgAssert(expr) assert(expr)
#else
#define DbgAssert(expr)
#endif
#else
#ifdef _DEBUG
#define DbgAssert(expr) BOOST_ASSERT(expr)
#else
#define DbgAssert(expr)
#endif
#endif

#if __GNUG__ >= 5
namespace sef = std::experimental::filesystem;
namespace fs = sef;
#else
namespace bfs = boost::filesystem;
namespace fs = bfs;
#endif


namespace exp_fs
{

// This function will list all the files and folders at a specified path.
// If the parameter "extensions" is set so only the files the extension(s) will be take in consideration.
//
// path : path where are the files and subfolders to list.
//
// files : files contained at the specified path.
//
// extentions : extentions to keep.
//
void content(const fs::path& path,std::vector<fs::path>& files,const std::vector<std::string>& extentions = std::vector<std::string>());

// This function return a list of the folders contained int the input arguments.
//
// in : list of files and folders.
//
// out : list of folders.
//
void get_files(const std::vector<fs::path>& in,std::vector<fs::path>& out);

// This function return a list of the files contained int the input arguments.
//
// in : list of files and folders.
//
// out : list of files.
//
void get_folders(const std::vector<fs::path>& in,std::vector<fs::path>& out);



}




#endif // FS

