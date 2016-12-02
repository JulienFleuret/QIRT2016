
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

void content(const fs::path& path,std::vector<fs::path>& files,const std::vector<std::string>& extentions = std::vector<std::string>());

void get_files(const std::vector<fs::path>& in,std::vector<fs::path>& out);
void get_folders(const std::vector<fs::path>& in,std::vector<fs::path>& out);



}




#endif // FS

