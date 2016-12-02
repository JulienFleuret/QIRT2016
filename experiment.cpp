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


#include "experiment.h"


#include <stdexcept>


#include <mutex>
#include <tbb/parallel_for_each.h>

#include <opencv2/core.hpp>

#include <sstream>

#include <iostream>

#include "fs.h"

namespace experiment
{

namespace
{

void get_files_and_folders(const fs::path& input_folder,const std::vector<std::string>& ext,std::vector<fs::path>& files,std::vector<fs::path>& folders)
{

    std::vector<fs::path> files_and_folders;

    exp_fs::content(input_folder,files_and_folders,ext);


    if(!files_and_folders.empty())
    {
        exp_fs::get_files(files_and_folders,files);
        exp_fs::get_folders(files_and_folders,folders);
    }

}

inline std::uintmax_t fun(const std::uintmax_t&cnt,const fs::path& pth)
{
    std::uintmax_t ret = cnt;

    if(fs::is_regular_file(pth))
        ret += fs::file_size(pth);

    if(fs::is_symlink(pth))
    {
        fs::path fl = fs::read_symlink(pth);

        if(!fl.empty())
            ret+= fs::file_size(pth);
    }

    return ret;
}

void check_memory_space(const std::vector<fs::path>& files)
{
    std::uintmax_t size = std::accumulate(files.begin(),files.end(),0,fun);

    fs::space_info s = fs::space(fs::current_path());

    if(s.available < size)
        throw std::runtime_error("Not enough memory availlable for this experiment.");
}


void prepare_outputs(const fs::path& input_folder,std::vector<fs::path>& input_files,std::vector<fs::path>& input_sub_folder,std::vector<fs::path>& output_files,const int& options)
{

    typedef std::lock_guard<std::mutex> lock_t;

    fs::path output_folder;

    output_folder = fs::current_path() / ".." / ("output_" + input_folder.filename().string());

    std::string fld = input_folder.filename().string();

    std::mutex mtx;

    // if the output folder already exist two options exist : either erase everything and recreate it, or create another folder with a slight modification in the foldername.
    if(fs::exists(output_folder))
    {
        if(options & CREATE_FOLDER_NEAR)
        {
            std::size_t cnt(1);

            fs::path tmp = output_folder;

            while(fs::exists(tmp))
            {
                tmp = (output_folder.string() + std::to_string(cnt));

                cnt++;
            }

            output_folder = tmp;
        }
        else
            fs::remove_all(output_folder);
    }

    fs::create_directory(output_folder);

    output_folder +="/";

    std::string fld_o = output_folder.string();

    if(output_files.empty() || output_files.size() != input_files.size())
        output_files.reserve(input_files.size());

    // generate all output filenames
    tbb::parallel_for_each(input_files.begin(),input_files.end(),[&](const fs::path& pth)
    {
        std::string pth_str = pth.string();

        int pos = pth_str.find(fld);

        pos+= fld.size()+1;

        pth_str.erase(0,pos);

        pth_str = fld_o + pth_str;


        std::string stm = pth.stem().string();

        pth_str.replace(pth_str.find(stm)+stm.size(),4,".tiff");


        lock_t lck(mtx);

        output_files.push_back(pth_str);
    });



    // Create all subfolders in the output folder.
    std::for_each(input_sub_folder.begin(),input_sub_folder.end(),[&](const fs::path& pth)
    {

        std::string pth_str = pth.string();

        int pos = pth_str.find(fld);

        pos+= fld.size()+1;

        pth_str.erase(0,pos);

        pth_str = output_folder.string() + pth_str;

        fs::create_directory(pth_str);


    });

}


void prepare_outputs(const fs::path& input_folder,const fs::path& _output_folder,std::vector<fs::path>& input_files,std::vector<fs::path>& input_sub_folder,std::vector<fs::path>& output_files,const int& options)
{

    typedef std::lock_guard<std::mutex> lock_t;

    fs::path output_folder = _output_folder;

    std::string fld = input_folder.filename().string();

    std::mutex mtx;

    // if the output folder already exist two options exist : either erase everything and recreate it, or create another folder with a slight modification in the foldername.
    if(fs::exists(output_folder))
    {
        if(options & CREATE_FOLDER_NEAR)
        {
            std::size_t cnt(1);

            fs::path tmp = output_folder;

            while(fs::exists(tmp))
            {
                tmp = (output_folder.string() + std::to_string(cnt));

                cnt++;
            }

            output_folder = tmp;
        }
        else
            fs::remove_all(output_folder);
    }

    fs::create_directory(output_folder);

    output_folder +="/";

    std::string fld_o = output_folder.string();

    if(output_files.empty() || output_files.size() != input_files.size())
        output_files.reserve(input_files.size());

    // generate all output filenames
    tbb::parallel_for_each(input_files.begin(),input_files.end(),[&](const fs::path& pth)
    {
        std::string pth_str = pth.string();

        int pos = pth_str.find(fld);

        pos+= fld.size()+1;

        pth_str.erase(0,pos);

        pth_str = fld_o + pth_str;


        std::string stm = pth.stem().string();

        pth_str.replace(pth_str.find(stm)+stm.size(),4,".tiff");


        lock_t lck(mtx);

        output_files.push_back(pth_str);
    });



    // Create all subfolders in the output folder.
    std::for_each(input_sub_folder.begin(),input_sub_folder.end(),[&](const fs::path& pth)
    {

        std::string pth_str = pth.string();

        int pos = pth_str.find(fld);

        pos+= fld.size()+1;

        pth_str.erase(0,pos);

        pth_str = output_folder.string() + pth_str;

        fs::create_directory(pth_str);


    });

}

}

void prepration(const fs::path &input_folder,const std::vector<std::string>& ext, std::vector<fs::path> &input_files, std::vector<fs::path> &output_files, const int &options)
{
    DbgAssert(fs::is_directory(input_folder) && !fs::is_empty(input_folder));


    std::vector<fs::path> folders;

    get_files_and_folders(input_folder,ext,input_files,folders);


    check_memory_space(input_files);

    prepare_outputs(input_folder,input_files,folders,output_files,options);

}


void prepration(const fs::path &input_folder, const std::vector<std::string> &ext, const fs::path& output_folder, std::vector<fs::path> &input_files, std::vector<fs::path> &output_files, const int &options)
{
    DbgAssert(fs::is_directory(input_folder) && !fs::is_empty(input_folder));


    std::vector<fs::path> folders;

    get_files_and_folders(input_folder,ext,input_files,folders);


    check_memory_space(input_files);

    prepare_outputs(input_folder,output_folder,input_files,folders,output_files,options);

}


void create_output_dir(const fs::path& src, fs::path& dst, const std::string &ext)
{
    if(dst.empty())
        dst = fs::current_path() / (src.filename().string() + ext);
    else
        dst /= (src.filename().string() + ext);

    fs::create_directory(dst);
}

void create_output_filename(const fs::path &src, fs::path &dst,const std::string& ext_fn)
{
    std::string fn = src.string();
    std::string ext = src.extension().string();

    std::size_t pos = fn.find(ext);

    fn.insert(pos,ext_fn);

    if(!dst.empty())
    {
        std::string input_path = src.parent_path().string();
        std::string output_path = dst.string();

        if(output_path.back() == '/')
            output_path.pop_back();

        pos = fn.find(input_path);

        fn.replace(pos,input_path.size(),output_path);
    }

    dst = fn;
}


void create_output_filenames(const std::vector<fs::path>& src, std::vector<fs::path>& dst, const std::string &ext)
{
    if(dst.empty() || (dst.size() < src.size()))
    {
        dst.reserve(src.size());
        dst.resize(dst.capacity());
    }

    typename std::vector<fs::path>::iterator it_dst = dst.begin();

    for(typename std::vector<fs::path>::const_iterator it_src = src.begin();it_src != src.end();it_src++,it_dst++)
        create_output_filename(*it_src,*it_dst,ext);
}


void split(const cv::String &str, std::vector<std::string> &exts)
{

    typename cv::String::const_pointer beg = str.c_str();
    typename cv::String::const_pointer end = str.c_str() + str.size();

    std::stringstream strstr;

    std::size_t cnt(0);

    for(typename cv::String::const_pointer it = beg; it != end;it++)
    {
        if(*it == ' ')
            strstr<<std::endl;
        else
            strstr<<*it;

        if(*it == '.')
            cnt++;
    }

    if(!exts.empty() || exts.size() != cnt || exts.capacity() != cnt)
    {
        exts.clear();
        exts.reserve(cnt);
    }

    while(!strstr.eof())
    {
        std::string str;

        std::getline(strstr,str);

        exts.push_back(std::move(str));
    }
}

//namespace
//{

//std::vector<std::string> split_string(const std::string& str, const std::string& delimiters)
//{
//    std::vector<std::string> res;

//    std::string split_str = str;
//    size_t pos_delim = split_str.find(delimiters);

//    while ( pos_delim != std::string::npos)
//    {
//        if (pos_delim == 0)
//        {
//            res.push_back("");
//            split_str.erase(0, 1);
//        }
//        else
//        {
//            res.push_back(split_str.substr(0, pos_delim));
//            split_str.erase(0, pos_delim + 1);
//        }

//        pos_delim = split_str.find(delimiters);
//    }

//    res.push_back(split_str);

//    return res;
//}

//std::string del_space(std::string name)
//{
//    while ((name.find_first_of(' ') == 0)  && (name.length() > 0))
//        name.erase(0, 1);

//    while ((name.find_last_of(' ') == (name.length() - 1)) && (name.length() > 0))
//        name.erase(name.end() - 1, name.end());

//    return name;
//}


//bool keyIsNumber(const std::string & option, size_t start)
//{
//    bool isNumber = true;
//    size_t end = option.find_first_of('=', start);
//    end = option.npos == end ? option.length() : end;

//    for ( ; start < end; ++start)
//        if (!std::isdigit(option[start]))
//        {
//            isNumber = false;
//            break;
//        }

//    return isNumber;
//}

//CommandLineParser::CommandLineParser(int argc, const char * const argv[], const cv::String keys):
//    CommandLineParser(argc,argv,keys.c_str())
//{}

//CommandLineParser::CommandLineParser(int argc, const char* const argv[], const char* keys)
//{
//    std::string keys_buffer;
//    std::string values_buffer;
//    std::string buffer;
//    std::string curName;
//    std::vector<string> keysVector;
//    std::vector<string> paramVector;
//    std::map<std::string, std::vector<std::string> >::iterator it;
//    size_t flagPosition;
//    int currentIndex = 1;
//    //bool isFound = false;
//    bool withNoKey = false;
//    bool hasValueThroughEq = false;

//    keys_buffer = keys;
//    while (!keys_buffer.empty())
//    {

//        flagPosition = keys_buffer.find_first_of('}');
//        flagPosition++;
//        buffer = keys_buffer.substr(0, flagPosition);
//        keys_buffer.erase(0, flagPosition);

//        flagPosition = buffer.find('{');
//        if (flagPosition != buffer.npos)
//            buffer.erase(flagPosition, (flagPosition + 1));

//        flagPosition = buffer.find('}');
//        if (flagPosition != buffer.npos)
//            buffer.erase(flagPosition);

//        paramVector = split_string(buffer, "|");
//        while (paramVector.size() < 4) paramVector.push_back("");

//        buffer = paramVector[0];
//        buffer += '|' + paramVector[1];

//        //if (buffer == "") CV_ERROR(CV_StsBadArg, "In CommandLineParser need set short and full name");

//        paramVector.erase(paramVector.begin(), paramVector.begin() + 2);
//        data[buffer] = paramVector;
//    }

//    buffer.clear();
//    keys_buffer.clear();
//    paramVector.clear();
//    for (int i = 1; i < argc; i++)
//    {
//        if (!argv[i])
//            break;
//        curName = argv[i];

//        size_t nondash = curName.find_first_not_of("-");
//        if (nondash == 0 || nondash == curName.npos || keyIsNumber(curName, nondash))
//            withNoKey = true;
//        else
//            curName.erase(0, nondash);

//        if (curName.find('=') != curName.npos)
//        {
//            hasValueThroughEq = true;
//            buffer = curName;
//            curName.erase(curName.find('='));
//            buffer.erase(0, (buffer.find('=') + 1));
//        }

//        values_buffer = del_space(values_buffer);

//        for(it = data.begin(); it != data.end(); it++)
//        {
//            keys_buffer = it->first;
//            keysVector = split_string(keys_buffer, "|");

//            for (size_t j = 0; j < keysVector.size(); j++) keysVector[j] = del_space(keysVector[j]);

//            values_buffer = it->second[0];
//            if (((curName == keysVector[0]) || (curName == keysVector[1])) && hasValueThroughEq)
//            {
//                it->second[0] = buffer;
//                //isFound = true;
//                break;
//            }

//            if (!hasValueThroughEq && ((curName == keysVector[0]) || (curName == keysVector[1]))
//                && (
//                    values_buffer.find("false") != values_buffer.npos ||
//                    values_buffer == ""
//                ))
//            {
//                it->second[0] = "true";
//                //isFound = true;
//                break;
//            }

//            if (!hasValueThroughEq && (values_buffer.find("false") == values_buffer.npos) &&
//                ((curName == keysVector[0]) || (curName == keysVector[1])))
//            {
//                it->second[0] = argv[++i];
//                //isFound = true;
//                break;
//            }


//            if (withNoKey)
//            {
//                std::string noKeyStr = it->first;
//                if(atoi(noKeyStr.c_str()) == currentIndex)
//                {
//                    it->second[0] = curName;
//                    currentIndex++;
//                    //isFound = true;
//                    break;
//                }
//            }
//        }

//        withNoKey = false;
//        hasValueThroughEq = false;
//        //isFound = false;
//    }
//}

//bool CommandLineParser::has(const std::string& keys)
//{
//    std::map<std::string, std::vector<std::string> >::iterator it;
//    std::vector<string> keysVector;

//    for(it = data.begin(); it != data.end(); it++)
//    {
//        keysVector = split_string(it->first, "|");
//        for (size_t i = 0; i < keysVector.size(); i++) keysVector[i] = del_space(keysVector[i]);

//        if (keysVector.size() == 1) keysVector.push_back("");

//        if ((del_space(keys).compare(keysVector[0]) == 0) ||
//            (del_space(keys).compare(keysVector[1]) == 0))
//            return true;
//    }

//    return false;
//}

//std::string CommandLineParser::getString(const std::string& keys)
//{
//    std::map<std::string, std::vector<std::string> >::iterator it;
//    std::vector<string> valueVector;

//    for(it = data.begin(); it != data.end(); it++)
//    {
//        valueVector = split_string(it->first, "|");
//        for (size_t i = 0; i < valueVector.size(); i++) valueVector[i] = del_space(valueVector[i]);

//        if (valueVector.size() == 1) valueVector.push_back("");

//        if ((del_space(keys).compare(valueVector[0]) == 0) ||
//            (del_space(keys).compare(valueVector[1]) == 0))
//            return it->second[0];
//    }
//    return string();
//}

//template<typename _Tp>
// _Tp CommandLineParser::fromStringNumber(const std::string& str)//the default conversion function for numbers
//{
//    return getData<_Tp>(str);
//}

// void CommandLineParser::printParams()
// {
//    int col_p = 30;
//    int col_d = 50;

//    std::map<std::string, std::vector<std::string> >::iterator it;
//    std::vector<std::string> keysVector;
//    std::string buf;
//    for(it = data.begin(); it != data.end(); it++)
//    {
//        keysVector = split_string(it->first, "|");
//        for (size_t i = 0; i < keysVector.size(); i++) keysVector[i] = del_space(keysVector[i]);

//        std::cout << "  ";
//        buf = "";
//        if (keysVector[0] != "")
//        {
//            buf = "-" + keysVector[0];
//            if (keysVector[1] != "") buf += ", --" + keysVector[1];
//        }
//        else if (keysVector[1] != "") buf += "--" + keysVector[1];
//        if (del_space(it->second[0]) != "") buf += "=[" + del_space(it->second[0]) + "]";

//        std::cout << setw(col_p-2) << left << buf;

//        if ((int)buf.length() > col_p-2)
//        {
//            std::cout << std::endl << "  ";
//            std::cout << setw(col_p-2) << left << " ";
//        }

//        buf = "";
//        if (del_space(it->second[1]) != "") buf += del_space(it->second[1]);

//        for(;;)
//        {
//            bool tr = ((int)buf.length() > col_d-2) ? true: false;
//            std::string::size_type pos = 0;

//            if (tr)
//            {
//                pos = buf.find_first_of(' ');
//                for(;;)
//                {
//                    if (buf.find_first_of(' ', pos + 1 ) < (std::string::size_type)(col_d-2) &&
//                        buf.find_first_of(' ', pos + 1 ) != std::string::npos)
//                        pos = buf.find_first_of(' ', pos + 1);
//                    else
//                        break;
//                }
//                pos++;
//                std::cout << setw(col_d-2) << left << buf.substr(0, pos) << std::endl;
//            }
//            else
//            {
//                std::cout << setw(col_d-2) << left << buf<< std::endl;
//                break;
//            }

//            buf.erase(0, pos);
//            std::cout << "  ";
//            std::cout << setw(col_p-2) << left << " ";
//        }
//    }
// }

//template<>
//bool CommandLineParser::get<bool>(const std::string& name, bool space_delete)
//{
//    std::string str_buf = getString(name);

//    if (space_delete && str_buf != "")
//    {
//        str_buf = del_space(str_buf);
//    }

//    if (str_buf == "true")
//        return true;

//    return false;
//}
//template<>
//std::string CommandLineParser::analyzeValue<std::string>(const std::string& str, bool space_delete)
//{
//    if (space_delete)
//    {
//        return del_space(str);
//    }
//    return str;
//}

//template<>
//int CommandLineParser::analyzeValue<int>(const std::string& str, bool /*space_delete*/)
//{
//    return fromStringNumber<int>(str);
//}

//template<>
//unsigned int CommandLineParser::analyzeValue<unsigned int>(const std::string& str, bool /*space_delete*/)
//{
//    return fromStringNumber<unsigned int>(str);
//}

//template<>
//uint64 CommandLineParser::analyzeValue<uint64>(const std::string& str, bool /*space_delete*/)
//{
//    return fromStringNumber<uint64>(str);
//}

//template<>
//float CommandLineParser::analyzeValue<float>(const std::string& str, bool /*space_delete*/)
//{
//    return fromStringNumber<float>(str);
//}

//template<>
//double CommandLineParser::analyzeValue<double>(const std::string& str, bool /*space_delete*/)
//{
//    return fromStringNumber<double>(str);
//}

//}//namespace




}
