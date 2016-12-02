
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


#ifndef REGIONS_H
#define REGIONS_H

#include <opencv2/core.hpp>

//@article{ding2009efficient,
//  title={An efficient image segmentation technique by fast scanning and adaptive merging},
//  author={Ding, Jian-Jiun and Kuo, CJ and Hong, WC},
//  journal={CVGIP, Aug},
//  year={2009}
//}

namespace regions
{
void fast_scan(cv::InputArray _src, cv::OutputArray _dst,cv::OutputArray _stats,cv::OutputArrayOfArrays _pts, const cv::Scalar &thresh = cv::Scalar::all(25.), const int &delta = 100);
void fast_scan(cv::InputArray _src,cv::OutputArray _dst,const cv::Scalar& thresh = cv::Scalar::all(25.),const int& delta = 100);
}


#endif // REGIONS_H
