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


#ifndef SEGMENTATION
#define SEGMENTATION

#include <opencv2/core.hpp>

namespace segmentation
{
// Real time.
cv::Vec2d get_structure_of_interest(cv::InputArray _src, cv::OutputArray _background, cv::OutputArray _foreground,cv::OutputArray _fp, cv::OutputArray _contours,cv::OutputArray _map);

// Not real time at all.
cv::Vec2d get_structure_of_interest_refined(cv::InputArray _src, cv::OutputArray _background, cv::OutputArray _foreground,cv::OutputArray _fp, cv::OutputArray _contours,cv::OutputArray _map);

void smooth(cv::InputArray _src,cv::InputArray _map,cv::OutputArray _foreground,cv::OutputArray background,cv::OutputArray false_positive,cv::OutputArray new_map);

}

#endif // SEGMENTATION

