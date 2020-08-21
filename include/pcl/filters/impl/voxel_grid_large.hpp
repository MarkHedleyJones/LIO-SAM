/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2009, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id$
 *
 */

#ifndef PCL_FILTERS_IMPL_VOXEL_GRID_LARGE_H_
#define PCL_FILTERS_IMPL_VOXEL_GRID_LARGE_H_

#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/common/io.h>
#include <pcl/filters/voxel_grid_large.h>

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::VoxelGridLarge<PointT>::applyFilter (PointCloud &output)
{
  // Has the input dataset been set already?
  if (!input_)
  {
    PCL_WARN ("[pcl::%s::applyFilter] No input dataset given!\n", getClassName ().c_str ());
    output.width = output.height = 0;
    output.points.clear ();
    return;
  }

  // Copy the header (and thus the frame_id) + allocate enough space for points
  output.height       = 1;                    // downsampling breaks the organized structure
  output.is_dense     = true;                 // we filter out invalid points

  Eigen::Vector4f min_p, max_p;
  // Get the minimum and maximum dimensions
  getMinMax3D<PointT> (*input_, *indices_, min_p, max_p);

  // Check that the leaf size is not too small, given the size of the data
  int64_t dx = static_cast<int64_t>((max_p[0] - min_p[0]) * inverse_leaf_size_[0])+1;
  int64_t dy = static_cast<int64_t>((max_p[1] - min_p[1]) * inverse_leaf_size_[1])+1;
  int64_t dz = static_cast<int64_t>((max_p[2] - min_p[2]) * inverse_leaf_size_[2])+1;

  if ((dx > static_cast<int64_t>(std::numeric_limits<int64_t>::max()) / dy) ||
      ((dz > static_cast<int64_t>(std::numeric_limits<int64_t>::max()) / (dx * dy))))
  {
    PCL_WARN("[pcl::%s::applyFilter] Leaf size is too small for the input dataset. Integer indices would overflow.", getClassName().c_str());
    output = *input_;
    return;
  }

  // Compute the minimum and maximum bounding box values
  min_b_[0] = static_cast<int> (floor (min_p[0] * inverse_leaf_size_[0]));
  max_b_[0] = static_cast<int> (floor (max_p[0] * inverse_leaf_size_[0]));
  min_b_[1] = static_cast<int> (floor (min_p[1] * inverse_leaf_size_[1]));
  max_b_[1] = static_cast<int> (floor (max_p[1] * inverse_leaf_size_[1]));
  min_b_[2] = static_cast<int> (floor (min_p[2] * inverse_leaf_size_[2]));
  max_b_[2] = static_cast<int> (floor (max_p[2] * inverse_leaf_size_[2]));

  // Compute the number of divisions needed along all axis
  div_b_ = max_b_ - min_b_ + Eigen::Vector4i::Ones ();
  div_b_[3] = 0;

  // Set up the division multiplier
  divb_mul_ = Vector4int64_t (1, div_b_[0], div_b_[0] * div_b_[1], 0);

  // Storage for mapping leaf and pointcloud indexes
  std::vector<cloud_point_index_idx_large> index_vector;
  index_vector.reserve (indices_->size ());

  // First pass: go over all points and insert them into the index_vector vector
  // with calculated idx. Points with the same idx value will contribute to the
  // same point of resulting CloudPoint
  for (std::vector<int>::const_iterator it = indices_->begin (); it != indices_->end (); ++it)
  {
    if (!input_->is_dense)
      // Check if the point is invalid
      if (!pcl_isfinite (input_->points[*it].x) ||
          !pcl_isfinite (input_->points[*it].y) ||
          !pcl_isfinite (input_->points[*it].z))
        continue;

    int ijk0 = static_cast<int> (floor (input_->points[*it].x * inverse_leaf_size_[0]) - static_cast<float> (min_b_[0]));
    int ijk1 = static_cast<int> (floor (input_->points[*it].y * inverse_leaf_size_[1]) - static_cast<float> (min_b_[1]));
    int ijk2 = static_cast<int> (floor (input_->points[*it].z * inverse_leaf_size_[2]) - static_cast<float> (min_b_[2]));

    // Compute the centroid leaf index
    int64_t idx = ijk0 * divb_mul_[0] + ijk1 * divb_mul_[1] + ijk2 * divb_mul_[2];
    index_vector.push_back (cloud_point_index_idx_large (idx, *it));
  }

  // Second pass: sort the index_vector vector using value representing target cell as index
  // in effect all points belonging to the same output cell will be next to each other
  std::sort (index_vector.begin (), index_vector.end (), std::less<cloud_point_index_idx_large> ());

  // Third pass: count output cells
  // we need to skip all the same, adjacenent idx values
  unsigned int total = 0;
  unsigned int index = 0;
  // first_and_last_indices_vector[i] represents the index in index_vector of the first point in
  // index_vector belonging to the voxel which corresponds to the i-th output point,
  // and of the first point not belonging to.
  std::vector<std::pair<unsigned int, unsigned int> > first_and_last_indices_vector;
  // Worst case size
  first_and_last_indices_vector.reserve (index_vector.size ());
  while (index < index_vector.size ())
  {
    unsigned int i = index + 1;
    while (i < index_vector.size () && index_vector[i].idx == index_vector[index].idx)
      ++i;
    if (i - index >= min_points_per_voxel_)
    {
      ++total;
      first_and_last_indices_vector.push_back (std::pair<unsigned int, unsigned int> (index, i));
    }
    index = i;
  }

  // Fourth pass: compute centroids, insert them into their final position
  output.points.resize (total);

  index = 0;
  for (unsigned int cp = 0; cp < first_and_last_indices_vector.size (); ++cp)
  {
    // calculate centroid - sum values from all input points, that have the same idx value in index_vector array
    unsigned int first_index = first_and_last_indices_vector[cp].first;
    unsigned int last_index = first_and_last_indices_vector[cp].second;

    CentroidPoint<PointT> centroid;

    // fill in the accumulator with leaf points
    for (unsigned int li = first_index; li < last_index; ++li)
      centroid.add (input_->points[index_vector[li].cloud_point_index]);

    centroid.get (output.points[index]);

    ++index;
  }
  output.width = static_cast<uint32_t> (output.points.size ());
}

#define PCL_INSTANTIATE_VoxelGridLarge(T) template class PCL_EXPORTS pcl::VoxelGridLarge<T>;

#endif    // PCL_FILTERS_IMPL_VOXEL_GRID_LARGE_H_

