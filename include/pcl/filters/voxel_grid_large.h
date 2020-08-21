/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2011, Willow Garage, Inc.
 *
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

#ifndef PCL_FILTERS_VOXEL_GRID_LARGE_MAP_H_
#define PCL_FILTERS_VOXEL_GRID_LARGE_MAP_H_

#include <pcl/filters/boost.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <map>

///////////////////////////////////////////////////////////////////////////////////////////

struct cloud_point_index_idx_large
{
  uint64_t idx;
  unsigned int cloud_point_index;

  cloud_point_index_idx_large (uint64_t idx_, unsigned int cloud_point_index_) : idx (idx_), cloud_point_index (cloud_point_index_) {}
  bool operator < (const cloud_point_index_idx_large &p) const { return (idx < p.idx); }
};

using Vector4int64_t = Eigen::Matrix<int64_t, 4, 1>;
using Vector3int64_t = Eigen::Matrix<int64_t, 3, 1>;

namespace pcl
{
  /** \brief VoxelGridLarge assembles a local 3D grid over a given PointCloud, and downsamples + filters the data.
    *
    * The VoxelGridLarge class creates a *3D voxel grid* (think about a voxel
    * grid as a set of tiny 3D boxes in space) over the input point cloud data.
    * Then, in each *voxel* (i.e., 3D box), all the points present will be
    * approximated (i.e., *downsampled*) with their centroid. This approach is
    * a bit slower than approximating them with the center of the voxel, but it
    * represents the underlying surface more accurately.
    *
    * \author Radu B. Rusu, Bastian Steder
    * \ingroup filters
    */
  template <typename PointT>
  class VoxelGridLarge: public Filter<PointT>
  {
    protected:
      using Filter<PointT>::filter_name_;
      using Filter<PointT>::getClassName;
      using Filter<PointT>::input_;
      using Filter<PointT>::indices_;

      typedef typename Filter<PointT>::PointCloud PointCloud;
      typedef typename PointCloud::Ptr PointCloudPtr;
      typedef typename PointCloud::ConstPtr PointCloudConstPtr;
      typedef boost::shared_ptr< VoxelGridLarge<PointT> > Ptr;
      typedef boost::shared_ptr< const VoxelGridLarge<PointT> > ConstPtr;


    public:
      /** \brief Empty constructor. */
      VoxelGridLarge () :
        leaf_size_ (Eigen::Vector4f::Zero ()),
        inverse_leaf_size_ (Eigen::Array4f::Zero ()),
        min_b_ (Eigen::Vector4i::Zero ()),
        max_b_ (Eigen::Vector4i::Zero ()),
        div_b_ (Eigen::Vector4i::Zero ()),
        divb_mul_ (Vector4int64_t::Zero ()),
        min_points_per_voxel_ (0)
      {
        filter_name_ = "VoxelGridLarge";
      }

      /** \brief Destructor. */
      virtual ~VoxelGridLarge ()
      {
      }

      /** \brief Set the voxel grid leaf size.
        * \param[in] leaf_size the voxel grid leaf size
        */
      inline void
      setLeafSize (const Eigen::Vector4f &leaf_size)
      {
        leaf_size_ = leaf_size;
        // Avoid division errors
        if (leaf_size_[3] == 0)
          leaf_size_[3] = 1;
        // Use multiplications instead of divisions
        inverse_leaf_size_ = Eigen::Array4f::Ones () / leaf_size_.array ();
      }

      /** \brief Set the voxel grid leaf size.
        * \param[in] lx the leaf size for X
        * \param[in] ly the leaf size for Y
        * \param[in] lz the leaf size for Z
        */
      inline void
      setLeafSize (float lx, float ly, float lz)
      {
        leaf_size_[0] = lx; leaf_size_[1] = ly; leaf_size_[2] = lz;
        // Avoid division errors
        if (leaf_size_[3] == 0)
          leaf_size_[3] = 1;
        // Use multiplications instead of divisions
        inverse_leaf_size_ = Eigen::Array4f::Ones () / leaf_size_.array ();
      }

      /** \brief Get the voxel grid leaf size. */
      inline Eigen::Vector3f
      getLeafSize () { return (leaf_size_.head<3> ()); }

      /** \brief Set the minimum number of points required for a voxel to be used.
        * \param[in] min_points_per_voxel the minimum number of points for required for a voxel to be used
        */
      inline void
      setMinimumPointsNumberPerVoxel (unsigned int min_points_per_voxel) { min_points_per_voxel_ = min_points_per_voxel; }

      /** \brief Return the minimum number of points required for a voxel to be used.
       */
      inline unsigned int
      getMinimumPointsNumberPerVoxel () { return min_points_per_voxel_; }

      /** \brief Get the minimum coordinates of the bounding box (after
        * filtering is performed).
        */
      inline Eigen::Vector3i
      getMinBoxCoordinates () { return (min_b_.head<3> ()); }

      /** \brief Get the minimum coordinates of the bounding box (after
        * filtering is performed).
        */
      inline Eigen::Vector3i
      getMaxBoxCoordinates () { return (max_b_.head<3> ()); }

      /** \brief Get the number of divisions along all 3 axes (after filtering
        * is performed).
        */
      inline Eigen::Vector3i
      getNrDivisions () { return (div_b_.head<3> ()); }

      /** \brief Get the multipliers to be applied to the grid coordinates in
        * order to find the centroid index (after filtering is performed).
        */
      inline Vector3int64_t
      getDivisionMultiplier () { return (divb_mul_.head<3> ()); }

    protected:
      /** \brief The size of a leaf. */
      Eigen::Vector4f leaf_size_;

      /** \brief Internal leaf sizes stored as 1/leaf_size_ for efficiency reasons. */
      Eigen::Array4f inverse_leaf_size_;

      /** \brief The minimum and maximum bin coordinates, and the number of divisions. */
      Eigen::Vector4i min_b_, max_b_, div_b_;

      /** \brief The division multiplier. */
      Vector4int64_t divb_mul_;

      /** \brief Minimum number of points per voxel for the centroid to be computed */
      unsigned int min_points_per_voxel_;

      typedef typename pcl::traits::fieldList<PointT>::type FieldList;

      /** \brief Downsample a Point Cloud using a voxelized grid approach
        * \param[out] output the resultant point cloud message
        */
      void
      applyFilter (PointCloud &output);
  };

}

#ifdef PCL_NO_PRECOMPILE
#include <pcl/filters/impl/voxel_grid_large.hpp>
#endif

#endif  //#ifndef PCL_FILTERS_VOXEL_GRID_LARGE_MAP_H_
