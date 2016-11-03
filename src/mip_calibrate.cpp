/*!
  \file        mip_calibrate.cpp
  \author      Arnaud Ramey <arnaud.a.ramey@gmail.com>
                -- Robotics Lab, University Carlos III of Madrid
  \date        2015/08

________________________________________________________________________________

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
________________________________________________________________________________
\brief Detects MIP in the depth image using a simple background
substraction.
The background model can be loaded from a static image.
It can also be updated at each frame.
The size of the blobs (in pixels) to keep can be configured,
and also the minimum and maximum position of the upper points of the blob:
this can be useful to discriminate between users and other objects for instance.

\section Parameters
  - \b "~foreground_min_depth_ratio"
        [double (ratio), default: .9]
        For each pixel, the minimum ratio current depth / bg depth
        to consider this pixel as not belonging to the background

  - \b "~foreground_min_bw_diff"
        [int, default: 20]
        For each pixel, the minimum ratio current gray level / bg gray level
        to consider this pixel as not belonging to the background

  - \b "~min_comp_size"
        [int (pixels), default: 200 pixels]
        The minimal size for a connected component to be considered to be an object.

  - \b "~min_comp_z, max_comp_z"
        [double (meters), default: -10 and 10]
        The minimal and maximal values of the z component
        for a connected component to be considered to be an object.

  - \b "~static_frame"
        [string, default:"/floor"]
        The static frame in which min_comp_z and max_comp_z are defined.

  - \b "~canny_thres1, canny_thres2"
        [double (meters), default: DepthCanny::DEFAULT_CANNY_THRES1/2]
        \see DepthCanny::set_canny_thresholds()
        Decrease to make more edges appear.

  - \b "~update_background"
        [bool, default: true]
        If true, the background model will be updated upon reception of
        each depth frame.
        Points further in the current

  - \b "~background_models_filename_prefix"
        [std::string, default:""]
        Pre-load a background model from an image file on the disk.
        If empty, do not pre-load anything, use the first acquired depth image as model.

\section Subscriptions
  - \b {start_topic}, {stop_topic}
        [std_msgs::Int16]
        \see RgbDepthSkill.

  - \b {rgb_topic}, {depth_topic}
        [sensor_msgs::Image]
        \see RgbDepthSkill

\section Publications
  None

TODO see integration of https://github.com/andrewssobral/bgslibrary ?
 */
// AD

#include "vision_utils/depth_canny.h"
#include "vision_utils/disjoint_sets2.h"
#include "vision_utils/rgb_depth_skill.h"
#include "vision_utils/timer.h"
#include "vision_utils/timestamp.h"
// ROS
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <ros/package.h>

#undef DEBUG_PRINT
#define DEBUG_PRINT(...)   printf(__VA_ARGS__)


class MIPCalibrate : public RgbDepthSkill  {
public:
  MIPCalibrate() : RgbDepthSkill("MIP_CALIBRATE_START",
                                 "MIP_CALIBRATE_STOP")
  {
    _tf_listener = new tf::TransformListener();

    // parameters
    std::string bg_prefix = "";
    _nh_private.param("background_models_filename_prefix", bg_prefix, bg_prefix);
    _nh_private.param("update_background", _update_background, true);
    _nh_private.param("use_rgb", _use_rgb, false);
    _nh_private.param("use_depth_canny", _use_depth_canny, true);
    _min_comp_size = 200;
    _nh_private.param("min_comp_size", _min_comp_size, _min_comp_size);
    _nh_private.param("foreground_min_depth_ratio", _foreground_min_depth_ratio, .9);
    _nh_private.param("foreground_min_bw_diff", _foreground_min_bw_diff, 20);
    _static_frame = "/floor";
    _nh_private.param("static_frame", _static_frame, _static_frame);
    _nh_private.param("min_comp_z", _min_comp_z, -10.);
    _nh_private.param("max_comp_z", _max_comp_z, 10.);
    double canny_thres1, canny_thres2;
    const double DEFAULT_CANNY_THRES1_auxConst = DepthCanny::DEFAULT_CANNY_THRES1;
    const double DEFAULT_CANNY_THRES2_auxConst = DepthCanny::DEFAULT_CANNY_THRES2;
    _nh_private.param("canny_thres1", canny_thres1, DEFAULT_CANNY_THRES1_auxConst);
    _nh_private.param("canny_thres1", canny_thres2, DEFAULT_CANNY_THRES2_auxConst);
    _canny.set_canny_thresholds(canny_thres1, canny_thres2);

    if (!bg_prefix.empty()) {
      cv::Mat3b bw_background3b;
      bool has_rgb_model = vision_utils::read_rgb_and_depth_image_from_image_file
          (bg_prefix, &bw_background3b, NULL);
      // grayscale conversion
      cv::cvtColor(bw_background3b, _bw_background, CV_BGR2GRAY);
      bool has_depth_model = vision_utils::read_rgb_and_depth_image_from_image_file
          (bg_prefix, NULL, &_depth_background);
      imshow_backgrounds(has_rgb_model, has_depth_model);
    }

    // get camera model
    image_geometry::PinholeCameraModel rgb_camera_model;
    vision_utils::read_camera_model_files
        (DEFAULT_KINECT_SERIAL(), _default_depth_camera_model, rgb_camera_model);

    std::ostringstream info;
    info << "MIPCalibrate: started with '" << get_start_stopic()
         << "' and stopped with '" << get_stop_stopic()
         << "', subscribing to '" << get_rgb_topic() << "', '" << get_depth_topic()
         << "', static_frame:'" << _static_frame
         << "', update_background:" << _update_background
         << ", use_rgb:" << _use_rgb
         << ", display:" << _display;
    if (bg_prefix.empty())
      info << ", without bg model. ";
    else
      info << ", using bg model '" << bg_prefix << "'. ";
    info << "Press ' ' in the windows to define a new background model. ";
    info << "Press 'm' in the windows to switch between depth and rgb background substraction. ";
    if (_display)
      info << "Press 's' in the windows to save background models to disk.";
    DEBUG_PRINT("%s\n", info.str().c_str());

    if (_display) {
      cv::namedWindow("MIPCalibrate-bw_background");
      cv::createTrackbar("foreground_min_bw_diff", "MIPCalibrate-bw_background",
                         &_foreground_min_bw_diff, 255);
    }
  } // end ctor

  //////////////////////////////////////////////////////////////////////////////

  virtual void create_subscribers_and_publishers() {
    printf("create_subscribers_and_publishers()\n");
  }

  //////////////////////////////////////////////////////////////////////////////

  virtual void shutdown_subscribers_and_publishers() {}

  //////////////////////////////////////////////////////////////////////////////

  //! this function is called each time an image is received
  virtual void process_rgb_depth(const cv::Mat3b & rgb,
                                 const cv::Mat1f & depth) {
    DEBUG_PRINT("process_rgb_depth()\n");

    vision_utils::Timer timer;
    // clear previous data
    _cuts_offsets.clear();
    _rgb_cuts.clear();
    _depth_cuts.clear();
    _user_cuts.clear();
    _blobs_centers2d3d.clear();

    // compute foreground
    // foreground objects are such as background - depth > thresh
    _foreground.create(depth.size());
    _foreground.setTo(0);
    unsigned int n_pixels = depth.cols * depth.rows;
    uchar* foreground_ptr = _foreground.ptr();

    if (_use_rgb) {
      // grayscale conversion
      cv::cvtColor(rgb, _bw_frame, CV_BGR2GRAY);
      cv::blur(_bw_frame, _bw_frame, cv::Size(3, 3));
      // equalize hist
      cv::equalizeHist(_bw_frame, _bw_frame);
      // create a background from rgb the first time
      if (_bw_background.empty()) {
        _bw_frame.copyTo(_bw_background);
        imshow_backgrounds(true, false);
      }
      cv::absdiff(_bw_frame, _bw_background, _foreground_nothres);
      cv::threshold(_foreground_nothres, _foreground, _foreground_min_bw_diff, 255, CV_THRESH_BINARY);
    } // end if (_use_rgb)
    else { // use depth
      // create a background from depth the first time
      if (_depth_background.empty()) {
        depth.copyTo(_depth_background);
        imshow_backgrounds(false, true);
      }
      const float* depth_ptr = depth.ptr<float>();
      float* background_ptr = _depth_background.ptr<float>();
      for (unsigned int pixel_idx = 0; pixel_idx < n_pixels; ++pixel_idx) {
        // remove from foreground the points were depth is not defined (0)
        if(!vision_utils::is_nan_depth(*depth_ptr)
           && *depth_ptr >= _min_comp_z
           && *depth_ptr <= _max_comp_z) {
          bool background_defined = !vision_utils::is_nan_depth(*background_ptr);
          if (!background_defined
              || *depth_ptr < *background_ptr * _foreground_min_depth_ratio) { // this pixel is foreground
            *foreground_ptr = 255;
          }
          if (_update_background && // set new background
              (!background_defined || *depth_ptr > *background_ptr))
            *background_ptr = *depth_ptr;
        }
        ++depth_ptr;
        ++background_ptr;
        ++foreground_ptr;
      } // end loop pixel_idx
    } // end if use depth
    //cv::morphologyEx(_foreground, _foreground, cv::MORPH_OPEN, cv::Mat(10, 10, CV_8U, 255));
    //cv::morphologyEx(_foreground, _foreground, cv::MORPH_ERODE, cv::Mat(15, 15, CV_8U, 255));
    DEBUG_PRINT("Time for foreground computation:%g ms\n", timer.time());

    // compute a depth Canny
    if (_use_depth_canny) {
      _canny.thresh(depth);
      // combine depth Canny and background
      // all points equal to 0 in Canny must be set to 0 in foreground
      _foreground.setTo(0, _canny.get_thresholded_image() == 0);
      DEBUG_PRINT("Time for Canny:%g ms\n", timer.time());
    } // end if use_depth_canny

    // use disjoint sets to get components
    // get foreground points ROIS with disjoint sets
    _set.process_image(_foreground);
    _set.get_connected_components(_foreground.cols, _components_pts, _boundingBoxes);
    _set.sort_comps_by_decreasing_size(_components_pts, _boundingBoxes);
    unsigned int nbboxes = _boundingBoxes.size();
    DEBUG_PRINT("Time for Disjoint sets:%g ms, %i boxes\n", timer.time(), nbboxes);
    _comp_was_kept.resize(nbboxes, false);

    // for each found component
    _cuts_offsets.reserve(nbboxes);
    _rgb_cuts.reserve(nbboxes);
    _depth_cuts.reserve(nbboxes);
    _user_cuts.reserve(nbboxes);
    _blobs_centers2d3d.reserve(nbboxes);

    for (unsigned int bbox_idx = 0; bbox_idx < nbboxes; ++bbox_idx) {
      if ((int) _components_pts[bbox_idx].size() < _min_comp_size) // comp too small!
        break;
      //std::vector<cv::Point>* comp = &(_components_pts[bbox_idx]);
      cv::Rect roi = _boundingBoxes[bbox_idx];
      cv::Point tl = roi.tl();
      // reproject lower middle 3D
      cv::Point2d head2D (tl.x + roi.width / 2, tl.y + roi.height * .9);
      cv::Point3d head3D_world, head3D_cam =
          vision_utils::pixel2world_depth<cv::Point3d>
          (head2D, _default_depth_camera_model, depth);
      try {
        geometry_msgs::PoseStamped pose_cam, pose_world;
        vision_utils::copy3(head3D_cam, pose_cam.pose.position);
        pose_cam.pose.orientation = tf::createQuaternionMsgFromYaw(0);
        pose_cam.header = _images_header;
        pose_world.header.stamp = _images_header.stamp;
        pose_world.header.frame_id = _static_frame;
        _tf_listener->transformPose(_static_frame, ros::Time(0),
                                    pose_cam, _static_frame, pose_world);
        vision_utils::copy3(pose_world.pose.position, head3D_world);
      } catch (std::runtime_error e) {
        ROS_WARN("transform error:'%s'", e.what());
        continue;
      }
      DEBUG_PRINT("Head %i:%s\n", bbox_idx, vision_utils::printP(head3D_world).c_str());

      // keep it in _blobs_centers2d3d
      std::pair<cv::Point, cv::Point3d> new_pair(head2D, head3D_world);
      _blobs_centers2d3d.push_back(new_pair);

      // if size satisfying, keep it in _rgb_cuts, _depth_cuts, _user_cuts, etc.
      _comp_was_kept[bbox_idx] = true;
      _cuts_offsets.push_back(tl);
      _rgb_cuts.push_back(rgb(roi));
      _depth_cuts.push_back(depth(roi));
      _user_cuts.push_back(_foreground(roi));
    } // end for bbox_idx

    unsigned int nusers = _blobs_centers2d3d.size();
    // build_ppl_message();
    DEBUG_PRINT("Time for process_rgb_depth():%g ms, %i users\n", timer.time(), nusers);
    if (_display) display(rgb, depth);
  } // end process_rgb_depth();

  //////////////////////////////////////////////////////////////////////////////

  void imshow_backgrounds(bool bw, bool depth) {
    if (_display && bw && !_bw_background.empty())
      cv::imshow("MIPCalibrate-bw_background", _bw_background);
    if (_display && depth && !_bw_background.empty())
      cv::imshow("MIPCalibrate-depth_background",
                 vision_utils::depth2viz(_depth_background, vision_utils::FULL_RGB_STRETCHED));
  }

  //////////////////////////////////////////////////////////////////////////////

  void display(const cv::Mat3b & rgb,
               const cv::Mat1f & depth) {
    //    if (_use_rgb) {
    //      cv::imshow("rgb", rgb);
    //      cv::imshow("bw_frame", _bw_frame);
    //    }
    //    else
    //      cv::imshow("depth", vision_utils::depth2viz(depth, vision_utils::FULL_RGB_STRETCHED));

    //cv::imshow("MIPCalibrate_foreground", _foreground);
    cv::cvtColor(_foreground, img_out, CV_GRAY2BGR);
    // paint components
    for (unsigned int comp_idx = 0; comp_idx < _components_pts.size(); ++comp_idx) {
      if (_comp_was_kept[comp_idx])
        vision_utils::drawListOfPoints
            (img_out, _components_pts[comp_idx],
             vision_utils::color<cv::Vec3b>(comp_idx));
    } // end for comp_idx
    // paints strings for the blobs 3D positions
    for (unsigned int blob_idx = 0; blob_idx < _blobs_centers2d3d.size(); ++blob_idx) {
      cv::circle(img_out, _blobs_centers2d3d[blob_idx].first, 5, CV_RGB(150, 150, 150), -1);
      cv::Point3d blob3D = _blobs_centers2d3d[blob_idx].second;
      std::ostringstream text;
      text << "(" << std::setprecision(2) << blob3D.x
           << ", " << blob3D.y << ", " << blob3D.z << ")";
      cv::putText(img_out, text.str(),
                  _blobs_centers2d3d[blob_idx].first,
                  CV_FONT_HERSHEY_PLAIN, 1, CV_RGB(255, 255, 255));
    }
    cv::imshow("MIPCalibrate_objects", img_out);

    char c = cv::waitKey(25);
    if (c == 's') {
      std::ostringstream prefix;
      prefix << ros::package::getPath("mip_calibrate") << "/data/";
      prefix << "MIPCalibrate_" << vision_utils::timestamp();
      vision_utils::write_rgb_and_depth_image_to_image_file
          (prefix.str(),
           (_bw_background.empty() ? NULL : &_bw_background),
           (_depth_background.empty() ? NULL : &_depth_background));
    } // end 's'
    else if (c == 'm') {
      _use_rgb = !_use_rgb;
    } // end 'm'
    else if (c == ' ') {
      printf("MIPCalibrate: storing new background\n");
      if (_use_rgb) {
        _bw_frame.copyTo(_bw_background);
      }
      else { // depth
        depth.copyTo(_depth_background);
        if (_display)
          cv::imshow("MIPCalibrate_background", vision_utils::depth2viz(_depth_background, vision_utils::FULL_RGB_STRETCHED));
      }
    } // end ' '
  } // end display();

  //////////////////////////////////////////////////////////////////////////////

private:
  //! Canny
  DepthCanny _canny;

  //! disjoint sets
  DisjointSets2 _set;
  std::vector< std::vector<cv::Point> > _components_pts;
  std::vector<cv::Rect> _boundingBoxes;
  std::vector<bool> _comp_was_kept;
  std::vector< std::pair<cv::Point, cv::Point3d> > _blobs_centers2d3d;
  /*! the difference between a pixel depth and the background depth
  in meters to be considered as foreground */
  double _foreground_min_depth_ratio;
  int _foreground_min_bw_diff;
  int _min_comp_size; // pixels
  std::string _static_frame;
  double _min_comp_z, _max_comp_z;

  cv::Mat1f _depth_background;
  cv::Mat1b _bw_background, _bw_frame, _foreground_nothres;
  cv::Mat1b _foreground;
  bool _update_background, _use_rgb, _use_depth_canny;
  cv::Mat1b _user_mask;

  //! the list of blobs, with ROS orientation, in RGB frame
  tf::TransformListener* _tf_listener;

  std::vector< cv::Point > _cuts_offsets;
  std::vector<cv::Mat3b> _rgb_cuts;
  std::vector<cv::Mat1f> _depth_cuts;
  std::vector<cv::Mat1b> _user_cuts;

  //! clustering
  image_geometry::PinholeCameraModel _default_depth_camera_model;

  //! an image for drawing stuff
  cv::Mat3b img_out;
}; // end class MIPCalibrate

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
  ros::init(argc, argv, "MIPCalibrate");
  MIPCalibrate skill;
  skill.check_autostart();
  ros::spin();
  return 0;
}

