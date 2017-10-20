/*
 *
 * This file is part of the open-source SeetaFace engine, which includes three modules:
 * SeetaFace Detection, SeetaFace Alignment, and SeetaFace Identification.
 *
 * This file is an example of how to use SeetaFace engine for face alignment, the
 * face alignment method described in the following paper:
 *
 *
 *   Coarse-to-Fine Auto-Encoder Networks (CFAN) for Real-Time Face Alignment, 
 *   Jie Zhang, Shiguang Shan, Meina Kan, Xilin Chen. In Proceeding of the
 *   European Conference on Computer Vision (ECCV), 2014
 *
 *
 * Copyright (C) 2016, Visual Information Processing and Learning (VIPL) group,
 * Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China.
 *
 * The codes are mainly developed by Jie Zhang (a Ph.D supervised by Prof. Shiguang Shan)
 *
 * As an open-source face recognition engine: you can redistribute SeetaFace source codes
 * and/or modify it under the terms of the BSD 2-Clause License.
 *
 * You should have received a copy of the BSD 2-Clause License along with the software.
 * If not, see < https://opensource.org/licenses/BSD-2-Clause>.
 *
 * Contact Info: you can send an email to SeetaFace@vipl.ict.ac.cn for any problems.
 *
 * Note: the above information must be kept whenever or wherever the codes are used.
 *
 */

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

#include "cv.h"
#include "highgui.h"

#include "face_detection.h"
#include "face_alignment.h"
#include "opencv2/highgui/highgui.hpp"  
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs/imgcodecs_c.h"
#define PI 3.14159265

using namespace cv;
using namespace std;

#ifdef _WIN32
std::string DATA_DIR = "../../data/";
std::string MODEL_DIR = "../../model/";
#else
std::string DATA_DIR = "./data/";
std::string MODEL_DIR = "./model/";
#endif

Mat rotate(Mat src, double angle)
{
    Mat dst;
    Point2f pt(src.cols/2., src.rows/2.);    
    Mat r = getRotationMatrix2D(pt, angle, 1.0);
    warpAffine(src, dst, r, Size(src.cols, src.rows));
    return dst;
}

int main(int argc, char** argv)
{
  if(argc != 3)
  {
    printf("Usage: %s src_path dst_path\n", argv[0]);
    return 0;
  }

  // long t0 = getTickCount();

  // Initialize face detection model
  seeta::FaceDetection detector("./model/seeta_fd_frontal_v1.0.bin");
  detector.SetMinFaceSize(40);
  detector.SetScoreThresh(2.f);
  detector.SetImagePyramidScaleFactor(0.8f);
  detector.SetWindowStep(4, 4);

  // Initialize face alignment model 
  MODEL_DIR = "./model/";
  seeta::FaceAlignment point_detector((MODEL_DIR + "seeta_fa_v1.1.bin").c_str());

  //load image
  IplImage *img_grayscale = NULL;
  img_grayscale = cvLoadImage(argv[1], 0);
  if (img_grayscale == NULL)
  {
    return 0;
  }

  IplImage *img_color = cvLoadImage(argv[1], 1);
  int pts_num = 5;
  int im_width = img_grayscale->width;
  int im_height = img_grayscale->height;
  unsigned char* data = new unsigned char[im_width * im_height];
  unsigned char* data_ptr = data;
  unsigned char* image_data_ptr = (unsigned char*)img_grayscale->imageData;
  int h = 0;
  for (h = 0; h < im_height; h++) 
  {
    memcpy(data_ptr, image_data_ptr, im_width);
    data_ptr += im_width;
    image_data_ptr += img_grayscale->widthStep;
  }

  seeta::ImageData image_data;
  image_data.data = data;
  image_data.width = im_width;
  image_data.height = im_height;
  image_data.num_channels = 1;

  // Detect faces
  std::vector<seeta::FaceInfo> faces = detector.Detect(image_data);
  int32_t face_num = static_cast<int32_t>(faces.size());

  if (face_num == 0)
  {
    delete[]data;
    cvReleaseImage(&img_grayscale);
    cvReleaseImage(&img_color);
    return 0;
  }
  // Detect 5 facial landmarks
  seeta::FacialLandmark points[5];
  point_detector.PointDetectLandmarks(image_data, faces[0], points);
  // for (int i = 0; i<pts_num; i++)
  // {
  //   cvCircle(img_color, cvPoint(points[i].x, points[i].y), 2, CV_RGB(0, 255, 0), CV_FILLED);
  // }

  Mat src = cvarrToMat(img_color);

  Mat dst;
  Point2f src_center(faces[0].bbox.x + faces[0].bbox.width/2.0, faces[0].bbox.y + faces[0].bbox.height/2.0);
  // printf("x: %lf y: %lf\n", src_center.x, src_center.y);
  double angle = atan((double)(points[1].y-points[0].y) / (double)(points[1].x-points[0].x)) * 180 / PI;
  // printf("angle: %lf\n", angle);
  Mat rot_mat = getRotationMatrix2D(src_center, angle, 1.0);
  warpAffine(src, dst, rot_mat, src.size(), 1);

  IplImage copy = dst;
  IplImage* new_image = &copy;
  // cvRectangle(new_image, cvPoint(faces[0].bbox.x, faces[0].bbox.y), cvPoint(faces[0].bbox.x + faces[0].bbox.width - 1, faces[0].bbox.y + faces[0].bbox.height - 1), CV_RGB(255, 0, 0));
  // cvSaveImage("new_image.jpg", new_image);

  Mat after_rot;
  after_rot = cvarrToMat(new_image);

  double x = faces[0].bbox.x;
  double y = faces[0].bbox.y;
  double width = faces[0].bbox.width;
  double height = faces[0].bbox.height;
  // printf("x: %lf\n", x);
  // printf("y: %lf\n", y);
  // printf("width: %lf\n", width);
  // printf("height: %lf\n\n", height);

  Mat temp = after_rot.clone();
  Mat cropped;
  double rate = 0.3;
  int new_x = (x - width*rate >= 0)? x - width*rate : 0;
  int new_y = (y - height*rate*1.5 >= 0)? y - height*rate*1.5 : 0;
  int new_width = (new_x + width*(1+2*rate) <= temp.cols)? width*(1+2*rate)-1 : temp.cols-new_x-1;
  int new_height = (new_y + height*(1+2*rate) <= temp.rows)? height*(1+2*rate)-1 : temp.rows-new_y-1;
  // printf("rate: %lf\n", rate);
  // printf("x: %d\n", new_x);
  // printf("y: %d\n", new_y);
  // printf("new_width: %d\n", new_width);
  // printf("new_height: %d\n\n", new_height);

  Rect roi(new_x, new_y, new_width, new_height);
  temp(roi).copyTo(cropped);
  imwrite(argv[2], cropped);

  // Release memory
  cvReleaseImage(&img_color);
  cvReleaseImage(&img_grayscale);
  delete[]data;

  // long t1 = getTickCount();
  // double secs = (t1 - t0)/getTickFrequency();
  // printf("secs: %lf\n", secs);
  return 0;
}

