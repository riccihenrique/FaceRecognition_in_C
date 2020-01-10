#include <dlib/image_processing.h>
#include <dlib/dnn.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <dlib/misc_api.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace dlib;

std::vector<dlib::rectangle> faceDetection(Mat frame, int scale = 0)
{
    dlib::frontal_face_detector detector = get_frontal_face_detector();

    //dlib::array2d<unsigned char> img;
   // dlib::assign_image(img, dlib::cv_image<dlib::bgr_pixel>(frame));

    cv_image<bgr_pixel> img(frame);
    //matrix<rgb_pixel> img;
    //assign_image(img, image);

    //for (int i = 0; i < scale; i++)
       // pyramid_up(img);

    std::vector<dlib::rectangle> dets = detector(img);
    return dets;
}

void drawRectangle(Mat frame, std::vector<dlib::rectangle> dets)
{
    for (int i = 0; i < dets.size(); i++)
    {
        cv::Rect rect(dets[i].left(), dets[i].top(), dets[i].width(), dets[i].height());
        cv::rectangle(frame, rect, cv::Scalar(0, 255, 0));
    }
}