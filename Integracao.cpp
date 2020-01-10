#include "Train.h"
#include "Recognition.h"
#include "Detection.h"

using namespace std;
using namespace cv;
using namespace dlib;

void webcamTest()
{
    cv::VideoCapture cap;
    cv::Mat frame;

    if (!cap.open(0))
        return;
    while (true)
    {
        cap >> frame;
        std::vector<dlib::rectangle> dets = faceDetection(frame, 0);
        if (dets.size() > 0)
        {
            faceRecognition(dets, frame);
            drawRectangle(frame, dets);
        }
        cv::imshow("Teste", frame);
        if (cv::waitKey(1) == 27)
            return;
    }
}

void imgtest(string image_path)
{
    Mat frame = imread(image_path);
    std::vector<dlib::rectangle> dets = faceDetection(frame, 0);
    if (dets.size() > 0)
    {
        faceRecognition(dets, frame);
        drawRectangle(frame, dets);
    }
    cv::imshow("Teste", frame);
    if (cv::waitKey(1) == 27)
        return;
}

int main()
{
    //trainModel();
    webcamTest();
    //imgtest("test.bmp");
    
    return 1;
}
