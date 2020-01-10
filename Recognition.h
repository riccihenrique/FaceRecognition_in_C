#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <math.h>
#include <Windows.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;
using namespace dlib;

// --------------------- RECOGNITION ---------------------
template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET> using ares = relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

//using anet_type = loss_metric<fc_no_bias<128, avg_pool_everything<alevel0<alevel1<alevel2<alevel3<alevel4<max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2,input_rgb_image_sized<150>>>>>>>>>>>>>;

// ----------------------------------------------------------------------------------------

std::vector<float> convert_matrix_to_vector(matrix<float, 0, 1> descritores)
{
    std::vector<float> result;
    for (auto d : descritores)
        result.push_back(d);

    return result;
}

std::vector<float> distanciaEuclidiana(std::vector<std::vector<float>> descs, std::vector<float> desc)
{
    std::vector<std::vector<float>> vetaux;
    for (int i = 0; i < descs.size(); i++)
    {
        std::vector<float> mataux;
        for (int j = 0; j < 128; j++)
            mataux.push_back(desc[j] - descs[i][j]);

        vetaux.push_back(mataux);
    }

    std::vector<float> res;
    float som;
    for (auto mat : vetaux)
    {
        som = 0;
        for (auto val : mat)
            som += pow(val, 2);

        res.push_back(sqrt(som));
    }
    return res;
}

int minarg(std::vector<float> vet)
{
    float min = vet[0];
    int min_index = 0;
    for(int i = 1; i < vet.size(); i++)
        if (vet[i] < min)
        {
            min = vet[i];
            min_index = i;
        }
    return min_index;
}

void faceRecognition(std::vector<dlib::rectangle> dets, cv::Mat frame)
{
    frontal_face_detector detector = get_frontal_face_detector();

    shape_predictor sp;
    deserialize("shape_predictor_5_face_landmarks.dat") >> sp;

    loss_metric<fc_no_bias<128, avg_pool_everything<alevel0<alevel1<alevel2<alevel3<alevel4<max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2, input_rgb_image_sized<150>>>>>>>>>>>>> net;
    deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;

    float limiar = 0.6;

    std::vector<std::vector<float>> face_descriptors;
    deserialize("descritores") >> face_descriptors;

    std::vector<string> labels;
    deserialize("labels") >> labels;
    
    std::vector<matrix<rgb_pixel>> faces;
    cv_image<bgr_pixel> img(frame);

    for (auto face : detector(img))
    {
        auto shape = sp(img, face);
        matrix<rgb_pixel> face_chip;
        extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
        faces.push_back(move(face_chip));    
    }

    if (faces.size() != 0)
    {
        std::vector<matrix<float, 0, 1>> descritorFacial = net(faces);
        for (auto face : descritorFacial)
        {
            std::vector<float> distancias = distanciaEuclidiana(face_descriptors, convert_matrix_to_vector(face));
            int index = minarg(distancias);
            if (distancias[index] <= limiar)
                cout << "Encontrei o " << labels[index] << endl;
        }
    }
}