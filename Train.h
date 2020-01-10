#include <dlib/image_processing.h>
#include <dlib/dnn.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <dlib/misc_api.h>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;
using namespace dlib;

// ------------------------ - TRAIN------------------------------

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;


template <int N, typename SUBNET> using res = relu<residual<block, N, bn_con, SUBNET>>;
template <int N, typename SUBNET> using ares = relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET> using res_down = relu<residual_down<block, N, bn_con, SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

// ----------------------------------------------------------------------------------------

template <typename SUBNET> using level0 = res_down<256, SUBNET>;
template <typename SUBNET> using level1 = res<256, res<256, res_down<256, SUBNET>>>;
template <typename SUBNET> using level2 = res<128, res<128, res_down<128, SUBNET>>>;
template <typename SUBNET> using level3 = res<64, res<64, res<64, res_down<64, SUBNET>>>>;
template <typename SUBNET> using level4 = res<32, res<32, res<32, SUBNET>>>;

template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;


// training network type
using net_type = loss_metric<fc_no_bias<128, avg_pool_everything<
    level0<
    level1<
    level2<
    level3<
    level4<
    max_pool<3, 3, 2, 2, relu<bn_con<con<32, 7, 7, 2, 2,
    input_rgb_image
    >>>>>>>>>>>>;

// testing network type (replaced batch normalization with fixed affine transforms)
using anet_type = loss_metric<fc_no_bias<128, avg_pool_everything<
    alevel0<
    alevel1<
    alevel2<
    alevel3<
    alevel4<
    max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2,
    input_rgb_image
    >>>>>>>>>>>>;

std::vector<std::vector<string>> load_objects_list(const string& dir)
{
    std::vector<std::vector<string>> objs;
    for (auto subdir : directory(dir).get_dirs())
    {       
        std::vector<string> imgs;
        for (auto img : subdir.get_files())
        {
            string name = img.full_name();
            if(name[0] != '|')
                imgs.push_back(name);
        }
        objs.push_back(imgs);
    }
    return objs;
}

std::vector<string> get_dirs(string dir)
{
    std::vector<string> splito;
    for (auto subdir : directory(dir).get_dirs())
        splito.push_back(subdir.name());
    return splito;
}

std::vector<float> convert_matrix_to_vector(std::vector<dlib::matrix<float, 0, 1>> descritores)
{
    std::vector<float> result;
    for (auto d : descritores[0])
        result.push_back(d);

    return result;
}

void trainModel()
{
    std::vector<std::vector<string>> imgs = load_objects_list("dataset");
    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor sp;
    deserialize("shape_predictor_5_face_landmarks.dat") >> sp;

    loss_metric<fc_no_bias<128, avg_pool_everything<alevel0<alevel1<alevel2<alevel3<alevel4<max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2, input_rgb_image_sized<150>>>>>>>>>>>>> net;
    deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;
    matrix<rgb_pixel> img;    
    
    std::vector<std::vector<float>> face_descriptors;
    deserialize("descritores") >> face_descriptors;
    std::vector<string> labels;
    deserialize("labels") >> labels;
    
    std::vector<string> dirs = get_dirs("dataset");
    for (int i = 0; i < imgs.size(); i++)
    {
        for (int j = 0; j < imgs[i].size(); j++)
        {
            load_image(img, imgs[i][j]);

            std::vector<matrix<rgb_pixel>> faces;
            for (auto face : detector(img))
            {
                auto shape = sp(img, face);
                matrix<rgb_pixel> face_chip;
                extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
                faces.push_back(move(face_chip));
            }

            if (faces.size() != 0)
            {
                std::vector<matrix<float, 0, 1>> descritores = net(faces); 
                face_descriptors.push_back(convert_matrix_to_vector(descritores));
                labels.push_back(dirs[i]);
            }
        }
    }

    serialize("descritores") << face_descriptors;
    serialize("labels") << labels;

}