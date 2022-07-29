//
// Created by Rqg on 2022/7/25.
//

#include "yolo_v6.h"
#include "inference_backend/image_inference.h"
#include "inference_backend/logger.h"
#include "safe_arithmetic.hpp"
#include <fstream>

namespace post_processing {
size_t load_file_content(std::string file_path, char *&buf, bool text) {
    //    auto fd = fopen(file_path.c_str(), text ? "r" : "rb");
    //
    //    fseek(fd,)
    std::ifstream file(file_path, text ? std::ifstream::in : std::ifstream::binary);
    // get length of file:
    file.seekg(0, file.end);
    long buf_len = file.tellg();
    file.seekg(0, file.beg);
    // allocate memory:
    buf = new char[buf_len + 1];
    buf[buf_len] = '\0';
    // read data as a block:
    file.read(buf, buf_len);
    file.close();
    //    auto hash = md5(std::string(buf, buf_len));
    //    printf("%s -- %lld ---- %s\n", file_path.c_str(), buf_len.operator long long(), hash.c_str());
    return buf_len;
}
void cmp_blob(const float *blob_data, const std::vector<size_t> &blob_dims) {
    std::string file_name = "/home/aibox/test_data/";
    int blob_size = 1;
    for (auto d : blob_dims) {
        file_name += std::to_string(d);
        file_name += "_";
        blob_size *= d;
    }
    file_name.pop_back();

    GST_WARNING("cmp_blob %s", file_name.c_str());

    char *buf;
    auto buf_size = load_file_content(file_name, buf, false);

    float *py_blob = (float *)buf;

    double diff = 0;

    for (int i = 0; i < blob_size; ++i) {
        auto d = (blob_data[i] - py_blob[i]);
        diff += d * d;
    }

    GST_WARNING("%s -- diff:%f", file_name.c_str(), diff / blob_size);
}
void YoloV6Converter::parseOutputBlob(const float *blob_data, const std::vector<size_t> &blob_dims, size_t blob_size,
                                      std::vector<DetectedObject> &objects, int stride) const {

    g_assert(blob_dims.size() == 5);
    g_assert(blob_dims[1] == 1);
    g_assert(blob_dims[4] == (size_t)(v6initializer.classes_number + 5));
    //    cmp_blob(blob_data, blob_dims);

    // 1xHxWx(n+5)
    int h = blob_dims[blob_dims.size() - 3];
    int w = blob_dims[blob_dims.size() - 2];
    int na = blob_dims[blob_dims.size() - 1];
    const float *raw;
    float conf;
    for (int i = 0; i < na; ++i) {
        printf("%f,", blob_data[i]);
    }
    printf("\n");
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            raw = blob_data + (i * w + j) * na;
            if (raw[4] < confidence_threshold) {
                continue;
            }
            auto bx = (raw[0] + j) * stride;
            auto by = (raw[1] + i) * stride;
            auto bw = expf(raw[2]) * stride;
            auto bh = expf(raw[3]) * stride;

            for (int k = 0; k < v6initializer.classes_number; ++k) {
                conf = raw[5 + k] * raw[4];

                if (conf < confidence_threshold) {
                    continue;
                }

//                GST_WARNING("conf:%f -- x:%f y:%f, w:%f, h:%f , stride:%d, x:%d/%d, y:%d/%d, na:%d", conf, bx, by, bw,
//                            bh, stride, j, w, i, h, na);

                objects.emplace_back(bx, by, bw, bh, conf, k, getLabelByLabelId(k),
                                     1.0f / getModelInputImageInfo().width, 1.0f / getModelInputImageInfo().height,
                                     true);
                //                return;
            }
        }
    }
}

TensorsTable YoloV6Converter::convert(const OutputBlobs &output_blobs) const {
    ITT_TASK(__FUNCTION__);
    try {
        const auto &model_input_image_info = getModelInputImageInfo();
        size_t batch_size = model_input_image_info.batch_size;

        DetectedObjectsTable objects_table(batch_size);
        int layer_size = v6initializer.layer_names.size();
        for (size_t batch_number = 0; batch_number < batch_size; ++batch_number) {
            auto &objects = objects_table[batch_number];

            for (int i = 0; i < layer_size; ++i) {
                const auto &blob_iter = output_blobs.at(v6initializer.layer_names[i]);
                const InferenceBackend::OutputBlob::Ptr &blob = blob_iter;
                if (not blob)
                    throw std::invalid_argument("Output blob is nullptr.");

                size_t unbatched_size = blob->GetSize() / batch_size;
                parseOutputBlob(reinterpret_cast<const float *>(blob->GetData()) + unbatched_size * batch_number,
                                blob->GetDims(), unbatched_size, objects, v6initializer.strides[i]);
            }
        }
        return storeObjects(objects_table);
    } catch (const std::exception &e) {

        std::throw_with_nested(std::runtime_error("Failed to do YoloV6 post-processing."));
    }
    return TensorsTable{};
}
YoloV6Converter::YoloV6Converter(BlobToMetaConverter::Initializer initializer, double confidence_threshold,
                                 double iou_threshold, YoloV6Converter::Initializer v6Initializer)
    : BlobToROIConverter(std::move(initializer), confidence_threshold, true, iou_threshold),
      v6initializer(std::move(v6Initializer)) {
}

std::vector<int> getStrides(GstStructure *s) {
    if (!gst_structure_has_field(s, "strides"))
        throw std::runtime_error("model proc does not have \"strides\" parameter.");

    GValueArray *arr = nullptr;
    gst_structure_get_array(s, "strides", &arr);

    std::vector<int> strides;
    if (arr) {
        strides.reserve(arr->n_values);
        for (guint i = 0; i < arr->n_values; ++i)
            strides.push_back(g_value_get_int(g_value_array_get_nth(arr, i)));
        g_value_array_free(arr);
    } else {
        throw std::runtime_error("\"strides\" array is null.");
    }

    return strides;
}

double getIOUThreshold(GstStructure *s) {
    double iou_threshold = 0.5;
    if (gst_structure_has_field(s, "iou_threshold")) {
        gst_structure_get_double(s, "iou_threshold", &iou_threshold);
    }

    return iou_threshold;
}

std::vector<std::string> getLayerNames(GstStructure *s) {
    std::vector<std::string> layers;

    if (s == nullptr)
        throw std::runtime_error("Can not get model_proc output information.");

    if (!gst_structure_has_field(s, "layer_names"))
        return layers;

    if (gst_structure_has_field(s, "layer_names")) {
        GValueArray *arr = nullptr;
        gst_structure_get_array(const_cast<GstStructure *>(s), "layer_names", &arr);
        if (arr and arr->n_values) {
            for (guint i = 0; i < arr->n_values; ++i)
                layers.emplace_back(g_value_get_string(g_value_array_get_nth(arr, i)));
            g_value_array_free(arr);
        } else {
            throw std::runtime_error("\"layer_names\" array is null.");
        }
    }

    return layers;
}

BlobToMetaConverter::Ptr YoloV6Converter::create(BlobToMetaConverter::Initializer initializer,
                                                 double confidence_threshold) {
    YoloV6Converter::Initializer v6Initializer{};
    GstStructure *model_proc_output_info = initializer.model_proc_output_info.get();
    double iou_threshold;
    try {
        v6Initializer.strides = getStrides(model_proc_output_info);
        const auto classes_number = initializer.labels.size();
        if (!classes_number)
            throw std::runtime_error("Number of classes if null.");
        v6Initializer.classes_number = classes_number;
        iou_threshold = getIOUThreshold(model_proc_output_info);
        v6Initializer.layer_names = getLayerNames(model_proc_output_info);
    } catch (const std::exception &e) {
        std::throw_with_nested(std::runtime_error("Failed to create \" yolo_v6 \" converter."));
    }

    return BlobToMetaConverter::Ptr(
        new YoloV6Converter(std::move(initializer), confidence_threshold, iou_threshold, std::move(v6Initializer)));
}

std::string YoloV6Converter::getName() {
    return "yolo_v6";
}

} // namespace post_processing