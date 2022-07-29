//
// Created by Rqg on 2022/7/25.
//

#ifndef DL_STREAMER_YOLO_V6_H
#define DL_STREAMER_YOLO_V6_H
#pragma once

#include "blob_to_roi_converter.h"

namespace post_processing {

class YoloV6Converter : public BlobToROIConverter {
  public:
    struct Initializer {
        std::vector<int> strides;
        std::vector<std::string> layer_names;
        int classes_number;
    };

  public:
    YoloV6Converter(BlobToMetaConverter::Initializer initializer, double confidence_threshold, double iou_threshold,
                    YoloV6Converter::Initializer v6Initializer);

    TensorsTable convert(const OutputBlobs &output_blobs) const override;

    static BlobToMetaConverter::Ptr create(BlobToMetaConverter::Initializer initializer, double confidence_threshold);
    static std::string getName();

  protected:
    void parseOutputBlob(const float *blob_data, const std::vector<size_t> &blob_dims, size_t blob_size,
                         std::vector<DetectedObject> &objects, int stride) const;

  private:
    Initializer v6initializer;
};

} // namespace post_processing
#endif // DL_STREAMER_YOLO_V6_H
