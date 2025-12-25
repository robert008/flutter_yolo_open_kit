#include "yolo_detector.hpp"

// Set to 1 to enable debug logging, 0 for production
#define YOLO_DEBUG 0

#ifdef __ANDROID__
#include <opencv2/opencv.hpp>
#include <android/log.h>
#include <onnxruntime/nnapi_provider_factory.h>
#if YOLO_DEBUG
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, "YoloKit", __VA_ARGS__)
#else
#define LOGD(...) do {} while(0)
#endif
#elif defined(__APPLE__)
#include <opencv2/opencv.hpp>
#include <os/log.h>
#include <onnxruntime/coreml_provider_factory.h>
#if YOLO_DEBUG
#define LOGD(...) os_log(OS_LOG_DEFAULT, __VA_ARGS__)
#else
#define LOGD(...) do {} while(0)
#endif
#else
#include <opencv2/opencv.hpp>
#define LOGD(...) do {} while(0)
#endif

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <chrono>

using namespace std::chrono;

// Default COCO class names (80 classes)
static const std::vector<std::string> COCO_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

YoloDetector::YoloDetector()
    : m_initialized(false)
    , m_input_width(640)
    , m_input_height(640)
    , m_num_classes(80)
    , m_model_type(ModelType::YOLOX)  // Default to YOLOX
    , m_class_names(COCO_CLASSES) {
}

YoloDetector::~YoloDetector() {
    release();
}

void YoloDetector::release() {
    m_session.reset();
    m_session_options.reset();
    m_env.reset();
    m_initialized = false;
    LOGD("YOLO detector released");
}

bool YoloDetector::init(const std::string& model_path) {
    try {
        LOGD("Initializing YOLO detector with model: %s", model_path.c_str());

        // Create environment
        m_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "YoloKit");

        // Create session options
        m_session_options = std::make_unique<Ort::SessionOptions>();
        m_session_options->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        m_session_options->SetIntraOpNumThreads(4);
        m_session_options->SetInterOpNumThreads(2);

        // Enable hardware acceleration
#ifdef __ANDROID__
        LOGD("Attempting to enable NNAPI...");
        uint32_t nnapi_flags = NNAPI_FLAG_USE_NONE;
        OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_Nnapi(*m_session_options, nnapi_flags);
        if (status != nullptr) {
            const char* error_msg = Ort::GetApi().GetErrorMessage(status);
            LOGD("NNAPI failed: %s", error_msg);
            Ort::GetApi().ReleaseStatus(status);
        } else {
            LOGD("NNAPI execution provider enabled");
        }
#elif defined(__APPLE__)
        LOGD("Attempting to enable Core ML...");
        uint32_t coreml_flags = 0;
        OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CoreML(*m_session_options, coreml_flags);
        if (status != nullptr) {
            const char* error_msg = Ort::GetApi().GetErrorMessage(status);
            LOGD("Core ML failed: %s", error_msg);
            Ort::GetApi().ReleaseStatus(status);
        } else {
            LOGD("Core ML execution provider enabled");
        }
#endif

        // Create session
        m_session = std::make_unique<Ort::Session>(*m_env, model_path.c_str(), *m_session_options);

        // Get input info
        size_t num_inputs = m_session->GetInputCount();
        LOGD("Model has %zu inputs", num_inputs);

        bool has_scale_factor_input = false;
        for (size_t i = 0; i < num_inputs; i++) {
            auto name = m_session->GetInputNameAllocated(i, m_allocator);
            std::string name_str = name.get();
            m_input_names_str.push_back(name_str);

            auto type_info = m_session->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            auto shape = tensor_info.GetShape();

            // Check if this is a scale_factor input (PP-YOLOE)
            if (name_str.find("scale") != std::string::npos) {
                has_scale_factor_input = true;
                LOGD("Input %zu: %s (scale_factor)", i, name_str.c_str());
            } else {
                // Extract input dimensions from image input (assuming NCHW format)
                if (shape.size() == 4) {
                    // Handle dynamic dimensions (-1)
                    if (shape[2] > 0) m_input_height = static_cast<int>(shape[2]);
                    if (shape[3] > 0) m_input_width = static_cast<int>(shape[3]);
                }
                LOGD("Input %zu: %s, shape: [%lld, %lld, %lld, %lld]",
                     i, name_str.c_str(),
                     shape.size() > 0 ? shape[0] : -1,
                     shape.size() > 1 ? shape[1] : -1,
                     shape.size() > 2 ? shape[2] : -1,
                     shape.size() > 3 ? shape[3] : -1);
            }
        }

        // If model has scale_factor input, it's PP-YOLOE
        if (has_scale_factor_input) {
            m_model_type = ModelType::PPYOLOE;
            m_num_classes = 80;
            LOGD("Detected PP-YOLOE model (has scale_factor input)");
        }

        // Get output info
        size_t num_outputs = m_session->GetOutputCount();
        LOGD("Model has %zu outputs", num_outputs);

        for (size_t i = 0; i < num_outputs; i++) {
            auto name = m_session->GetOutputNameAllocated(i, m_allocator);
            m_output_names_str.push_back(name.get());

            auto type_info = m_session->GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            auto shape = tensor_info.GetShape();

            int64_t dim1 = shape.size() > 1 ? shape[1] : 0;
            int64_t dim2 = shape.size() > 2 ? shape[2] : 0;

            LOGD("Output %zu: %s, shape: [%lld, %lld, %lld]",
                 i, name.get(), shape[0], dim1, dim2);

            // Skip model type detection if already detected as PP-YOLOE
            if (m_model_type == ModelType::PPYOLOE) {
                continue;
            }

            if (shape.size() >= 2) {
                // Detect model type from output shape
                // YOLOX: [1, num_boxes, 85] where 85 = 4 + 1 + 80 (has objectness)
                // YOLOv8: [1, 84, num_boxes] where 84 = 4 + 80 (no objectness)
                // PP-YOLOE: [1, 6, N] or [1, N, 6] (already decoded)
                if (dim1 == 6 || dim2 == 6) {
                    m_model_type = ModelType::PPYOLOE;
                    m_num_classes = 80;
                    LOGD("Detected PP-YOLOE model format");
                } else if (dim2 == 85 || dim1 == 85) {
                    // YOLOX: has objectness (85 = 4 + 1 + 80)
                    m_model_type = ModelType::YOLOX;
                    m_num_classes = 80;
                    LOGD("Detected YOLOX model format (has objectness)");
                } else if (dim1 == 84 || dim2 == 84) {
                    // YOLOv8: no objectness (84 = 4 + 80)
                    m_model_type = ModelType::YOLOV8;
                    m_num_classes = 80;
                    LOGD("Detected YOLOv8 model format (no objectness)");
                } else {
                    // Generic detection
                    int64_t features = (dim1 < dim2) ? dim1 : dim2;
                    if (features > 5) {
                        // Assume YOLOX format if features = 4 + 1 + num_classes
                        m_model_type = ModelType::YOLOX;
                        m_num_classes = static_cast<int>(features - 5);
                    } else if (features > 0) {
                        m_model_type = ModelType::YOLOV8;
                        m_num_classes = static_cast<int>(features - 4);
                    }
                    LOGD("Auto-detected %d classes", m_num_classes);
                }
            }
        }

        m_initialized = true;
        LOGD("YOLO detector initialized successfully (input: %dx%d, classes: %d)",
             m_input_width, m_input_height, m_num_classes);
        return true;

    } catch (const Ort::Exception& e) {
        LOGD("ONNX Runtime error: %s", e.what());
        return false;
    } catch (const std::exception& e) {
        LOGD("Error: %s", e.what());
        return false;
    }
}

void YoloDetector::setClassNames(const std::vector<std::string>& names) {
    m_class_names = names;
    m_num_classes = static_cast<int>(names.size());
}

char* YoloDetector::detectFromPath(
    const char* image_path,
    float conf_threshold,
    float iou_threshold
) {
    if (!m_initialized) {
        return strdup("{\"error\":\"Detector not initialized\",\"code\":\"NOT_INITIALIZED\"}");
    }

    auto start = high_resolution_clock::now();

    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        return strdup("{\"error\":\"Could not load image\",\"code\":\"IMAGE_LOAD_FAILED\"}");
    }

    std::vector<Detection> detections = detect(
        image.data, image.cols, image.rows, conf_threshold, iou_threshold);

    auto end = high_resolution_clock::now();
    long long inference_time = duration_cast<milliseconds>(end - start).count();

    return toJson(detections, inference_time, image.cols, image.rows);
}

char* YoloDetector::detectFromBuffer(
    const uint8_t* image_data,
    int width,
    int height,
    int stride,
    float conf_threshold,
    float iou_threshold
) {
    if (!m_initialized) {
        return strdup("{\"error\":\"Detector not initialized\",\"code\":\"NOT_INITIALIZED\"}");
    }

    auto start = high_resolution_clock::now();

    // Convert BGRA to BGR
    cv::Mat bgra(height, width, CV_8UC4, const_cast<uint8_t*>(image_data), stride);
    cv::Mat bgr;
    cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);

    std::vector<Detection> detections = detect(
        bgr.data, bgr.cols, bgr.rows, conf_threshold, iou_threshold);

    auto end = high_resolution_clock::now();
    long long inference_time = duration_cast<milliseconds>(end - start).count();

    return toJson(detections, inference_time, width, height);
}

char* YoloDetector::detectFromYUV(
    const uint8_t* y_data,
    const uint8_t* u_data,
    const uint8_t* v_data,
    int width,
    int height,
    int y_row_stride,
    int uv_row_stride,
    int uv_pixel_stride,
    int rotation,
    float conf_threshold,
    float iou_threshold
) {
    if (!m_initialized) {
        return strdup("{\"error\":\"Detector not initialized\",\"code\":\"NOT_INITIALIZED\"}");
    }

    auto start = high_resolution_clock::now();

    // Create NV21 format buffer (Y plane + interleaved VU)
    // This is the most common format on Android
    std::vector<uint8_t> nv21_data(width * height * 3 / 2);

    // Copy Y plane (handle stride if different from width)
    if (y_row_stride == width) {
        memcpy(nv21_data.data(), y_data, width * height);
    } else {
        for (int row = 0; row < height; row++) {
            memcpy(nv21_data.data() + row * width, y_data + row * y_row_stride, width);
        }
    }

    // Copy UV data - handle different formats
    uint8_t* uv_dst = nv21_data.data() + width * height;

    if (uv_pixel_stride == 1) {
        // I420/YV12 format: U and V are in separate planes
        // Interleave V and U to create NV21
        for (int row = 0; row < height / 2; row++) {
            for (int col = 0; col < width / 2; col++) {
                int src_offset = row * uv_row_stride + col;
                int dst_offset = row * width + col * 2;
                uv_dst[dst_offset] = v_data[src_offset];      // V first (NV21)
                uv_dst[dst_offset + 1] = u_data[src_offset];  // U second
            }
        }
    } else if (uv_pixel_stride == 2) {
        // NV21 or NV12 format: UV already interleaved
        // Android YUV_420_888 with pixel_stride=2 means U and V point to same buffer
        // U points to first byte, V points to second byte (or vice versa)
        // We need to check which order and convert to NV21 (VU order)

        // Check if it's already NV21 (V comes before U in memory)
        if (v_data < u_data) {
            // Already NV21 format, just copy
            if (uv_row_stride == width) {
                memcpy(uv_dst, v_data, width * height / 2);
            } else {
                for (int row = 0; row < height / 2; row++) {
                    memcpy(uv_dst + row * width, v_data + row * uv_row_stride, width);
                }
            }
        } else {
            // NV12 format (UV order), need to swap to NV21 (VU order)
            for (int row = 0; row < height / 2; row++) {
                for (int col = 0; col < width / 2; col++) {
                    int src_offset = row * uv_row_stride + col * 2;
                    int dst_offset = row * width + col * 2;
                    uv_dst[dst_offset] = v_data[src_offset];      // V (was at +1 in NV12)
                    uv_dst[dst_offset + 1] = u_data[src_offset];  // U (was at +0 in NV12)
                }
            }
        }
    }

    // Convert NV21 to BGR
    cv::Mat nv21(height * 3 / 2, width, CV_8UC1, nv21_data.data());
    cv::Mat bgr;
    cv::cvtColor(nv21, bgr, cv::COLOR_YUV2BGR_NV21);

    // Apply rotation if needed
    if (rotation == 90) {
        cv::rotate(bgr, bgr, cv::ROTATE_90_CLOCKWISE);
    } else if (rotation == 180) {
        cv::rotate(bgr, bgr, cv::ROTATE_180);
    } else if (rotation == 270) {
        cv::rotate(bgr, bgr, cv::ROTATE_90_COUNTERCLOCKWISE);
    }

    // Get final dimensions after rotation
    int final_width = bgr.cols;
    int final_height = bgr.rows;

    std::vector<Detection> detections = detect(
        bgr.data, final_width, final_height, conf_threshold, iou_threshold);

    auto end = high_resolution_clock::now();
    long long inference_time = duration_cast<milliseconds>(end - start).count();

    return toJson(detections, inference_time, final_width, final_height);
}

std::vector<Detection> YoloDetector::detect(
    const uint8_t* bgr_data,
    int width,
    int height,
    float conf_threshold,
    float iou_threshold
) {
    std::vector<Detection> results;

    try {
        // Preprocess
        float scale;
        int pad_x, pad_y;
        std::vector<float> input_tensor = preprocess(bgr_data, width, height, scale, pad_x, pad_y);

        // Prepare input
        std::vector<int64_t> input_shape = {1, 3, m_input_height, m_input_width};
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // Prepare input names and output names
        std::vector<const char*> input_names;
        std::vector<const char*> output_names;
        for (const auto& name : m_input_names_str) {
            input_names.push_back(name.c_str());
        }
        for (const auto& name : m_output_names_str) {
            output_names.push_back(name.c_str());
        }

        std::vector<Ort::Value> input_values;

        // Declare scale_factor outside if block to keep it alive during inference
        std::vector<float> scale_factor_data;
        std::vector<int64_t> scale_shape = {1, 2};

        if (m_model_type == ModelType::PPYOLOE && m_input_names_str.size() >= 2) {
            // PP-YOLOE requires two inputs: image and scale_factor
            // Find which input is which by name
            int image_idx = -1;
            int scale_idx = -1;
            for (size_t i = 0; i < m_input_names_str.size(); i++) {
                if (m_input_names_str[i].find("image") != std::string::npos) {
                    image_idx = static_cast<int>(i);
                } else if (m_input_names_str[i].find("scale") != std::string::npos) {
                    scale_idx = static_cast<int>(i);
                }
            }

            // Default to first=scale, second=image if not found by name
            if (image_idx == -1) image_idx = 1;
            if (scale_idx == -1) scale_idx = 0;

            // Create scale_factor tensor [1, 2] = [scale_y, scale_x]
            // PP-YOLOE expects: input_size / original_size (the resize ratio applied)
            // Model will use this to scale output coordinates back to original space
            float scale_y = static_cast<float>(m_input_height) / static_cast<float>(height);
            float scale_x = static_cast<float>(m_input_width) / static_cast<float>(width);
            scale_factor_data = {scale_y, scale_x};

            LOGD("PP-YOLOE scale_factor: [%.4f, %.4f] (input/orig, orig: %dx%d, input: %dx%d)",
                 scale_y, scale_x, width, height, m_input_width, m_input_height);

            // Create input tensors in correct order
            std::vector<Ort::Value> temp_values;
            for (size_t i = 0; i < m_input_names_str.size(); i++) {
                if (static_cast<int>(i) == image_idx) {
                    temp_values.push_back(Ort::Value::CreateTensor<float>(
                        memory_info, input_tensor.data(), input_tensor.size(),
                        input_shape.data(), input_shape.size()));
                } else if (static_cast<int>(i) == scale_idx) {
                    temp_values.push_back(Ort::Value::CreateTensor<float>(
                        memory_info, scale_factor_data.data(), scale_factor_data.size(),
                        scale_shape.data(), scale_shape.size()));
                }
            }
            input_values = std::move(temp_values);

            LOGD("PP-YOLOE: Using %zu inputs (image_idx=%d, scale_idx=%d)",
                 input_values.size(), image_idx, scale_idx);
        } else {
            // Single input model (YOLOX, YOLOv8)
            input_values.push_back(Ort::Value::CreateTensor<float>(
                memory_info, input_tensor.data(), input_tensor.size(),
                input_shape.data(), input_shape.size()));
        }

        // Run inference
        auto outputs = m_session->Run(
            Ort::RunOptions{nullptr},
            input_names.data(), input_values.data(), input_values.size(),
            output_names.data(), output_names.size());

        // Get output tensor info
        auto output_info = outputs[0].GetTensorTypeAndShapeInfo();
        auto output_shape = output_info.GetShape();
        size_t output_count = output_info.GetElementCount();
        float* output_data = outputs[0].GetTensorMutableData<float>();

        LOGD("Output tensor: shape dims=%zu, element_count=%zu", output_shape.size(), output_count);

        // Postprocess
        results = postprocess(output_data, output_shape, output_count, width, height, scale, pad_x, pad_y, conf_threshold, iou_threshold);

    } catch (const Ort::Exception& e) {
        LOGD("ONNX Runtime error: %s", e.what());
    } catch (const cv::Exception& e) {
        LOGD("OpenCV error: %s", e.what());
    } catch (const std::exception& e) {
        LOGD("Error: %s", e.what());
    }

    return results;
}

std::vector<float> YoloDetector::preprocess(
    const uint8_t* bgr_data,
    int width,
    int height,
    float& scale,
    int& pad_x,
    int& pad_y
) {
    // Create cv::Mat from BGR data
    cv::Mat image(height, width, CV_8UC3, const_cast<uint8_t*>(bgr_data));

    cv::Mat resized_final;

    if (m_model_type == ModelType::PPYOLOE) {
        // PP-YOLOE: Direct resize to input size (NO letterbox)
        cv::resize(image, resized_final, cv::Size(m_input_width, m_input_height), 0, 0, cv::INTER_LINEAR);
        scale = 1.0f;  // Not used for PP-YOLOE
        pad_x = 0;
        pad_y = 0;
        LOGD("PP-YOLOE preprocess: direct resize %dx%d -> %dx%d", width, height, m_input_width, m_input_height);
    } else {
        // YOLOX/YOLOv8: Letterbox resize (keep aspect ratio with padding)
        float scale_x = static_cast<float>(m_input_width) / width;
        float scale_y = static_cast<float>(m_input_height) / height;
        scale = std::min(scale_x, scale_y);

        int new_width = static_cast<int>(width * scale);
        int new_height = static_cast<int>(height * scale);

        pad_x = (m_input_width - new_width) / 2;
        pad_y = (m_input_height - new_height) / 2;

        // Resize
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR);

        // Create padded image (gray padding: 114)
        resized_final = cv::Mat(m_input_height, m_input_width, CV_8UC3, cv::Scalar(114, 114, 114));
        resized.copyTo(resized_final(cv::Rect(pad_x, pad_y, new_width, new_height)));
    }

    // Convert to CHW format
    std::vector<float> tensor(3 * m_input_height * m_input_width);
    int channel_size = m_input_height * m_input_width;

    if (m_model_type == ModelType::YOLOX) {
        // YOLOX: BGR format, NO normalization (0-255 range)
        for (int y = 0; y < m_input_height; y++) {
            for (int x = 0; x < m_input_width; x++) {
                cv::Vec3b pixel = resized_final.at<cv::Vec3b>(y, x);
                int idx = y * m_input_width + x;
                tensor[0 * channel_size + idx] = static_cast<float>(pixel[0]);  // B
                tensor[1 * channel_size + idx] = static_cast<float>(pixel[1]);  // G
                tensor[2 * channel_size + idx] = static_cast<float>(pixel[2]);  // R
            }
        }
    } else {
        // YOLOv8/PP-YOLOE: RGB format, normalized to [0, 1]
        cv::Mat rgb;
        cv::cvtColor(resized_final, rgb, cv::COLOR_BGR2RGB);

        for (int y = 0; y < m_input_height; y++) {
            for (int x = 0; x < m_input_width; x++) {
                cv::Vec3b pixel = rgb.at<cv::Vec3b>(y, x);
                int idx = y * m_input_width + x;
                tensor[0 * channel_size + idx] = pixel[0] / 255.0f;  // R
                tensor[1 * channel_size + idx] = pixel[1] / 255.0f;  // G
                tensor[2 * channel_size + idx] = pixel[2] / 255.0f;  // B
            }
        }
    }

    return tensor;
}

std::vector<Detection> YoloDetector::postprocess(
    const float* output,
    const std::vector<int64_t>& output_shape,
    size_t output_count,
    int original_width,
    int original_height,
    float scale,
    int pad_x,
    int pad_y,
    float conf_threshold,
    float iou_threshold
) {
    std::vector<Detection> detections;

    // Handle different output shape dimensions
    int64_t dim1 = 0, dim2 = 0;

    if (output_shape.size() == 2) {
        // 2D output: could be [batch, total_values] or [num_detections, 6]
        dim1 = output_shape[0];
        dim2 = output_shape[1];
        LOGD("Postprocess: model_type=%d, shape=[%lld, %lld] (2D), count=%zu",
             static_cast<int>(m_model_type), dim1, dim2, output_count);
    } else if (output_shape.size() >= 3) {
        dim1 = output_shape[1];
        dim2 = output_shape[2];
        LOGD("Postprocess: model_type=%d, shape=[1, %lld, %lld], count=%zu",
             static_cast<int>(m_model_type), dim1, dim2, output_count);
    } else if (output_shape.size() == 1) {
        // 1D output - use element count
        dim1 = static_cast<int64_t>(output_count);
        dim2 = 0;
        LOGD("Postprocess: model_type=%d, shape=[%lld] (1D), count=%zu",
             static_cast<int>(m_model_type), dim1, output_count);
    } else {
        LOGD("Unexpected output shape size: %zu", output_shape.size());
        return detections;
    }

    if (m_model_type == ModelType::PPYOLOE) {
        // PP-YOLOE output can be:
        // [N, 6] = 2D format (dim1=N, dim2=6)
        // [1, N, 6] = 3D format (dim1=N, dim2=6)
        // [1, 6, N] = transposed 3D format (dim1=6, dim2=N)
        // Coordinates are in letterbox scale, need to convert back

        bool transposed = false;
        int num_detections = 0;

        if (output_shape.size() == 2) {
            // 2D output: [N, 6] format
            if (dim2 == 6) {
                num_detections = static_cast<int>(dim1);
                transposed = false;
            } else if (dim1 == 6) {
                num_detections = static_cast<int>(dim2);
                transposed = true;
            } else {
                // Use element count
                num_detections = static_cast<int>(output_count / 6);
            }
        } else if (output_shape.size() >= 3) {
            if (dim1 == 6 && dim2 > 0) {
                // [1, 6, N] format
                transposed = true;
                num_detections = static_cast<int>(dim2);
            } else if (dim2 == 6 && dim1 > 0) {
                // [1, N, 6] format
                transposed = false;
                num_detections = static_cast<int>(dim1);
            } else if (dim1 == 6) {
                // [1, 6, N] with dynamic N - use element count
                transposed = true;
                num_detections = static_cast<int>(output_count / 6);
            }
        } else {
            // Fallback: use element count
            num_detections = static_cast<int>(output_count / 6);
        }

        LOGD("PP-YOLOE: processing %d detections, transposed=%d, dim1=%lld, dim2=%lld, count=%zu",
             num_detections, transposed, dim1, dim2, output_count);


        // If num_detections is 0, no objects detected
        if (num_detections <= 0) {
            LOGD("PP-YOLOE: no detections");
            return detections;
        }

        for (int i = 0; i < num_detections; i++) {
            int class_id;
            float score, x1, y1, x2, y2;

            // PP-YOLOE output [N, 6] format - always row-major
            // Each row: [class_id, score, x1, y1, x2, y2]
            const float* det_data = output + i * 6;
            class_id = static_cast<int>(det_data[0]);
            score = det_data[1];
            x1 = det_data[2];
            y1 = det_data[3];
            x2 = det_data[4];
            y2 = det_data[5];

            if (score < conf_threshold) continue;
            if (class_id < 0) continue;

            // PP-YOLOE with scale_factor=[input/orig] outputs coordinates already in original image space
            // The model internally uses scale_factor to convert its predictions
            // No additional scaling needed here

            // Clamp to image bounds
            x1 = std::max(0.0f, std::min(x1, static_cast<float>(original_width)));
            y1 = std::max(0.0f, std::min(y1, static_cast<float>(original_height)));
            x2 = std::max(0.0f, std::min(x2, static_cast<float>(original_width)));
            y2 = std::max(0.0f, std::min(y2, static_cast<float>(original_height)));

            Detection det;
            det.x1 = x1;
            det.y1 = y1;
            det.x2 = x2;
            det.y2 = y2;
            det.confidence = score;
            det.class_id = class_id;
            det.class_name = (class_id < static_cast<int>(m_class_names.size()))
                             ? m_class_names[class_id]
                             : "class_" + std::to_string(class_id);

            detections.push_back(det);
        }

        // PP-YOLOE already has NMS applied
        LOGD("PP-YOLOE: detected %zu objects", detections.size());
        return detections;
    }

    if (m_model_type == ModelType::YOLOX) {
        // YOLOX output: [1, num_boxes, 85] = [cx, cy, w, h, objectness, class_scores...]
        // Need to decode with grid and stride
        // 8400 = 80*80 + 40*40 + 20*20 (strides: 8, 16, 32)
        int num_boxes = static_cast<int>(dim1);
        int features = static_cast<int>(dim2);  // 85 = 4 + 1 + 80

        LOGD("YOLOX: processing %d boxes with %d features", num_boxes, features);

        // Generate grids and strides for decoding
        std::vector<float> grids_x, grids_y, strides_vec;
        int strides[] = {8, 16, 32};

        for (int stride : strides) {
            int grid_size = m_input_width / stride;
            for (int gy = 0; gy < grid_size; gy++) {
                for (int gx = 0; gx < grid_size; gx++) {
                    grids_x.push_back(static_cast<float>(gx));
                    grids_y.push_back(static_cast<float>(gy));
                    strides_vec.push_back(static_cast<float>(stride));
                }
            }
        }

        for (int i = 0; i < num_boxes; i++) {
            const float* box_data = output + i * features;

            float objectness = box_data[4];

            // Early filter by objectness
            if (objectness < conf_threshold) continue;

            // Find max class score
            float max_class_score = 0.0f;
            int max_class = 0;
            const float* class_scores = box_data + 5;

            for (int c = 0; c < m_num_classes; c++) {
                if (class_scores[c] > max_class_score) {
                    max_class_score = class_scores[c];
                    max_class = c;
                }
            }

            // Final confidence = objectness * class_score
            float confidence = objectness * max_class_score;
            if (confidence < conf_threshold) continue;

            // Decode coordinates using grid and stride
            float grid_x = grids_x[i];
            float grid_y = grids_y[i];
            float stride = strides_vec[i];

            float cx = (box_data[0] + grid_x) * stride;
            float cy = (box_data[1] + grid_y) * stride;
            float w = std::exp(box_data[2]) * stride;
            float h = std::exp(box_data[3]) * stride;

            // Convert from letterbox coordinates to original image coordinates
            float x1 = (cx - w / 2.0f - pad_x) / scale;
            float y1 = (cy - h / 2.0f - pad_y) / scale;
            float x2 = (cx + w / 2.0f - pad_x) / scale;
            float y2 = (cy + h / 2.0f - pad_y) / scale;

            // Clamp to image bounds
            x1 = std::max(0.0f, std::min(x1, static_cast<float>(original_width)));
            y1 = std::max(0.0f, std::min(y1, static_cast<float>(original_height)));
            x2 = std::max(0.0f, std::min(x2, static_cast<float>(original_width)));
            y2 = std::max(0.0f, std::min(y2, static_cast<float>(original_height)));

            Detection det;
            det.x1 = x1;
            det.y1 = y1;
            det.x2 = x2;
            det.y2 = y2;
            det.confidence = confidence;
            det.class_id = max_class;
            det.class_name = (max_class < static_cast<int>(m_class_names.size()))
                             ? m_class_names[max_class]
                             : "class_" + std::to_string(max_class);

            detections.push_back(det);
        }
    } else {
        // YOLOv8/v11 output: [1, 84, num_boxes] or [1, num_boxes, 84]
        // No objectness, direct class scores
        bool transposed = (dim1 > dim2);  // [1, num_boxes, features]
        int num_boxes = transposed ? static_cast<int>(dim1) : static_cast<int>(dim2);
        int features = transposed ? static_cast<int>(dim2) : static_cast<int>(dim1);
        int num_classes = features - 4;

        LOGD("YOLOv8: processing %d boxes, transposed=%d", num_boxes, transposed);

        for (int i = 0; i < num_boxes; i++) {
            float cx, cy, w, h;

            if (transposed) {
                const float* box_data = output + i * features;
                cx = box_data[0];
                cy = box_data[1];
                w = box_data[2];
                h = box_data[3];
            } else {
                cx = output[0 * num_boxes + i];
                cy = output[1 * num_boxes + i];
                w = output[2 * num_boxes + i];
                h = output[3 * num_boxes + i];
            }

            // Find max class score
            float max_score = 0.0f;
            int max_class = 0;

            for (int c = 0; c < num_classes; c++) {
                float score;
                if (transposed) {
                    score = output[i * features + 4 + c];
                } else {
                    score = output[(4 + c) * num_boxes + i];
                }

                if (score > max_score) {
                    max_score = score;
                    max_class = c;
                }
            }

            if (max_score < conf_threshold) continue;

            // Convert from letterbox coordinates to original image coordinates
            float x1 = (cx - w / 2.0f - pad_x) / scale;
            float y1 = (cy - h / 2.0f - pad_y) / scale;
            float x2 = (cx + w / 2.0f - pad_x) / scale;
            float y2 = (cy + h / 2.0f - pad_y) / scale;

            // Clamp to image bounds
            x1 = std::max(0.0f, std::min(x1, static_cast<float>(original_width)));
            y1 = std::max(0.0f, std::min(y1, static_cast<float>(original_height)));
            x2 = std::max(0.0f, std::min(x2, static_cast<float>(original_width)));
            y2 = std::max(0.0f, std::min(y2, static_cast<float>(original_height)));

            Detection det;
            det.x1 = x1;
            det.y1 = y1;
            det.x2 = x2;
            det.y2 = y2;
            det.confidence = max_score;
            det.class_id = max_class;
            det.class_name = (max_class < static_cast<int>(m_class_names.size()))
                             ? m_class_names[max_class]
                             : "class_" + std::to_string(max_class);

            detections.push_back(det);
        }
    }

    // Apply NMS
    detections = nms(detections, iou_threshold);

    LOGD("Detected %zu objects after NMS", detections.size());
    return detections;
}

float YoloDetector::iou(const Detection& a, const Detection& b) {
    float x1 = std::max(a.x1, b.x1);
    float y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2);
    float y2 = std::min(a.y2, b.y2);

    float inter_area = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float a_area = (a.x2 - a.x1) * (a.y2 - a.y1);
    float b_area = (b.x2 - b.x1) * (b.y2 - b.y1);
    float union_area = a_area + b_area - inter_area;

    return (union_area > 0) ? inter_area / union_area : 0.0f;
}

std::vector<Detection> YoloDetector::nms(
    std::vector<Detection>& detections,
    float iou_threshold
) {
    // Sort by confidence (descending)
    std::sort(detections.begin(), detections.end(),
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });

    std::vector<Detection> result;
    std::vector<bool> suppressed(detections.size(), false);

    for (size_t i = 0; i < detections.size(); i++) {
        if (suppressed[i]) continue;

        result.push_back(detections[i]);

        for (size_t j = i + 1; j < detections.size(); j++) {
            if (suppressed[j]) continue;

            // Only suppress if same class
            if (detections[i].class_id == detections[j].class_id) {
                if (iou(detections[i], detections[j]) > iou_threshold) {
                    suppressed[j] = true;
                }
            }
        }
    }

    return result;
}

char* YoloDetector::toJson(
    const std::vector<Detection>& detections,
    long long inference_time_ms,
    int image_width,
    int image_height
) {
    std::ostringstream oss;
    oss << "{\"detections\":[";

    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& d = detections[i];
        if (i > 0) oss << ",";
        oss << "{"
            << "\"class_id\":" << d.class_id << ","
            << "\"class_name\":\"" << d.class_name << "\","
            << "\"confidence\":" << std::fixed << std::setprecision(4) << d.confidence << ","
            << "\"x1\":" << std::setprecision(2) << d.x1 << ","
            << "\"y1\":" << d.y1 << ","
            << "\"x2\":" << d.x2 << ","
            << "\"y2\":" << d.y2
            << "}";
    }

    oss << "],"
        << "\"count\":" << detections.size() << ","
        << "\"inference_time_ms\":" << inference_time_ms << ","
        << "\"image_width\":" << image_width << ","
        << "\"image_height\":" << image_height
        << "}";

    std::string json = oss.str();
    char* result = static_cast<char*>(malloc(json.size() + 1));
    strcpy(result, json.c_str());
    return result;
}
