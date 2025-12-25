import 'dart:convert';
import 'dart:ffi';
import 'dart:io';

import 'package:ffi/ffi.dart';

import 'flutter_yolo_open_kit_bindings_generated.dart';

/// Detection result from YOLO model
class YoloDetection {
  final int classId;
  final String className;
  final double confidence;
  final double x1;
  final double y1;
  final double x2;
  final double y2;

  YoloDetection({
    required this.classId,
    required this.className,
    required this.confidence,
    required this.x1,
    required this.y1,
    required this.x2,
    required this.y2,
  });

  factory YoloDetection.fromJson(Map<String, dynamic> json) {
    return YoloDetection(
      classId: json['class_id'] as int,
      className: json['class_name'] as String,
      confidence: (json['confidence'] as num).toDouble(),
      x1: (json['x1'] as num).toDouble(),
      y1: (json['y1'] as num).toDouble(),
      x2: (json['x2'] as num).toDouble(),
      y2: (json['y2'] as num).toDouble(),
    );
  }

  /// Get bounding box width
  double get width => x2 - x1;

  /// Get bounding box height
  double get height => y2 - y1;

  /// Get bounding box center X
  double get centerX => (x1 + x2) / 2;

  /// Get bounding box center Y
  double get centerY => (y1 + y2) / 2;

  @override
  String toString() {
    return 'YoloDetection($className: ${(confidence * 100).toStringAsFixed(1)}% at [$x1, $y1, $x2, $y2])';
  }
}

/// Result from YOLO detection
class YoloResult {
  final List<YoloDetection> detections;
  final int count;
  final int inferenceTimeMs;
  final int imageWidth;
  final int imageHeight;
  final String? error;
  final String? errorCode;

  YoloResult({
    required this.detections,
    required this.count,
    required this.inferenceTimeMs,
    required this.imageWidth,
    required this.imageHeight,
    this.error,
    this.errorCode,
  });

  factory YoloResult.fromJson(Map<String, dynamic> json) {
    if (json.containsKey('error')) {
      return YoloResult(
        detections: [],
        count: 0,
        inferenceTimeMs: 0,
        imageWidth: 0,
        imageHeight: 0,
        error: json['error'] as String?,
        errorCode: json['code'] as String?,
      );
    }

    final detectionsList = (json['detections'] as List)
        .map((d) => YoloDetection.fromJson(d as Map<String, dynamic>))
        .toList();

    return YoloResult(
      detections: detectionsList,
      count: json['count'] as int,
      inferenceTimeMs: json['inference_time_ms'] as int,
      imageWidth: json['image_width'] as int,
      imageHeight: json['image_height'] as int,
    );
  }

  bool get hasError => error != null;

  @override
  String toString() {
    if (hasError) {
      return 'YoloResult(error: $error)';
    }
    return 'YoloResult($count detections in ${inferenceTimeMs}ms)';
  }
}

/// Flutter YOLO Open Kit - YOLO object detection plugin
class FlutterYoloOpenKit {
  static FlutterYoloOpenKit? _instance;
  late final FlutterYoloOpenKitBindings _bindings;
  bool _initialized = false;

  FlutterYoloOpenKit._() {
    _bindings = FlutterYoloOpenKitBindings(_dylib);
  }

  /// Get singleton instance
  static FlutterYoloOpenKit get instance {
    _instance ??= FlutterYoloOpenKit._();
    return _instance!;
  }

  /// Initialize YOLO detector with model path
  ///
  /// [modelPath] - Path to ONNX model file (YOLOv8/v11)
  /// Returns true on success
  bool init(String modelPath) {
    final pathPtr = modelPath.toNativeUtf8();
    try {
      final result = _bindings.yolo_init(pathPtr.cast());
      _initialized = result == 1;
      return _initialized;
    } finally {
      malloc.free(pathPtr);
    }
  }

  /// Run detection on image file
  ///
  /// [imagePath] - Path to image file
  /// [confThreshold] - Confidence threshold (0-1), default 0.25
  /// [iouThreshold] - IoU threshold for NMS (0-1), default 0.45
  YoloResult detectFromPath(
    String imagePath, {
    double confThreshold = 0.25,
    double iouThreshold = 0.45,
  }) {
    final pathPtr = imagePath.toNativeUtf8();
    Pointer<Char>? resultPtr;

    try {
      resultPtr = _bindings.yolo_detect_path(
        pathPtr.cast(),
        confThreshold,
        iouThreshold,
      );

      if (resultPtr == nullptr) {
        return YoloResult(
          detections: [],
          count: 0,
          inferenceTimeMs: 0,
          imageWidth: 0,
          imageHeight: 0,
          error: 'Detection failed',
          errorCode: 'NULL_RESULT',
        );
      }

      final jsonStr = resultPtr.cast<Utf8>().toDartString();
      final json = jsonDecode(jsonStr) as Map<String, dynamic>;
      return YoloResult.fromJson(json);
    } finally {
      malloc.free(pathPtr);
      if (resultPtr != null && resultPtr != nullptr) {
        _bindings.free_string(resultPtr);
      }
    }
  }

  /// Run detection on image buffer (BGRA format)
  ///
  /// [imageData] - Pointer to BGRA image data
  /// [width] - Image width
  /// [height] - Image height
  /// [stride] - Bytes per row
  /// [confThreshold] - Confidence threshold (0-1), default 0.25
  /// [iouThreshold] - IoU threshold for NMS (0-1), default 0.45
  YoloResult detectFromBuffer(
    Pointer<Uint8> imageData,
    int width,
    int height,
    int stride, {
    double confThreshold = 0.25,
    double iouThreshold = 0.45,
  }) {
    Pointer<Char>? resultPtr;

    try {
      resultPtr = _bindings.yolo_detect_buffer(
        imageData,
        width,
        height,
        stride,
        confThreshold,
        iouThreshold,
      );

      if (resultPtr == nullptr) {
        return YoloResult(
          detections: [],
          count: 0,
          inferenceTimeMs: 0,
          imageWidth: width,
          imageHeight: height,
          error: 'Detection failed',
          errorCode: 'NULL_RESULT',
        );
      }

      final jsonStr = resultPtr.cast<Utf8>().toDartString();
      final json = jsonDecode(jsonStr) as Map<String, dynamic>;
      return YoloResult.fromJson(json);
    } finally {
      if (resultPtr != null && resultPtr != nullptr) {
        _bindings.free_string(resultPtr);
      }
    }
  }

  /// Run detection on YUV420 buffer (Android camera format)
  ///
  /// [yData] - Pointer to Y plane data
  /// [uData] - Pointer to U plane data
  /// [vData] - Pointer to V plane data
  /// [width] - Image width
  /// [height] - Image height
  /// [yRowStride] - Bytes per row for Y plane
  /// [uvRowStride] - Bytes per row for UV planes
  /// [uvPixelStride] - Pixel stride for UV planes (1 for I420, 2 for NV21/NV12)
  /// [rotation] - Rotation in degrees (0, 90, 180, 270), default 0
  /// [confThreshold] - Confidence threshold (0-1), default 0.25
  /// [iouThreshold] - IoU threshold for NMS (0-1), default 0.45
  YoloResult detectFromYUV(
    Pointer<Uint8> yData,
    Pointer<Uint8> uData,
    Pointer<Uint8> vData,
    int width,
    int height,
    int yRowStride,
    int uvRowStride,
    int uvPixelStride, {
    int rotation = 0,
    double confThreshold = 0.25,
    double iouThreshold = 0.45,
  }) {
    Pointer<Char>? resultPtr;

    try {
      resultPtr = _bindings.yolo_detect_yuv(
        yData,
        uData,
        vData,
        width,
        height,
        yRowStride,
        uvRowStride,
        uvPixelStride,
        rotation,
        confThreshold,
        iouThreshold,
      );

      if (resultPtr == nullptr) {
        return YoloResult(
          detections: [],
          count: 0,
          inferenceTimeMs: 0,
          imageWidth: width,
          imageHeight: height,
          error: 'Detection failed',
          errorCode: 'NULL_RESULT',
        );
      }

      final jsonStr = resultPtr.cast<Utf8>().toDartString();
      final json = jsonDecode(jsonStr) as Map<String, dynamic>;
      return YoloResult.fromJson(json);
    } finally {
      if (resultPtr != null && resultPtr != nullptr) {
        _bindings.free_string(resultPtr);
      }
    }
  }

  /// Set custom class names for the model
  ///
  /// [classNames] - List of class names
  void setClassNames(List<String> classNames) {
    final jsonStr = jsonEncode(classNames);
    final ptr = jsonStr.toNativeUtf8();
    try {
      _bindings.yolo_set_classes(ptr.cast());
    } finally {
      malloc.free(ptr);
    }
  }

  /// Release detector resources
  void release() {
    _bindings.yolo_release();
    _initialized = false;
  }

  /// Check if detector is initialized
  bool get isInitialized {
    return _bindings.yolo_is_initialized() == 1;
  }

  /// Get library version
  String get version {
    final ptr = _bindings.yolo_get_version();
    return ptr.cast<Utf8>().toDartString();
  }
}

const String _libName = 'flutter_yolo_open_kit';

/// The dynamic library in which the symbols for [FlutterYoloOpenKitBindings] can be found.
final DynamicLibrary _dylib = () {
  if (Platform.isIOS) {
    // iOS uses static library linked into the app
    return DynamicLibrary.process();
  }
  if (Platform.isMacOS) {
    return DynamicLibrary.open('$_libName.framework/$_libName');
  }
  if (Platform.isAndroid || Platform.isLinux) {
    return DynamicLibrary.open('lib$_libName.so');
  }
  if (Platform.isWindows) {
    return DynamicLibrary.open('$_libName.dll');
  }
  throw UnsupportedError('Unknown platform: ${Platform.operatingSystem}');
}();
