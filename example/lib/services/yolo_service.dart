import 'dart:async';
import 'dart:ffi';
import 'dart:io';
import 'dart:isolate';

import 'package:ffi/ffi.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_yolo_open_kit/flutter_yolo_open_kit.dart';

/// Message types for isolate communication
enum _MessageType { init, detectPath, detectBuffer, detectYuv, release }

/// Request message to AI isolate
class _IsolateRequest {
  final _MessageType type;
  final Map<String, dynamic>? params;
  final SendPort replyPort;

  _IsolateRequest(this.type, this.params, this.replyPort);
}

/// Singleton service for YOLO detection running in a dedicated isolate
class YoloService {
  static final YoloService instance = YoloService._();

  YoloService._();

  Isolate? _isolate;
  SendPort? _sendPort;
  bool _isInitialized = false;
  bool _isInitializing = false;
  Completer<void>? _initCompleter;

  bool get isInitialized => _isInitialized;

  /// Initialize the service and spawn the AI isolate
  Future<int> init(String modelPath, {String? warmupImagePath}) async {
    if (_isInitialized) {
      return 0; // Already initialized
    }

    if (_isInitializing) {
      await _initCompleter?.future;
      return 0;
    }

    _isInitializing = true;
    _initCompleter = Completer<void>();

    try {
      // Create receive port for handshake
      final receivePort = ReceivePort();

      // Spawn isolate
      _isolate = await Isolate.spawn(
        _isolateEntryPoint,
        receivePort.sendPort,
      );

      // Wait for isolate to send its SendPort
      _sendPort = await receivePort.first as SendPort;

      // Initialize model in isolate
      final responsePort = ReceivePort();
      _sendPort!.send(_IsolateRequest(
        _MessageType.init,
        {
          'modelPath': modelPath,
          'warmupPath': warmupImagePath,
        },
        responsePort.sendPort,
      ));

      final result = await responsePort.first;
      responsePort.close();

      if (result is int) {
        _isInitialized = true;
        _initCompleter?.complete();
        return result; // warmup time in ms
      } else if (result is String) {
        throw Exception(result);
      }

      _isInitialized = true;
      _initCompleter?.complete();
      return 0;
    } catch (e) {
      _initCompleter?.completeError(e);
      rethrow;
    } finally {
      _isInitializing = false;
    }
  }

  /// Detect objects from image file path
  Future<YoloResult> detectFromPath(
    String imagePath, {
    double confThreshold = 0.25,
    double iouThreshold = 0.45,
  }) async {
    if (!_isInitialized || _sendPort == null) {
      return YoloResult(
        detections: [],
        count: 0,
        inferenceTimeMs: 0,
        imageWidth: 0,
        imageHeight: 0,
        error: 'Service not initialized',
      );
    }

    final responsePort = ReceivePort();
    _sendPort!.send(_IsolateRequest(
      _MessageType.detectPath,
      {
        'imagePath': imagePath,
        'confThreshold': confThreshold,
        'iouThreshold': iouThreshold,
      },
      responsePort.sendPort,
    ));

    final result = await responsePort.first;
    responsePort.close();

    if (result is Map<String, dynamic>) {
      return YoloResult.fromJson(result);
    } else if (result is String) {
      return YoloResult(
        detections: [],
        count: 0,
        inferenceTimeMs: 0,
        imageWidth: 0,
        imageHeight: 0,
        error: result,
      );
    }

    return YoloResult(
      detections: [],
      count: 0,
      inferenceTimeMs: 0,
      imageWidth: 0,
      imageHeight: 0,
      error: 'Unknown error',
    );
  }

  /// Detect objects from BGRA buffer (iOS camera format)
  Future<YoloResult> detectFromBuffer({
    required Uint8List imageData,
    required int width,
    required int height,
    required int stride,
    double confThreshold = 0.25,
    double iouThreshold = 0.45,
  }) async {
    if (!_isInitialized || _sendPort == null) {
      return YoloResult(
        detections: [],
        count: 0,
        inferenceTimeMs: 0,
        imageWidth: width,
        imageHeight: height,
        error: 'Service not initialized',
      );
    }

    final responsePort = ReceivePort();
    _sendPort!.send(_IsolateRequest(
      _MessageType.detectBuffer,
      {
        'imageData': imageData,
        'width': width,
        'height': height,
        'stride': stride,
        'confThreshold': confThreshold,
        'iouThreshold': iouThreshold,
      },
      responsePort.sendPort,
    ));

    final result = await responsePort.first;
    responsePort.close();

    if (result is Map<String, dynamic>) {
      return YoloResult.fromJson(result);
    } else if (result is String) {
      return YoloResult(
        detections: [],
        count: 0,
        inferenceTimeMs: 0,
        imageWidth: width,
        imageHeight: height,
        error: result,
      );
    }

    return YoloResult(
      detections: [],
      count: 0,
      inferenceTimeMs: 0,
      imageWidth: width,
      imageHeight: height,
      error: 'Unknown error',
    );
  }

  /// Detect objects from YUV camera frame (Android)
  Future<YoloResult> detectFromYuv({
    required Uint8List yData,
    required Uint8List uData,
    required Uint8List vData,
    required int width,
    required int height,
    required int yRowStride,
    required int uvRowStride,
    required int uvPixelStride,
    int rotation = 0,
    double confThreshold = 0.25,
    double iouThreshold = 0.45,
  }) async {
    if (!_isInitialized || _sendPort == null) {
      return YoloResult(
        detections: [],
        count: 0,
        inferenceTimeMs: 0,
        imageWidth: width,
        imageHeight: height,
        error: 'Service not initialized',
      );
    }

    final responsePort = ReceivePort();
    _sendPort!.send(_IsolateRequest(
      _MessageType.detectYuv,
      {
        'yData': yData,
        'uData': uData,
        'vData': vData,
        'width': width,
        'height': height,
        'yRowStride': yRowStride,
        'uvRowStride': uvRowStride,
        'uvPixelStride': uvPixelStride,
        'rotation': rotation,
        'confThreshold': confThreshold,
        'iouThreshold': iouThreshold,
      },
      responsePort.sendPort,
    ));

    final result = await responsePort.first;
    responsePort.close();

    if (result is Map<String, dynamic>) {
      return YoloResult.fromJson(result);
    } else if (result is String) {
      return YoloResult(
        detections: [],
        count: 0,
        inferenceTimeMs: 0,
        imageWidth: width,
        imageHeight: height,
        error: result,
      );
    }

    return YoloResult(
      detections: [],
      count: 0,
      inferenceTimeMs: 0,
      imageWidth: width,
      imageHeight: height,
      error: 'Unknown error',
    );
  }

  /// Release resources and kill isolate
  void dispose() {
    if (_sendPort != null) {
      final responsePort = ReceivePort();
      _sendPort!.send(_IsolateRequest(
        _MessageType.release,
        null,
        responsePort.sendPort,
      ));
      // Don't wait for response
    }

    _isolate?.kill(priority: Isolate.immediate);
    _isolate = null;
    _sendPort = null;
    _isInitialized = false;
  }
}

/// Isolate entry point - runs in separate isolate
void _isolateEntryPoint(SendPort mainSendPort) {
  final receivePort = ReceivePort();
  mainSendPort.send(receivePort.sendPort);

  final yolo = FlutterYoloOpenKit.instance;
  bool modelLoaded = false;

  receivePort.listen((message) {
    if (message is! _IsolateRequest) return;

    switch (message.type) {
      case _MessageType.init:
        try {
          final modelPath = message.params!['modelPath'] as String;
          final warmupPath = message.params!['warmupPath'] as String?;

          final success = yolo.init(modelPath);
          if (!success) {
            message.replyPort.send('Failed to initialize model');
            return;
          }
          modelLoaded = true;

          // Warmup if path provided
          int warmupTime = 0;
          if (warmupPath != null && File(warmupPath).existsSync()) {
            final start = DateTime.now();
            yolo.detectFromPath(warmupPath);
            warmupTime = DateTime.now().difference(start).inMilliseconds;
          }

          message.replyPort.send(warmupTime);
        } catch (e) {
          message.replyPort.send('Init error: $e');
        }
        break;

      case _MessageType.detectPath:
        if (!modelLoaded) {
          message.replyPort.send('Model not initialized');
          return;
        }

        try {
          final imagePath = message.params!['imagePath'] as String;
          final confThreshold = message.params!['confThreshold'] as double;
          final iouThreshold = message.params!['iouThreshold'] as double;

          final result = yolo.detectFromPath(
            imagePath,
            confThreshold: confThreshold,
            iouThreshold: iouThreshold,
          );

          message.replyPort.send(_yoloResultToJson(result));
        } catch (e) {
          message.replyPort.send('Detection error: $e');
        }
        break;

      case _MessageType.detectBuffer:
        if (!modelLoaded) {
          message.replyPort.send('Model not initialized');
          return;
        }

        try {
          final imageData = message.params!['imageData'] as Uint8List;
          final width = message.params!['width'] as int;
          final height = message.params!['height'] as int;
          final stride = message.params!['stride'] as int;
          final confThreshold = message.params!['confThreshold'] as double;
          final iouThreshold = message.params!['iouThreshold'] as double;

          // Allocate native buffer
          final buffer = malloc.allocate<Uint8>(imageData.length);

          try {
            buffer.asTypedList(imageData.length).setAll(0, imageData);

            final result = yolo.detectFromBuffer(
              buffer,
              width,
              height,
              stride,
              confThreshold: confThreshold,
              iouThreshold: iouThreshold,
            );

            message.replyPort.send(_yoloResultToJson(result));
          } finally {
            malloc.free(buffer);
          }
        } catch (e) {
          message.replyPort.send('Detection error: $e');
        }
        break;

      case _MessageType.detectYuv:
        if (!modelLoaded) {
          message.replyPort.send('Model not initialized');
          return;
        }

        try {
          final yData = message.params!['yData'] as Uint8List;
          final uData = message.params!['uData'] as Uint8List;
          final vData = message.params!['vData'] as Uint8List;
          final width = message.params!['width'] as int;
          final height = message.params!['height'] as int;
          final yRowStride = message.params!['yRowStride'] as int;
          final uvRowStride = message.params!['uvRowStride'] as int;
          final uvPixelStride = message.params!['uvPixelStride'] as int;
          final rotation = message.params!['rotation'] as int;
          final confThreshold = message.params!['confThreshold'] as double;
          final iouThreshold = message.params!['iouThreshold'] as double;

          // Allocate native buffers
          final yBuffer = malloc.allocate<Uint8>(yData.length);
          final uBuffer = malloc.allocate<Uint8>(uData.length);
          final vBuffer = malloc.allocate<Uint8>(vData.length);

          try {
            yBuffer.asTypedList(yData.length).setAll(0, yData);
            uBuffer.asTypedList(uData.length).setAll(0, uData);
            vBuffer.asTypedList(vData.length).setAll(0, vData);

            final result = yolo.detectFromYUV(
              yBuffer,
              uBuffer,
              vBuffer,
              width,
              height,
              yRowStride,
              uvRowStride,
              uvPixelStride,
              rotation: rotation,
              confThreshold: confThreshold,
              iouThreshold: iouThreshold,
            );

            message.replyPort.send(_yoloResultToJson(result));
          } finally {
            malloc.free(yBuffer);
            malloc.free(uBuffer);
            malloc.free(vBuffer);
          }
        } catch (e) {
          message.replyPort.send('Detection error: $e');
        }
        break;

      case _MessageType.release:
        yolo.release();
        modelLoaded = false;
        message.replyPort.send('released');
        break;
    }
  });
}

/// Convert YoloResult to JSON map for passing between isolates
Map<String, dynamic> _yoloResultToJson(YoloResult result) {
  final json = <String, dynamic>{
    'count': result.count,
    'inference_time_ms': result.inferenceTimeMs,
    'image_width': result.imageWidth,
    'image_height': result.imageHeight,
    'detections': result.detections.map((d) => {
      'class_id': d.classId,
      'class_name': d.className,
      'confidence': d.confidence,
      'x1': d.x1,
      'y1': d.y1,
      'x2': d.x2,
      'y2': d.y2,
    }).toList(),
  };

  // Only include error fields if there's an actual error
  if (result.error != null) {
    json['error'] = result.error;
    json['code'] = result.errorCode;
  }

  return json;
}
