import 'dart:async';
import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_yolo_open_kit/flutter_yolo_open_kit.dart';
import 'package:path_provider/path_provider.dart';
import 'package:file_selector/file_selector.dart';
import 'package:media_kit/media_kit.dart';
import 'package:media_kit_video/media_kit_video.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  MediaKit.ensureInitialized();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'YOLO Detection - Linux',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue),
        useMaterial3: true,
      ),
      home: const HomePage(),
    );
  }
}

// Available models
enum ModelType {
  yoloxM('YOLOX-M', 'yolox_m.onnx'),
  yoloxL('YOLOX-L', 'yolox_l.onnx'),
  ppyoloeS('PP-YOLOE+ S', 'ppyoloe_plus_s.onnx'),
  ppyoloeM('PP-YOLOE+ M', 'ppyoloe_plus_m.onnx'),
  ppyoloeL('PP-YOLOE+ L', 'ppyoloe_plus_l.onnx');

  const ModelType(this.displayName, this.fileName);
  final String displayName;
  final String fileName;
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final _yolo = FlutterYoloOpenKit.instance;

  String _status = 'Initializing...';
  bool _isLoading = false;
  ModelType _selectedModel = ModelType.ppyoloeS;
  double _confidenceThreshold = 0.5;
  String? _currentModelPath;

  @override
  void initState() {
    super.initState();
    _initModel();
  }

  Future<void> _initModel() async {
    setState(() {
      _isLoading = true;
      _status = 'Loading ${_selectedModel.displayName}...';
    });

    try {
      final docDir = await getApplicationDocumentsDirectory();
      final modelPath = '${docDir.path}/${_selectedModel.fileName}';

      final modelFile = File(modelPath);
      if (!await modelFile.exists()) {
        setState(() {
          _status = 'Copying model from assets...';
        });
        try {
          final data = await rootBundle.load('assets/${_selectedModel.fileName}');
          await modelFile.writeAsBytes(data.buffer.asUint8List());
        } catch (e) {
          setState(() {
            _isLoading = false;
            _status = 'Model not found in assets. Please add ${_selectedModel.fileName} to assets folder.';
          });
          return;
        }
      }

      // Release previous model if any
      _yolo.release();

      final success = _yolo.init(modelPath);

      setState(() {
        _isLoading = false;
        if (success) {
          _status = '${_selectedModel.displayName} loaded';
          _currentModelPath = modelPath;
        } else {
          _status = 'Failed to load model';
          _currentModelPath = null;
        }
      });
    } catch (e) {
      setState(() {
        _isLoading = false;
        _status = 'Error: $e';
      });
    }
  }

  Future<void> _pickAndDetectImage() async {
    if (!_yolo.isInitialized) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Model not loaded yet')),
      );
      return;
    }

    const typeGroup = XTypeGroup(
      label: 'Images',
      extensions: ['jpg', 'jpeg', 'png', 'bmp'],
      mimeTypes: ['image/jpeg', 'image/png', 'image/bmp'],
    );

    final file = await openFile(acceptedTypeGroups: [typeGroup]);
    if (file != null && mounted) {
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => DetectionResultPage(
            imagePath: file.path,
            confidenceThreshold: _confidenceThreshold,
          ),
        ),
      );
    }
  }

  Future<void> _pickAndDetectVideo() async {
    if (!_yolo.isInitialized) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Model not loaded yet')),
      );
      return;
    }

    const typeGroup = XTypeGroup(
      label: 'Videos',
      extensions: ['mp4', 'avi', 'mov', 'mkv', 'webm'],
      mimeTypes: ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska', 'video/webm'],
    );

    final file = await openFile(acceptedTypeGroups: [typeGroup]);
    if (file != null && mounted) {
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => VideoDetectionPage(
            videoPath: file.path,
            confidenceThreshold: _confidenceThreshold,
          ),
        ),
      );
    }
  }

  void _showModelSelector() {
    showModalBottomSheet(
      context: context,
      builder: (context) => SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Padding(
              padding: EdgeInsets.all(16),
              child: Text(
                'Select Model',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              ),
            ),
            ...ModelType.values.map((model) => ListTile(
                  leading: Radio<ModelType>(
                    value: model,
                    groupValue: _selectedModel,
                    onChanged: (value) {
                      Navigator.pop(context);
                      if (value != null && value != _selectedModel) {
                        setState(() {
                          _selectedModel = value;
                        });
                        _initModel();
                      }
                    },
                  ),
                  title: Text(model.displayName),
                  onTap: () {
                    Navigator.pop(context);
                    if (model != _selectedModel) {
                      setState(() {
                        _selectedModel = model;
                      });
                      _initModel();
                    }
                  },
                )),
            const SizedBox(height: 16),
          ],
        ),
      ),
    );
  }

  @override
  void dispose() {
    _yolo.release();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('YOLO Detection - Linux'),
        centerTitle: true,
      ),
      body: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 600),
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                // Status card
                Card(
                  color: _yolo.isInitialized
                      ? Colors.green.shade50
                      : Colors.orange.shade50,
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Row(
                      children: [
                        if (_isLoading)
                          const SizedBox(
                            width: 20,
                            height: 20,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                        else
                          Icon(
                            _yolo.isInitialized ? Icons.check_circle : Icons.warning,
                            color: _yolo.isInitialized ? Colors.green : Colors.orange,
                          ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: Text(
                            _status,
                            style: const TextStyle(fontSize: 14),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),

                const SizedBox(height: 32),

                // Main action buttons
                Row(
                  children: [
                    Expanded(
                      child: SizedBox(
                        height: 100,
                        child: ElevatedButton.icon(
                          onPressed: _yolo.isInitialized && !_isLoading
                              ? _pickAndDetectImage
                              : null,
                          icon: const Icon(Icons.image, size: 28),
                          label: const Text(
                            'Image',
                            style: TextStyle(fontSize: 16),
                          ),
                          style: ElevatedButton.styleFrom(
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(16),
                            ),
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(width: 16),
                    Expanded(
                      child: SizedBox(
                        height: 100,
                        child: ElevatedButton.icon(
                          onPressed: _yolo.isInitialized && !_isLoading
                              ? _pickAndDetectVideo
                              : null,
                          icon: const Icon(Icons.videocam, size: 28),
                          label: const Text(
                            'Video',
                            style: TextStyle(fontSize: 16),
                          ),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.deepPurple,
                            foregroundColor: Colors.white,
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(16),
                            ),
                          ),
                        ),
                      ),
                    ),
                  ],
                ),

                const SizedBox(height: 32),

                // Settings
                const Text(
                  'Settings',
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
                ),
                const SizedBox(height: 12),

                // Model selector
                Card(
                  child: ListTile(
                    leading: const Icon(Icons.model_training),
                    title: const Text('Model'),
                    subtitle: Text(_selectedModel.displayName),
                    trailing: const Icon(Icons.chevron_right),
                    onTap: _isLoading ? null : _showModelSelector,
                  ),
                ),

                // Confidence slider
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            const Icon(Icons.tune, color: Colors.grey),
                            const SizedBox(width: 16),
                            const Expanded(
                              child: Text('Confidence Threshold'),
                            ),
                            Text(
                              '${(_confidenceThreshold * 100).toInt()}%',
                              style: const TextStyle(
                                fontWeight: FontWeight.bold,
                                fontSize: 16,
                              ),
                            ),
                          ],
                        ),
                        Slider(
                          value: _confidenceThreshold,
                          min: 0.1,
                          max: 0.95,
                          divisions: 17,
                          onChanged: (value) {
                            setState(() {
                              _confidenceThreshold = value;
                            });
                          },
                        ),
                      ],
                    ),
                  ),
                ),

                const SizedBox(height: 24),

                // Version info
                Text(
                  'Plugin v${_yolo.version}',
                  textAlign: TextAlign.center,
                  style: TextStyle(color: Colors.grey.shade500, fontSize: 12),
                ),
                const SizedBox(height: 20),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

// Detection Result Page
class DetectionResultPage extends StatefulWidget {
  final String imagePath;
  final double confidenceThreshold;

  const DetectionResultPage({
    super.key,
    required this.imagePath,
    required this.confidenceThreshold,
  });

  @override
  State<DetectionResultPage> createState() => _DetectionResultPageState();
}

class _DetectionResultPageState extends State<DetectionResultPage> {
  final _yolo = FlutterYoloOpenKit.instance;
  YoloResult? _result;
  bool _isLoading = true;
  Size? _imageSize;

  @override
  void initState() {
    super.initState();
    _detect();
  }

  Future<void> _detect() async {
    // Get image size
    final bytes = await File(widget.imagePath).readAsBytes();
    final codec = await ui.instantiateImageCodec(bytes);
    final frame = await codec.getNextFrame();
    _imageSize = Size(
      frame.image.width.toDouble(),
      frame.image.height.toDouble(),
    );

    // Run detection
    final result = _yolo.detectFromPath(
      widget.imagePath,
      confThreshold: widget.confidenceThreshold,
      iouThreshold: 0.45,
    );

    setState(() {
      _result = result;
      _isLoading = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Detection Result'),
        actions: [
          if (_result != null && !_result!.hasError)
            Padding(
              padding: const EdgeInsets.only(right: 16),
              child: Center(
                child: Text(
                  '${_result!.count} objects | ${_result!.inferenceTimeMs}ms',
                  style: const TextStyle(fontSize: 14),
                ),
              ),
            ),
        ],
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : _result?.hasError == true
              ? Center(
                  child: Text(
                    'Error: ${_result!.error}',
                    style: const TextStyle(color: Colors.red),
                  ),
                )
              : Row(
                  children: [
                    // Image with detections
                    Expanded(
                      flex: 2,
                      child: Container(
                        color: Colors.black,
                        child: LayoutBuilder(
                          builder: (context, constraints) {
                            return Stack(
                              fit: StackFit.expand,
                              children: [
                                Image.file(
                                  File(widget.imagePath),
                                  fit: BoxFit.contain,
                                ),
                                if (_result != null && _imageSize != null)
                                  CustomPaint(
                                    painter: DetectionPainter(
                                      detections: _result!.detections,
                                      imageWidth: _result!.imageWidth.toDouble(),
                                      imageHeight: _result!.imageHeight.toDouble(),
                                      containerWidth: constraints.maxWidth,
                                      containerHeight: constraints.maxHeight,
                                      displayImageSize: _imageSize!,
                                    ),
                                  ),
                              ],
                            );
                          },
                        ),
                      ),
                    ),

                    // Detection list
                    SizedBox(
                      width: 300,
                      child: _result?.detections.isEmpty == true
                          ? const Center(
                              child: Text(
                                'No objects detected',
                                style: TextStyle(color: Colors.grey),
                              ),
                            )
                          : ListView.builder(
                              padding: const EdgeInsets.all(8),
                              itemCount: _result?.detections.length ?? 0,
                              itemBuilder: (context, index) {
                                final det = _result!.detections[index];
                                return Card(
                                  child: ListTile(
                                    leading: CircleAvatar(
                                      backgroundColor:
                                          _getColorForClass(det.classId),
                                      child: Text(
                                        '${det.classId}',
                                        style: const TextStyle(
                                          color: Colors.white,
                                          fontSize: 12,
                                        ),
                                      ),
                                    ),
                                    title: Text(det.className),
                                    subtitle: Text(
                                      '${(det.confidence * 100).toStringAsFixed(1)}%',
                                    ),
                                    trailing: Text(
                                      '${det.width.toInt()}x${det.height.toInt()}',
                                      style: const TextStyle(
                                        color: Colors.grey,
                                        fontSize: 12,
                                      ),
                                    ),
                                  ),
                                );
                              },
                            ),
                    ),
                  ],
                ),
    );
  }

  Color _getColorForClass(int classId) {
    final colors = [
      Colors.red,
      Colors.green,
      Colors.blue,
      Colors.orange,
      Colors.purple,
      Colors.cyan,
      Colors.pink,
      Colors.teal,
      Colors.amber,
      Colors.indigo,
    ];
    return colors[classId % colors.length];
  }
}

class DetectionPainter extends CustomPainter {
  final List<YoloDetection> detections;
  final double imageWidth;
  final double imageHeight;
  final double containerWidth;
  final double containerHeight;
  final Size displayImageSize;

  DetectionPainter({
    required this.detections,
    required this.imageWidth,
    required this.imageHeight,
    required this.containerWidth,
    required this.containerHeight,
    required this.displayImageSize,
  });

  @override
  void paint(Canvas canvas, Size size) {
    // Calculate displayed image size (BoxFit.contain)
    final imageAspect = displayImageSize.width / displayImageSize.height;
    final containerAspect = containerWidth / containerHeight;

    double displayWidth, displayHeight;
    if (imageAspect > containerAspect) {
      displayWidth = containerWidth;
      displayHeight = containerWidth / imageAspect;
    } else {
      displayHeight = containerHeight;
      displayWidth = containerHeight * imageAspect;
    }

    // Calculate scale from original image to displayed size
    final scale = displayWidth / imageWidth;

    // Calculate offset to center the image
    final offsetX = (containerWidth - displayWidth) / 2;
    final offsetY = (containerHeight - displayHeight) / 2;

    final boxPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3;

    final bgPaint = Paint()..style = PaintingStyle.fill;

    for (final det in detections) {
      final color = _getColorForClass(det.classId);
      boxPaint.color = color;
      bgPaint.color = color.withValues(alpha: 0.8);

      final rect = Rect.fromLTRB(
        det.x1 * scale + offsetX,
        det.y1 * scale + offsetY,
        det.x2 * scale + offsetX,
        det.y2 * scale + offsetY,
      );

      canvas.drawRect(rect, boxPaint);

      final label = '${det.className} ${(det.confidence * 100).toInt()}%';
      final textSpan = TextSpan(
        text: label,
        style: const TextStyle(
          color: Colors.white,
          fontSize: 14,
          fontWeight: FontWeight.bold,
        ),
      );
      final textPainter = TextPainter(
        text: textSpan,
        textDirection: TextDirection.ltr,
      );
      textPainter.layout();

      double labelY = rect.top - textPainter.height - 4;
      if (labelY < offsetY) labelY = rect.top + 4;

      final labelRect = Rect.fromLTWH(
        rect.left,
        labelY,
        textPainter.width + 8,
        textPainter.height + 4,
      );
      canvas.drawRRect(
        RRect.fromRectAndRadius(labelRect, const Radius.circular(4)),
        bgPaint,
      );
      textPainter.paint(canvas, Offset(rect.left + 4, labelY + 2));
    }
  }

  Color _getColorForClass(int classId) {
    final colors = [
      Colors.red,
      Colors.green,
      Colors.blue,
      Colors.orange,
      Colors.purple,
      Colors.cyan,
      Colors.pink,
      Colors.teal,
      Colors.amber,
      Colors.indigo,
    ];
    return colors[classId % colors.length];
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}

/// FFmpeg pipe-based frame extractor for efficient sequential playback
class FFmpegPipeExtractor {
  final String videoPath;
  final int width;
  final int height;
  final double fps;

  Process? _process;
  StreamSubscription<List<int>>? _subscription;
  bool _isRunning = false;
  double _currentPosition = 0;

  // Buffer for accumulating bytes
  final List<int> _buffer = [];
  final _frameCompleter = <Completer<Uint8List?>>[];

  // JPEG markers
  static const int _jpegStart1 = 0xFF;
  static const int _jpegStart2 = 0xD8;
  static const int _jpegEnd1 = 0xFF;
  static const int _jpegEnd2 = 0xD9;

  FFmpegPipeExtractor({
    required this.videoPath,
    required this.width,
    required this.height,
    this.fps = 5.0,
  });

  bool get isRunning => _isRunning;
  double get currentPosition => _currentPosition;

  /// Start extracting frames from the given position
  Future<void> start(double startSeconds) async {
    await stop();

    _currentPosition = startSeconds;
    _buffer.clear();

    debugPrint('FFmpeg: Starting from $startSeconds for $videoPath');

    // Start ffmpeg outputting JPEG frames to stdout
    _process = await Process.start('ffmpeg', [
      '-ss', startSeconds.toStringAsFixed(2),
      '-i', videoPath,
      '-vf', 'fps=$fps',
      '-f', 'image2pipe',
      '-vcodec', 'mjpeg',
      '-q:v', '5',
      '-',
    ]);

    _isRunning = true;

    // Log stderr for debugging
    _process!.stderr.transform(const SystemEncoding().decoder).listen((data) {
      // Only print errors, not progress
      if (data.contains('Error') || data.contains('error')) {
        debugPrint('FFmpeg stderr: $data');
      }
    });

    // Start listening to stdout and parse JPEG frames
    _subscription = _process!.stdout.listen(
      (chunk) {
        _buffer.addAll(chunk);

        // Prevent buffer from growing too large (max 500KB)
        // If too large, discard old data to keep only recent frames
        if (_buffer.length > 500000) {
          debugPrint('FFmpeg: Buffer too large (${_buffer.length}), trimming...');
          // Find the last JPEG start marker and keep from there
          for (int i = _buffer.length - 2; i >= 0; i--) {
            if (_buffer[i] == _jpegStart1 && _buffer[i + 1] == _jpegStart2) {
              _buffer.removeRange(0, i);
              break;
            }
          }
        }

        _tryExtractFrame();
      },
      onDone: () {
        debugPrint('FFmpeg: Stream ended');
        _isRunning = false;
        // Complete any pending requests with null
        for (final completer in _frameCompleter) {
          if (!completer.isCompleted) {
            completer.complete(null);
          }
        }
        _frameCompleter.clear();
      },
      onError: (e) {
        debugPrint('FFmpeg pipe error: $e');
        _isRunning = false;
      },
    );

    // Wait a bit for ffmpeg to start producing frames
    await Future.delayed(const Duration(milliseconds: 500));
  }

  void _tryExtractFrame() {
    if (_frameCompleter.isEmpty) {
      return;
    }

    // Look for JPEG frame in buffer
    int startIdx = -1;
    int endIdx = -1;

    for (int i = 0; i < _buffer.length - 1; i++) {
      if (_buffer[i] == _jpegStart1 && _buffer[i + 1] == _jpegStart2) {
        if (startIdx == -1) startIdx = i;
      }
      if (startIdx != -1 && _buffer[i] == _jpegEnd1 && _buffer[i + 1] == _jpegEnd2) {
        endIdx = i + 2;
        break;
      }
    }

    if (startIdx != -1 && endIdx != -1) {
      // Extract the JPEG frame
      final frameBytes = Uint8List.fromList(_buffer.sublist(startIdx, endIdx));
      _buffer.removeRange(0, endIdx);
      _currentPosition += 1.0 / fps;

      // Complete the first waiting completer
      if (_frameCompleter.isNotEmpty) {
        final completer = _frameCompleter.removeAt(0);
        if (!completer.isCompleted) {
          completer.complete(frameBytes);
        }
      }
    }
  }

  /// Read the next frame as JPEG bytes
  Future<Uint8List?> nextFrame() async {
    if (_process == null || !_isRunning) return null;

    final completer = Completer<Uint8List?>();
    _frameCompleter.add(completer);

    // Try to extract from existing buffer first
    _tryExtractFrame();

    // Wait with longer timeout (YOLO can take 500ms+)
    try {
      return await completer.future.timeout(
        const Duration(seconds: 5),
        onTimeout: () {
          debugPrint('Frame read timeout');
          return null;
        },
      );
    } catch (e) {
      debugPrint('Frame read error: $e');
      return null;
    }
  }

  /// Stop the ffmpeg process
  Future<void> stop() async {
    _isRunning = false;
    await _subscription?.cancel();
    _subscription = null;

    if (_process != null) {
      _process!.kill(ProcessSignal.sigterm);
      await _process!.exitCode.timeout(
        const Duration(milliseconds: 500),
        onTimeout: () {
          _process!.kill(ProcessSignal.sigkill);
          return -1;
        },
      );
      _process = null;
    }

    _buffer.clear();
    for (final completer in _frameCompleter) {
      if (!completer.isCompleted) {
        completer.complete(null);
      }
    }
    _frameCompleter.clear();
  }

  void dispose() {
    stop();
  }
}

// Data classes for Isolate communication
class _DetectionIsolateConfig {
  final SendPort sendPort;
  final String modelPath;
  final double confThreshold;

  _DetectionIsolateConfig({
    required this.sendPort,
    required this.modelPath,
    required this.confThreshold,
  });
}

class _DetectionRequest {
  final String videoPath;
  final double timestamp;
  final String outputPath;

  _DetectionRequest({
    required this.videoPath,
    required this.timestamp,
    required this.outputPath,
  });
}

class _DetectionResult {
  final List<YoloDetection> detections;
  final int inferenceTimeMs;
  final int imageWidth;
  final int imageHeight;
  final bool hasError;

  _DetectionResult({
    required this.detections,
    required this.inferenceTimeMs,
    required this.imageWidth,
    required this.imageHeight,
    required this.hasError,
  });
}

// Video Detection Page - Uses media_kit for video playback + async YOLO detection
class VideoDetectionPage extends StatefulWidget {
  final String videoPath;
  final double confidenceThreshold;

  const VideoDetectionPage({
    super.key,
    required this.videoPath,
    required this.confidenceThreshold,
  });

  @override
  State<VideoDetectionPage> createState() => _VideoDetectionPageState();
}

class _VideoDetectionPageState extends State<VideoDetectionPage> {
  final _yolo = FlutterYoloOpenKit.instance;

  // media_kit player and controller
  late final Player _player;
  VideoController? _controller;

  List<YoloDetection> _detections = [];
  int _inferenceTimeMs = 0;
  bool _isVideoReady = false;
  String? _errorMessage;

  Duration _position = Duration.zero;
  Duration _duration = Duration.zero;
  bool _isPlaying = false;

  int _frameWidth = 0;
  int _frameHeight = 0;
  String? _tempFramePath;

  // Detection loop control
  Timer? _detectionTimer;
  bool _isDetecting = false;

  // Isolate for frame extraction AND YOLO detection
  Isolate? _detectionIsolate;
  ReceivePort? _receivePort;
  SendPort? _sendPort;

  @override
  void initState() {
    super.initState();
    _initVideo();
  }

  Future<void> _initVideo() async {
    try {
      final tempDir = await getTemporaryDirectory();
      _tempFramePath = '${tempDir.path}/yolo_frame.jpg';

      // Setup detection isolate (ffmpeg + YOLO in background)
      await _setupDetectionIsolate();

      // Create player
      _player = Player();

      // Create video controller with software rendering
      _controller = VideoController(
        _player,
        configuration: const VideoControllerConfiguration(
          enableHardwareAcceleration: false,
          hwdec: 'no',
        ),
      );

      // Listen to player streams
      _player.stream.duration.listen((d) {
        if (mounted && d.inMilliseconds > 0) {
          setState(() => _duration = d);
        }
      });

      _player.stream.position.listen((p) {
        if (mounted) {
          setState(() => _position = p);
        }
      });

      _player.stream.playing.listen((playing) {
        if (mounted) {
          setState(() => _isPlaying = playing);
          if (playing) {
            _startDetectionLoop();
          } else {
            _stopDetectionLoop();
          }
        }
      });

      _player.stream.width.listen((w) {
        if (w != null && w > 0) _frameWidth = w;
      });

      _player.stream.height.listen((h) {
        if (h != null && h > 0) _frameHeight = h;
      });

      // Open video
      await _player.open(Media(widget.videoPath), play: false);

      setState(() {
        _isVideoReady = true;
      });

      // Wait a bit then run initial detection
      await Future.delayed(const Duration(milliseconds: 500));
      await _runDetection();
    } catch (e) {
      debugPrint('Video init error: $e');
      if (mounted) {
        setState(() {
          _errorMessage = 'Failed to open video: $e';
        });
      }
    }
  }

  Future<void> _setupDetectionIsolate() async {
    _receivePort = ReceivePort();

    // Get model path from main YOLO instance
    final docDir = await getApplicationDocumentsDirectory();
    final modelPath = '${docDir.path}/ppyoloe_plus_s.onnx';

    _detectionIsolate = await Isolate.spawn(
      _detectionIsolateEntry,
      _DetectionIsolateConfig(
        sendPort: _receivePort!.sendPort,
        modelPath: modelPath,
        confThreshold: widget.confidenceThreshold,
      ),
    );

    // Get SendPort from isolate and listen for results
    final completer = Completer<SendPort>();
    _receivePort!.listen((message) {
      if (message is SendPort) {
        completer.complete(message);
      } else if (message is _DetectionResult) {
        _onDetectionResult(message);
      }
    });
    _sendPort = await completer.future;
  }

  static void _detectionIsolateEntry(_DetectionIsolateConfig config) {
    final receivePort = ReceivePort();
    config.sendPort.send(receivePort.sendPort);

    // Initialize YOLO in this isolate
    final yolo = FlutterYoloOpenKit.instance;
    yolo.init(config.modelPath);
    debugPrint('YOLO initialized in isolate');

    receivePort.listen((message) async {
      if (message is _DetectionRequest) {
        // Extract frame using ffmpeg
        final result = await Process.run('ffmpeg', [
          '-y',
          '-ss', message.timestamp.toStringAsFixed(2),
          '-i', message.videoPath,
          '-vframes', '1',
          '-f', 'image2',
          '-q:v', '3',
          message.outputPath,
        ]);

        if (result.exitCode == 0) {
          // Run YOLO detection in this isolate
          final yoloResult = yolo.detectFromPath(
            message.outputPath,
            confThreshold: config.confThreshold,
            iouThreshold: 0.45,
          );

          // Send result back to main thread
          config.sendPort.send(_DetectionResult(
            detections: yoloResult.detections,
            inferenceTimeMs: yoloResult.inferenceTimeMs,
            imageWidth: yoloResult.imageWidth,
            imageHeight: yoloResult.imageHeight,
            hasError: yoloResult.hasError,
          ));
        }
      }
    });
  }

  void _onDetectionResult(_DetectionResult result) {
    if (!mounted) return;

    debugPrint('Detection (isolate): ${result.inferenceTimeMs}ms');

    if (result.imageWidth > 0) _frameWidth = result.imageWidth;
    if (result.imageHeight > 0) _frameHeight = result.imageHeight;

    setState(() {
      _detections = result.hasError ? [] : result.detections;
      _inferenceTimeMs = result.inferenceTimeMs;
    });

    _isDetecting = false;
  }

  void _startDetectionLoop() {
    _stopDetectionLoop();
    // Run detection every 500ms while playing (async, doesn't block video)
    _detectionTimer = Timer.periodic(const Duration(milliseconds: 500), (_) {
      _runDetection();
    });
  }

  void _stopDetectionLoop() {
    _detectionTimer?.cancel();
    _detectionTimer = null;
  }

  Future<void> _runDetection() async {
    if (_isDetecting || _tempFramePath == null || _sendPort == null) return;
    _isDetecting = true;

    // Send detection request to isolate (non-blocking)
    final currentSec = _position.inMilliseconds / 1000.0;
    _sendPort!.send(_DetectionRequest(
      videoPath: widget.videoPath,
      timestamp: currentSec,
      outputPath: _tempFramePath!,
    ));
    // Detection will complete in _onDetectionResult callback
  }

  void _togglePlayback() {
    _player.playOrPause();
  }

  void _seekTo(Duration position) {
    final clamped = Duration(
      milliseconds: position.inMilliseconds.clamp(0, _duration.inMilliseconds),
    );
    _player.seek(clamped);
    Future.delayed(const Duration(milliseconds: 100), _runDetection);
  }

  String _formatTime(Duration d) {
    final mins = d.inMinutes;
    final secs = d.inSeconds % 60;
    return '${mins.toString().padLeft(2, '0')}:${secs.toString().padLeft(2, '0')}';
  }

  @override
  void dispose() {
    _stopDetectionLoop();
    _detectionIsolate?.kill(priority: Isolate.immediate);
    _receivePort?.close();
    _player.dispose();
    if (_tempFramePath != null) {
      File(_tempFramePath!).delete().ignore();
    }
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_errorMessage != null) {
      return Scaffold(
        appBar: AppBar(title: const Text('Video Detection')),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.error, color: Colors.red, size: 64),
              const SizedBox(height: 16),
              Text(_errorMessage!, style: const TextStyle(color: Colors.red)),
              const SizedBox(height: 16),
              ElevatedButton(
                onPressed: () => Navigator.pop(context),
                child: const Text('Go Back'),
              ),
            ],
          ),
        ),
      );
    }

    if (!_isVideoReady || _controller == null) {
      return Scaffold(
        appBar: AppBar(title: const Text('Video Detection')),
        body: const Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              CircularProgressIndicator(),
              SizedBox(height: 16),
              Text('Loading video...'),
            ],
          ),
        ),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('Video Detection'),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 16),
            child: Center(
              child: Text(
                '${_detections.length} objects | ${_inferenceTimeMs}ms',
                style: const TextStyle(fontSize: 14),
              ),
            ),
          ),
        ],
      ),
      body: Row(
        children: [
          // Video with detection overlay
          Expanded(
            flex: 2,
            child: Column(
              children: [
                Expanded(
                  child: Container(
                    color: Colors.black,
                    child: LayoutBuilder(
                      builder: (context, constraints) {
                        return Stack(
                          fit: StackFit.expand,
                          children: [
                            // Video player (media_kit)
                            Video(
                              controller: _controller!,
                              fit: BoxFit.contain,
                            ),
                            // Detection overlay
                            if (_frameWidth > 0 && _frameHeight > 0)
                              CustomPaint(
                                painter: VideoFramePainter(
                                  detections: _detections,
                                  videoWidth: _frameWidth.toDouble(),
                                  videoHeight: _frameHeight.toDouble(),
                                  containerWidth: constraints.maxWidth,
                                  containerHeight: constraints.maxHeight,
                                ),
                              ),
                          ],
                        );
                      },
                    ),
                  ),
                ),
                // Controls
                Container(
                  color: Colors.grey.shade200,
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                  child: Column(
                    children: [
                      Row(
                        children: [
                          Text(_formatTime(_position)),
                          Expanded(
                            child: Slider(
                              value: _position.inMilliseconds.toDouble().clamp(
                                    0, _duration.inMilliseconds.toDouble()),
                              min: 0,
                              max: _duration.inMilliseconds > 0
                                  ? _duration.inMilliseconds.toDouble() : 1,
                              onChanged: (v) => _seekTo(Duration(milliseconds: v.toInt())),
                            ),
                          ),
                          Text(_formatTime(_duration)),
                        ],
                      ),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          IconButton(
                            icon: const Icon(Icons.replay_10),
                            onPressed: () => _seekTo(_position - const Duration(seconds: 10)),
                          ),
                          IconButton(
                            iconSize: 48,
                            icon: Icon(_isPlaying ? Icons.pause_circle : Icons.play_circle),
                            onPressed: _togglePlayback,
                          ),
                          IconButton(
                            icon: const Icon(Icons.forward_10),
                            onPressed: () => _seekTo(_position + const Duration(seconds: 10)),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
          // Detection list
          SizedBox(
            width: 280,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Padding(
                  padding: EdgeInsets.all(12),
                  child: Text('Detected Objects',
                      style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
                ),
                const Divider(height: 1),
                Expanded(
                  child: _detections.isEmpty
                      ? const Center(
                          child: Text('No objects detected',
                              style: TextStyle(color: Colors.grey)))
                      : ListView.builder(
                          padding: const EdgeInsets.all(8),
                          itemCount: _detections.length,
                          itemBuilder: (context, index) {
                            final det = _detections[index];
                            return Card(
                              child: ListTile(
                                dense: true,
                                leading: CircleAvatar(
                                  radius: 16,
                                  backgroundColor: _getColorForClass(det.classId),
                                  child: Text('${det.classId}',
                                      style: const TextStyle(color: Colors.white, fontSize: 10)),
                                ),
                                title: Text(det.className),
                                trailing: Text('${(det.confidence * 100).toInt()}%',
                                    style: const TextStyle(fontWeight: FontWeight.bold)),
                              ),
                            );
                          },
                        ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Color _getColorForClass(int classId) {
    final colors = [
      Colors.red, Colors.green, Colors.blue, Colors.orange, Colors.purple,
      Colors.cyan, Colors.pink, Colors.teal, Colors.amber, Colors.indigo,
    ];
    return colors[classId % colors.length];
  }
}

// Video Frame Painter - draws detection boxes on video frames
class VideoFramePainter extends CustomPainter {
  final List<YoloDetection> detections;
  final double videoWidth;
  final double videoHeight;
  final double containerWidth;
  final double containerHeight;

  VideoFramePainter({
    required this.detections,
    required this.videoWidth,
    required this.videoHeight,
    required this.containerWidth,
    required this.containerHeight,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final videoAspect = videoWidth / videoHeight;
    final containerAspect = containerWidth / containerHeight;

    double displayWidth, displayHeight;
    if (videoAspect > containerAspect) {
      displayWidth = containerWidth;
      displayHeight = containerWidth / videoAspect;
    } else {
      displayHeight = containerHeight;
      displayWidth = containerHeight * videoAspect;
    }

    final scale = displayWidth / videoWidth;
    final offsetX = (containerWidth - displayWidth) / 2;
    final offsetY = (containerHeight - displayHeight) / 2;

    final boxPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3;

    final bgPaint = Paint()..style = PaintingStyle.fill;

    for (final det in detections) {
      final color = _getColorForClass(det.classId);
      boxPaint.color = color;
      bgPaint.color = color.withValues(alpha: 0.8);

      final rect = Rect.fromLTRB(
        det.x1 * scale + offsetX,
        det.y1 * scale + offsetY,
        det.x2 * scale + offsetX,
        det.y2 * scale + offsetY,
      );

      canvas.drawRect(rect, boxPaint);

      final label = '${det.className} ${(det.confidence * 100).toInt()}%';
      final textSpan = TextSpan(
        text: label,
        style: const TextStyle(
          color: Colors.white,
          fontSize: 12,
          fontWeight: FontWeight.bold,
        ),
      );
      final textPainter = TextPainter(
        text: textSpan,
        textDirection: TextDirection.ltr,
      );
      textPainter.layout();

      double labelY = rect.top - textPainter.height - 4;
      if (labelY < offsetY) labelY = rect.top + 4;

      final labelRect = Rect.fromLTWH(
        rect.left,
        labelY,
        textPainter.width + 8,
        textPainter.height + 4,
      );
      canvas.drawRRect(
        RRect.fromRectAndRadius(labelRect, const Radius.circular(4)),
        bgPaint,
      );
      textPainter.paint(canvas, Offset(rect.left + 4, labelY + 2));
    }
  }

  Color _getColorForClass(int classId) {
    final colors = [
      Colors.red,
      Colors.green,
      Colors.blue,
      Colors.orange,
      Colors.purple,
      Colors.cyan,
      Colors.pink,
      Colors.teal,
      Colors.amber,
      Colors.indigo,
    ];
    return colors[classId % colors.length];
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
