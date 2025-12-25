import 'dart:ffi' hide Size;
import 'dart:io';
import 'dart:ui' as ui;

import 'package:camera/camera.dart';
import 'package:ffi/ffi.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_yolo_open_kit/flutter_yolo_open_kit.dart';
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';

import 'services/yolo_service.dart';

late List<CameraDescription> _cameras;

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  _cameras = await availableCameras();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'YOLO Detection',
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

// Common COCO class filters
class ClassFilter {
  final int classId;
  final String name;
  final String icon;

  const ClassFilter(this.classId, this.name, this.icon);

  static const List<ClassFilter> commonClasses = [
    ClassFilter(0, 'Person', '🧑'),
    ClassFilter(15, 'Cat', '🐱'),
    ClassFilter(16, 'Dog', '🐕'),
    ClassFilter(2, 'Car', '🚗'),
    ClassFilter(41, 'Cup', '☕'),
    ClassFilter(39, 'Bottle', '🍾'),
    ClassFilter(56, 'Chair', '🪑'),
    ClassFilter(67, 'Phone', '📱'),
    ClassFilter(63, 'Laptop', '💻'),
    ClassFilter(24, 'Backpack', '🎒'),
  ];
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final _yolo = FlutterYoloOpenKit.instance;
  final _picker = ImagePicker();

  String _status = 'Initializing...';
  bool _isLoading = false;
  ModelType _selectedModel = ModelType.ppyoloeM;
  double _confidenceThreshold = 0.7;
  Set<int> _selectedClassIds = {}; // Empty = all classes
  String? _currentModelPath; // Store model path for isolate

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
        final data = await rootBundle.load('assets/${_selectedModel.fileName}');
        await modelFile.writeAsBytes(data.buffer.asUint8List());
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

  Future<void> _pickAndDetect(ImageSource source) async {
    if (!_yolo.isInitialized) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Model not loaded yet')),
      );
      return;
    }

    final picked = await _picker.pickImage(source: source);
    if (picked != null && mounted) {
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => DetectionResultPage(
            imagePath: picked.path,
            confidenceThreshold: _confidenceThreshold,
            filterClassIds: _selectedClassIds,
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
        title: const Text('YOLO Detection'),
        centerTitle: true,
      ),
      body: SingleChildScrollView(
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

            // Main features
            const Text(
              'Detection',
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
            ),
            const SizedBox(height: 12),

            Row(
              children: [
                Expanded(
                  child: _FeatureCard(
                    icon: Icons.photo_library,
                    label: 'Gallery',
                    onTap: _yolo.isInitialized && !_isLoading
                        ? () => _pickAndDetect(ImageSource.gallery)
                        : null,
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: _FeatureCard(
                    icon: Icons.camera_alt,
                    label: 'Camera',
                    onTap: _yolo.isInitialized && !_isLoading
                        ? () => _pickAndDetect(ImageSource.camera)
                        : null,
                  ),
                ),
              ],
            ),

            const SizedBox(height: 12),

            _FeatureCard(
              icon: Icons.videocam,
              label: 'Real-time Scan',
              onTap: _yolo.isInitialized && !_isLoading && _currentModelPath != null
                  ? () => Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (context) => RealTimeScanPage(
                            modelPath: _currentModelPath!,
                            confidenceThreshold: _confidenceThreshold,
                            filterClassIds: _selectedClassIds,
                          ),
                        ),
                      )
                  : null,
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

            // Class filter
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        const Icon(Icons.filter_list, color: Colors.grey),
                        const SizedBox(width: 16),
                        const Expanded(
                          child: Text('Filter Classes'),
                        ),
                        Text(
                          _selectedClassIds.isEmpty
                              ? 'All'
                              : '${_selectedClassIds.length} selected',
                          style: const TextStyle(
                            fontWeight: FontWeight.bold,
                            fontSize: 14,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 12),
                    Wrap(
                      spacing: 8,
                      runSpacing: 8,
                      children: [
                        // "All" chip
                        FilterChip(
                          label: const Text('All'),
                          selected: _selectedClassIds.isEmpty,
                          onSelected: (selected) {
                            setState(() {
                              _selectedClassIds = {};
                            });
                          },
                        ),
                        // Class chips
                        ...ClassFilter.commonClasses.map((filter) => FilterChip(
                              avatar: Text(filter.icon),
                              label: Text(filter.name),
                              selected: _selectedClassIds.contains(filter.classId),
                              onSelected: (selected) {
                                setState(() {
                                  if (selected) {
                                    _selectedClassIds = {..._selectedClassIds, filter.classId};
                                  } else {
                                    _selectedClassIds = {..._selectedClassIds}..remove(filter.classId);
                                  }
                                });
                              },
                            )),
                      ],
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
    );
  }
}

class _FeatureCard extends StatelessWidget {
  final IconData icon;
  final String label;
  final String? subtitle;
  final VoidCallback? onTap;

  const _FeatureCard({
    required this.icon,
    required this.label,
    this.subtitle,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final isEnabled = onTap != null;
    return Card(
      color: isEnabled ? null : Colors.grey.shade100,
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(12),
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 24, horizontal: 16),
          child: Column(
            children: [
              Icon(
                icon,
                size: 40,
                color: isEnabled ? Theme.of(context).primaryColor : Colors.grey,
              ),
              const SizedBox(height: 8),
              Text(
                label,
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w500,
                  color: isEnabled ? null : Colors.grey,
                ),
              ),
              if (subtitle != null) ...[
                const SizedBox(height: 4),
                Text(
                  subtitle!,
                  style: TextStyle(
                    fontSize: 12,
                    color: Colors.grey.shade500,
                  ),
                ),
              ],
            ],
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
  final Set<int> filterClassIds;

  const DetectionResultPage({
    super.key,
    required this.imagePath,
    required this.confidenceThreshold,
    required this.filterClassIds,
  });

  @override
  State<DetectionResultPage> createState() => _DetectionResultPageState();
}

class _DetectionResultPageState extends State<DetectionResultPage> {
  final _yolo = FlutterYoloOpenKit.instance;
  YoloResult? _result;
  List<YoloDetection> _filteredDetections = [];
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

    // Filter detections by class if filter is set
    List<YoloDetection> filtered = result.detections;
    if (widget.filterClassIds.isNotEmpty) {
      filtered = result.detections
          .where((det) => widget.filterClassIds.contains(det.classId))
          .toList();
    }

    setState(() {
      _result = result;
      _filteredDetections = filtered;
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
                  widget.filterClassIds.isEmpty
                      ? '${_filteredDetections.length} objects | ${_result!.inferenceTimeMs}ms'
                      : '${_filteredDetections.length}/${_result!.count} objects | ${_result!.inferenceTimeMs}ms',
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
              : Column(
                  children: [
                    // Image with detections
                    Expanded(
                      flex: 3,
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
                                      detections: _filteredDetections,
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
                    Expanded(
                      flex: 2,
                      child: _filteredDetections.isEmpty
                          ? Center(
                              child: Text(
                                widget.filterClassIds.isEmpty
                                    ? 'No objects detected'
                                    : 'No matching objects (${_result?.count ?? 0} total)',
                                style: const TextStyle(color: Colors.grey),
                              ),
                            )
                          : ListView.builder(
                              padding: const EdgeInsets.all(8),
                              itemCount: _filteredDetections.length,
                              itemBuilder: (context, index) {
                                final det = _filteredDetections[index];
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
      bgPaint.color = color.withOpacity(0.8);

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

// Real-time Scan Page
class RealTimeScanPage extends StatefulWidget {
  final String modelPath;
  final double confidenceThreshold;
  final Set<int> filterClassIds;

  const RealTimeScanPage({
    super.key,
    required this.modelPath,
    required this.confidenceThreshold,
    required this.filterClassIds,
  });

  @override
  State<RealTimeScanPage> createState() => _RealTimeScanPageState();
}

class _RealTimeScanPageState extends State<RealTimeScanPage> {
  final _yoloService = YoloService.instance;
  final _yoloDirect = FlutterYoloOpenKit.instance;
  CameraController? _cameraController;
  bool _isDetecting = false;
  bool _isServiceReady = false;
  bool _useIsolate = true; // Toggle for isolate vs main isolate mode
  List<YoloDetection> _detections = [];
  int _fps = 0;
  int _inferenceMs = 0;
  DateTime _lastFpsUpdate = DateTime.now();
  int _frameCount = 0;

  @override
  void initState() {
    super.initState();
    _initService();
  }

  Future<void> _initService() async {
    // Initialize YoloService in isolate
    try {
      await _yoloService.init(widget.modelPath);
      _isServiceReady = true;
      _initCamera();
    } catch (_) {
      // Handle silently
    }
  }

  Future<void> _initCamera() async {
    if (_cameras.isEmpty) {
      return;
    }

    // Use back camera
    final camera = _cameras.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.back,
      orElse: () => _cameras.first,
    );

    _cameraController = CameraController(
      camera,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: Platform.isIOS
          ? ImageFormatGroup.bgra8888
          : ImageFormatGroup.yuv420,
    );

    try {
      await _cameraController!.initialize();
      if (mounted) {
        setState(() {});
        _startDetection();
      }
    } catch (_) {
      // Handle silently
    }
  }

  void _startDetection() {
    _cameraController?.startImageStream((CameraImage image) {
      if (_isDetecting || !_isServiceReady) return;
      _isDetecting = true;
      _processFrame(image);
    });
  }

  Future<void> _processFrame(CameraImage image) async {
    try {
      final result = await _detectFromCameraImage(image);

      // Filter detections
      List<YoloDetection> filtered = result.detections;
      if (widget.filterClassIds.isNotEmpty) {
        filtered = result.detections
            .where((det) => widget.filterClassIds.contains(det.classId))
            .toList();
      }

      // Update FPS
      _frameCount++;
      final now = DateTime.now();
      if (now.difference(_lastFpsUpdate).inMilliseconds >= 1000) {
        _fps = _frameCount;
        _frameCount = 0;
        _lastFpsUpdate = now;
      }

      if (mounted) {
        setState(() {
          _detections = filtered;
          _inferenceMs = result.inferenceTimeMs;
        });
      }
    } catch (_) {
      // Handle silently
    } finally {
      _isDetecting = false;
    }
  }

  Future<YoloResult> _detectFromCameraImage(CameraImage image) async {
    if (Platform.isIOS) {
      // iOS: BGRA format
      final plane = image.planes[0];
      final bytes = plane.bytes;

      if (_useIsolate) {
        // Use isolate-based detection
        return _yoloService.detectFromBuffer(
          imageData: bytes,
          width: image.width,
          height: image.height,
          stride: plane.bytesPerRow,
          confThreshold: widget.confidenceThreshold,
          iouThreshold: 0.45,
        );
      } else {
        // Direct FFI on main isolate
        final Pointer<Uint8> nativeBuffer = malloc.allocate<Uint8>(bytes.length);
        try {
          nativeBuffer.asTypedList(bytes.length).setAll(0, bytes);
          return _yoloDirect.detectFromBuffer(
            nativeBuffer,
            image.width,
            image.height,
            plane.bytesPerRow,
            confThreshold: widget.confidenceThreshold,
            iouThreshold: 0.45,
          );
        } finally {
          malloc.free(nativeBuffer);
        }
      }
    } else {
      // Android: YUV420 format
      final yPlane = image.planes[0];
      final uPlane = image.planes[1];
      final vPlane = image.planes[2];
      final sensorOrientation = _cameraController?.description.sensorOrientation ?? 0;

      if (_useIsolate) {
        // Use isolate-based detection
        return _yoloService.detectFromYuv(
          yData: yPlane.bytes,
          uData: uPlane.bytes,
          vData: vPlane.bytes,
          width: image.width,
          height: image.height,
          yRowStride: yPlane.bytesPerRow,
          uvRowStride: uPlane.bytesPerRow,
          uvPixelStride: uPlane.bytesPerPixel ?? 1,
          rotation: sensorOrientation,
          confThreshold: widget.confidenceThreshold,
          iouThreshold: 0.45,
        );
      } else {
        // Direct FFI on main isolate
        final Pointer<Uint8> yBuffer = malloc.allocate<Uint8>(yPlane.bytes.length);
        final Pointer<Uint8> uBuffer = malloc.allocate<Uint8>(uPlane.bytes.length);
        final Pointer<Uint8> vBuffer = malloc.allocate<Uint8>(vPlane.bytes.length);
        try {
          yBuffer.asTypedList(yPlane.bytes.length).setAll(0, yPlane.bytes);
          uBuffer.asTypedList(uPlane.bytes.length).setAll(0, uPlane.bytes);
          vBuffer.asTypedList(vPlane.bytes.length).setAll(0, vPlane.bytes);
          return _yoloDirect.detectFromYUV(
            yBuffer,
            uBuffer,
            vBuffer,
            image.width,
            image.height,
            yPlane.bytesPerRow,
            uPlane.bytesPerRow,
            uPlane.bytesPerPixel ?? 1,
            rotation: sensorOrientation,
            confThreshold: widget.confidenceThreshold,
            iouThreshold: 0.45,
          );
        } finally {
          malloc.free(yBuffer);
          malloc.free(uBuffer);
          malloc.free(vBuffer);
        }
      }
    }
  }

  @override
  void dispose() {
    _cameraController?.stopImageStream();
    _cameraController?.dispose();
    // Note: Don't dispose YoloService as it's a singleton that can be reused
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return Scaffold(
        appBar: AppBar(title: const Text('Real-time Scan')),
        body: const Center(child: CircularProgressIndicator()),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('Real-time Scan'),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 16),
            child: Center(
              child: Text(
                '$_fps FPS | ${_inferenceMs}ms',
                style: const TextStyle(fontSize: 14),
              ),
            ),
          ),
        ],
      ),
      body: Stack(
        fit: StackFit.expand,
        children: [
          // Camera preview
          CameraPreview(_cameraController!),

          // Detection overlay
          CustomPaint(
            painter: RealTimeDetectionPainter(
              detections: _detections,
              imageWidth: _cameraController!.value.previewSize!.height.toInt(),
              imageHeight: _cameraController!.value.previewSize!.width.toInt(),
              screenSize: MediaQuery.of(context).size,
              isFrontCamera: _cameraController!.description.lensDirection ==
                  CameraLensDirection.front,
            ),
          ),

          // Detection count badge
          Positioned(
            bottom: 20,
            left: 20,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              decoration: BoxDecoration(
                color: Colors.black54,
                borderRadius: BorderRadius.circular(20),
              ),
              child: Text(
                '${_detections.length} objects detected',
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ),

          // Isolate mode toggle
          Positioned(
            bottom: 20,
            right: 20,
            child: GestureDetector(
              onTap: () {
                setState(() {
                  _useIsolate = !_useIsolate;
                });
              },
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                decoration: BoxDecoration(
                  color: _useIsolate ? Colors.green.withOpacity(0.8) : Colors.orange.withOpacity(0.8),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      _useIsolate ? Icons.memory : Icons.speed,
                      color: Colors.white,
                      size: 18,
                    ),
                    const SizedBox(width: 6),
                    Text(
                      _useIsolate ? 'Isolate' : 'Main',
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 14,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class RealTimeDetectionPainter extends CustomPainter {
  final List<YoloDetection> detections;
  final int imageWidth;
  final int imageHeight;
  final Size screenSize;
  final bool isFrontCamera;

  RealTimeDetectionPainter({
    required this.detections,
    required this.imageWidth,
    required this.imageHeight,
    required this.screenSize,
    required this.isFrontCamera,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final scaleX = size.width / imageWidth;
    final scaleY = size.height / imageHeight;

    final boxPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3;

    final bgPaint = Paint()..style = PaintingStyle.fill;

    for (final det in detections) {
      final color = _getColorForClass(det.classId);
      boxPaint.color = color;
      bgPaint.color = color.withOpacity(0.8);

      double left = det.x1 * scaleX;
      double right = det.x2 * scaleX;

      // Mirror for front camera
      if (isFrontCamera) {
        final temp = left;
        left = size.width - right;
        right = size.width - temp;
      }

      final rect = Rect.fromLTRB(
        left,
        det.y1 * scaleY,
        right,
        det.y2 * scaleY,
      );

      canvas.drawRect(rect, boxPaint);

      // Draw label
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
      if (labelY < 0) labelY = rect.top + 4;

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
