import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_yolo_open_kit/flutter_yolo_open_kit.dart';
import 'package:path_provider/path_provider.dart';
import 'package:file_selector/file_selector.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
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

                // Main action button
                SizedBox(
                  height: 120,
                  child: ElevatedButton.icon(
                    onPressed: _yolo.isInitialized && !_isLoading
                        ? _pickAndDetectImage
                        : null,
                    icon: const Icon(Icons.image, size: 32),
                    label: const Text(
                      'Select Image',
                      style: TextStyle(fontSize: 18),
                    ),
                    style: ElevatedButton.styleFrom(
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(16),
                      ),
                    ),
                  ),
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
