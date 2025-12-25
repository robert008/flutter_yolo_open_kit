Pod::Spec.new do |s|
  s.name             = 'flutter_yolo_open_kit'
  s.version          = '1.0.0'
  s.summary          = 'YOLO object detection plugin for Flutter using ONNX Runtime'
  s.description      = <<-DESC
A commercial-friendly YOLO detector for Flutter, powered by YOLOX and PP-YOLOE+ (Apache 2.0).
Features: real-time object detection, multiple model support, Core ML acceleration.
                       DESC
  s.homepage         = 'https://github.com/robert008/flutter_yolo_open_kit'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Robert Chuang' => 'figo007007@gmail.com' }
  s.source           = { :path => '.' }

  s.source_files = 'Classes/**/*.{h,m}'
  s.preserve_paths = 'download_frameworks.sh'

  # Download frameworks before build
  s.prepare_command = <<-CMD
    ./download_frameworks.sh
  CMD

  # Static libraries: user code + ONNX Runtime
  s.vendored_libraries = 'libflutter_yolo_open_kit.a', 'static_libs/libonnxruntime_complete.a'

  # OpenCV framework
  s.vendored_frameworks = 'Frameworks/opencv2.framework'

  s.ios.deployment_target = '12.0'
  s.static_framework = true

  # Linker flags and header paths
  s.pod_target_xcconfig = {
    'OTHER_LDFLAGS' => '-force_load $(PODS_TARGET_SRCROOT)/libflutter_yolo_open_kit.a -force_load $(PODS_TARGET_SRCROOT)/static_libs/libonnxruntime_complete.a -lc++',
    'CLANG_CXX_LANGUAGE_STANDARD' => 'c++17',
    'DEFINES_MODULE' => 'YES',
    'GCC_SYMBOLS_PRIVATE_EXTERN' => 'NO',
    'HEADER_SEARCH_PATHS' => '$(PODS_TARGET_SRCROOT)/Headers'
  }

  # System frameworks
  s.frameworks = 'CoreML', 'Accelerate', 'Foundation', 'CoreVideo', 'CoreMedia', 'AVFoundation'

  # System libraries
  s.libraries = 'z', 'c++'

  s.dependency 'Flutter'
end
