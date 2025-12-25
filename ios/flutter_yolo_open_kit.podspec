Pod::Spec.new do |s|
  s.name             = 'flutter_yolo_open_kit'
  s.version          = '0.0.1'
  s.summary          = 'YOLO object detection plugin for Flutter using ONNX Runtime'
  s.homepage         = 'https://github.com/example/flutter_yolo_open_kit'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Author' => 'author@example.com' }
  s.source           = { :path => '.' }

  s.source_files = 'Classes/**/*.{h,m}'

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
