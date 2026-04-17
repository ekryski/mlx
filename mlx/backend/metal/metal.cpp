// Copyright © 2023-2024 Apple Inc.
#include <memory>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/stream.h"

namespace mlx::core::metal {

bool is_available() {
  return true;
}

void start_capture(std::string path, NS::Object* object) {
  auto pool = new_scoped_memory_pool();

  auto descriptor = MTL::CaptureDescriptor::alloc()->init()->autorelease();
  descriptor->setCaptureObject(object);

  if (!path.empty()) {
    auto string = NS::String::string(path.c_str(), NS::UTF8StringEncoding);
    auto url = NS::URL::fileURLWithPath(string);
    descriptor->setDestination(MTL::CaptureDestinationGPUTraceDocument);
    descriptor->setOutputURL(url);
  }

  auto manager = MTL::CaptureManager::sharedCaptureManager();
  NS::Error* error;
  bool started = manager->startCapture(descriptor, &error);
  if (!started) {
    std::ostringstream msg;
    msg << "[metal::start_capture] Failed to start: "
        << error->localizedDescription()->utf8String();
    throw std::runtime_error(msg.str());
  }
}

void start_capture(std::string path) {
  auto& device = metal::device(mlx::core::Device::gpu);
  return start_capture(path, device.mtl_device());
}

void stop_capture() {
  auto pool = new_scoped_memory_pool();
  auto manager = MTL::CaptureManager::sharedCaptureManager();
  manager->stopCapture();
}

void reset_dispatch_counter() {
  auto stream = default_stream(mlx::core::Device::gpu);
  auto& encoder = metal::get_command_encoder(stream);
  encoder.reset_dispatch_counter();
}

uint64_t total_dispatches() {
  auto stream = default_stream(mlx::core::Device::gpu);
  auto& encoder = metal::get_command_encoder(stream);
  return encoder.total_dispatches();
}

} // namespace mlx::core::metal
