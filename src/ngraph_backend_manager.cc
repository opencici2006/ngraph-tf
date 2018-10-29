/*******************************************************************************
 * Copyright 2018 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#include "ngraph_backend_manager.h"

#include <iostream>

namespace tensorflow {
namespace ngraph_bridge {

BackendManager* BackendManager::Instance() {
  static BackendManager instance;
  return &instance;
}

void BackendManager::SetBackendConfig(const std::string& config) {
  std::lock_guard<std::mutex> lock{mu_};
  LoadBackendLocked(config);
}

std::tuple<std::shared_ptr<ngraph::runtime::Backend>, bool>
BackendManager::GetBackend() {
  std::lock_guard<std::mutex> lock{mu_};

  if (!backend_) {
    const char* config = std::getenv("NGRAPH_TF_BACKEND");
    if (!config) {
      config = "CPU";
    }
    LoadBackendLocked(config);
  }

  // nGraph backend configuration strings start with the library name,
  // optionally followed by a ':' and a comma-separated sequence of
  // attributes.  So to discover whether we're using the CPU backend,
  // we match the config up to the first ':' (if any).
  auto colon = config_.find(':');
  bool is_cpu;
  if (colon == std::string::npos) {
    is_cpu = config_ == "CPU";
  } else {
    is_cpu = config_.substr(0, colon) == "CPU";
  }

  return std::make_tuple(backend_, is_cpu);
}

void BackendManager::LoadBackendLocked(std::string config) {
  backend_ = ngraph::runtime::Backend::create(config);
  config_ = std::move(config);
}
}
}
