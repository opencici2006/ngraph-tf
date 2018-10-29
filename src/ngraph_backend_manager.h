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

#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <tuple>

#include "ngraph/runtime/backend.hpp"

namespace tensorflow {
namespace ngraph_bridge {

// Manages the nGraph bridge's notion of the current execution backend.
class BackendManager {
 public:
  // Returns the global BackendManager instance.
  //
  // (Note that in testing scenarios, there may be additional instances.)
  static BackendManager* Instance();

  // Sets the manager's current backend.
  //
  // Throws an exception if the supplied configuration string can't be
  // resolved to an nGraph backend -- either if it doesn't match an
  // available backend, or if the loaded backend is unable to process
  // the backend-specific configuration component.
  //
  // If the configuration cannot be resolved, the manager's current
  // backend will remain unchanged.
  void SetBackendConfig(const std::string& config);

  // Gets the manager's current backend.
  //
  // If the backend has not previously been loaded, the implementation
  // will attempt to load it, using the configuration specified in the
  // NGRAPH_TF_BACKEND environment variable, or using "CPU" as a
  // fallback if NGRAPH_TF_BACKEND is not set.
  //
  // The second member of the tuple will be true iff the manager is
  // using the CPU backend.
  std::tuple<std::shared_ptr<ngraph::runtime::Backend>, bool> GetBackend();

 private:
  void LoadBackendLocked(std::string config);

  std::mutex mu_;
  std::string config_;
  std::shared_ptr<ngraph::runtime::Backend> backend_;
};
}
}
