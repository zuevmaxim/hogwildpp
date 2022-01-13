// Copyright 2012 Chris Re, Victor Bittorf
//
 //Licensed under the Apache License, Version 2.0 (the "License");
 //you may not use this file except in compliance with the License.
 //You may obtain a copy of the License at
 //    http://www.apache.org/licenses/LICENSE-2.0
 //Unless required by applicable law or agreed to in writing, software
 //distributed under the License is distributed on an "AS IS" BASIS,
 //WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 //See the License for the specific language governing permissions and
 //limitations under the License.

// The Hazy Project, http://research.cs.wisc.edu/hazy/
// Author : Victor Bittorf (bittorf [at] cs.wisc.edu)

#ifndef HAZY_HOGWILD_HOGWILD_INL_H
#define HAZY_HOGWILD_HOGWILD_INL_H

#include <cmath>
#include <cstdio>

#include "hazy/util/clock.h"
#include "hazy/hogwild/freeforall-inl.h"

// See for documentation
#include "hazy/hogwild/hogwild.h"

#include <functional>
#include <chrono>
#include <future>
#include <cstdio>

namespace {
template<class Model, class Params, class Example>
static double test(Model& model, Params& params, hazy::hogwild::ExampleBlock<Example> *block, double (*hook)(hazy::hogwild::HogwildTask<Model, Params, Example>&)) {
  hazy::hogwild::HogwildTask<Model, Params, Example> task{&model, &params, block};
  return hook(task);
}

void startRepeatedTask(int interval, const std::function<bool()>& func) {
  std::thread([interval, func]() {
    while (true) {
      if (func()) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(interval));
      }
    }).detach();
}

}

namespace hazy {
namespace hogwild {

template <class Model, class Params, class Exec>
template <class Scan>
void Hogwild<Model, Params, Exec>::UpdateModel(Scan &scan) {
  scan.Reset();
  Zero();
  // train_time_.Start();
  // epoch_time_.Start();
  FFAScan(model_, params_, scan, tpool_, Exec::UpdateModel, res_, train_time_, epoch_time_);
  // epoch_time_.Stop();
  // train_time_.Pause();
}

template <class Model, class Params, class Exec>
template <class Scan>
double Hogwild<Model, Params, Exec>::ComputeRMSE(Scan &scan) {
  scan.Reset();
  Zero();
  test_time_.Start();
  size_t count = FFAScan(model_, params_, scan, 
                         tpool_, Exec::TestModel, res_);
  test_time_.Stop();

  double sum_sqerr = 0;
  for (unsigned i = 0; i < tpool_.ThreadCount(); i++) {
    sum_sqerr += res_.values[i];
  }
  return std::sqrt(sum_sqerr) / std::sqrt(count);
}

template <class Model, class Params, class Exec>
template <class Scan>
double Hogwild<Model, Params, Exec>::ComputeObj(Scan &scan) {
  scan.Reset();
  Zero();
  test_time_.Start();
  size_t count = FFAScan(model_, params_, scan, 
                         tpool_, Exec::ModelObj, res_);
  test_time_.Stop();

  double obj = 0;
  for (unsigned i = 0; i < tpool_.ThreadCount(); i++) {
    obj += res_.values[i];
  }
  return obj;
}

template <class Model, class Params, class Exec>
template <class Scan>
double Hogwild<Model, Params, Exec>::ComputeAccuracy(Scan &scan) {
  scan.Reset();
  Zero();
  test_time_.Start();
  size_t count = FFAScan(model_, params_, scan, 
                         tpool_, Exec::ModelAccuracy, res_);
  test_time_.Stop();

  double obj = 0;
  for (unsigned i = 0; i < tpool_.ThreadCount(); i++) {
    obj += res_.values[i];
  }
  return obj / count;
}

template <class Model, class Params, class Exec>
template <class TrainScan, class TestScan>
void Hogwild<Model, Params, Exec>::RunExperiment(
    int nepochs, hazy::util::Clock &wall_clock, 
    TrainScan &trscan, TestScan &tescan) {
  printf("wall_clock: %.5f    Going Hogwild!\n", wall_clock.Read());
  bool stop = false;
  double time_s{};
  int epoch{};
  double TARGET_ACC = 0.97713;
//  startRepeatedTask(10, [&]() mutable {
//    double test_acc = test(model_, params_, &tescan.NextWithoutShuffle(), Exec::TotalModelAccuracy);
//    bool result = test_acc >= TARGET_ACC;
//    if (result) {
//      time_s = train_time_.Read();
//      epoch = e;
//      stop = true;
//    }
//    return result;
//  });
  for (int e = 1; e < nepochs; e++) {
//    if (stop) break;
    UpdateModel(trscan);
    double train_rmse = ComputeRMSE(trscan);
    double test_rmse = ComputeRMSE(tescan);
    double obj = ComputeObj(trscan);
    double train_acc = ComputeAccuracy(trscan);
    double test_acc = ComputeAccuracy(tescan);
    Exec::PostEpoch(model_, params_);
    Exec::PostUpdate(model_, params_);
/*
    printf("epoch: %d wall_clock: %.5f train_time: %.5f test_time: %.5f epoch_time: %.5f train_rmse: %.5g test_rmse: %.5g\n", 
           e, wall_clock.Read(), train_time_.value, test_time_.value, 
           epoch_time_.value, train_rmse, test_rmse);
*/
   
    printf("epoch: %d wall_clock: %.5f train_time!!!: %.5f test_time: %.5f epoch_time: %.5f train_rmse: %.5g test_rmse: %.5g obj: %.9g train_acc: %.5g test_acc: %.5g\n",
           e, wall_clock.Read(), train_time_.value, test_time_.value,
           epoch_time_.value, train_rmse, test_rmse, obj, train_acc, test_acc);
    fflush(stdout);

    if (test_acc >= TARGET_ACC) {
      time_s = train_time_.value;
      epoch = e;
      stop = true;
      break;
    }
  }
  if (stop) {
    printf("threads: %d epoch: %d train_time: %.5f\n", tpool_.ThreadCount(), epoch, wall_clock.Read(), time_s);
    fflush(stdout);
  }
}

template <class Model, class Params, class Exec>
template <class TrainScan>
void Hogwild<Model, Params, Exec>::RunExperiment(
    int nepochs, hazy::util::Clock &wall_clock, TrainScan &trscan) {
  printf("wall_clock: %.5f    Going Hogwild!\n", wall_clock.Read());
  for (int e = 1; e <= nepochs; e++) {
    UpdateModel(trscan);
    double train_rmse = ComputeRMSE(trscan);

    printf("epoch: %d wall_clock: %.5f train_time: %.5f test_time: %.5f epoch_time: %.5g train_rmse: %.5g\n", 
           e, wall_clock.Read(), train_time_.value, test_time_.value, 
           epoch_time_.value, train_rmse);
    fflush(stdout);
  }
}

} // namespace hogwild
} // namespace hazy

#endif


