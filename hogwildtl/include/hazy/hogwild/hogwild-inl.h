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

namespace hazy {
namespace hogwild {

template <class Model, class Params, class Exec>
template <class Scan>
double Hogwild<Model, Params, Exec>::UpdateModel(Scan &scan) {
  scan.Reset();
  Zero();
  // train_time_.Start();
  // epoch_time_.Start();
  FFAScan(model_, params_, scan, tpool_, Exec::UpdateModel, res_, train_time_, epoch_time_);
  // epoch_time_.Stop();
  // train_time_.Pause();
  double time = 0;
  for (unsigned i = 0; i < tpool_.ThreadCount(); i++) {
      time = std::max(time, res_.values[i]);
  }
  return time;
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

  template<class Model, class Example, class Scan>
  void CalcF1Score(const Model& model, int (* hook)(const Example& e, const Model& model), Scan& scan, int& tp, int& tn, int& fp, int& fn) {
      scan.Reset();
      while (scan.HasNext()) {
          ExampleBlock<Example> &ex = scan.Next();
          vector::FVector<Example> vec = ex.ex;
          for (int i = 0; i < vec.size; ++i) {
              bool correct = hook(vec[i], model) == 1;
              bool positive = vec[i].value > 0;
              if (correct) {
                  if (positive) tp++; else tn++;
              } else {
                  if (positive) fn++; else fp++;
              }
          }
      }
  }

  template<class Model, class Params, class Exec>
  template<class Scan>
  double Hogwild<Model, Params, Exec>::ComputeF1Score(Scan& scan) {
      int tp = 0, tn = 0, fp = 0, fn = 0;
      CalcF1Score(this->model_, Exec::ComputeAccuracy, scan, tp, tn, fp, fn);
      double precision = double(tp) / (tp + fp);
      double recall = double(tp) / (tp + fn);
      return 2 * precision * recall / (precision + recall);
  }

template <class Model, class Params, class Exec>
template <class TrainScan, class TestScan>
bool Hogwild<Model, Params, Exec>::RunExperiment(
    int nepochs, hazy::util::Clock &wall_clock, 
    TrainScan &trscan, TestScan &tescan, double target_accuracy) {
  printf("wall_clock: %.5f    Going Hogwild!\n", wall_clock.Read());
  bool stop = false;
  double time_s = 0.0;
  int epoch = 0;
  for (int e = 1; e <= nepochs; e++) {
    double epoch_time = UpdateModel(trscan);
      double f1_train = ComputeF1Score(tescan);
      double f1_test = ComputeF1Score(tescan);
    Exec::PostEpoch(model_, params_);
    Exec::PostUpdate(model_, params_);
/*
    printf("epoch: %d wall_clock: %.5f train_time: %.5f test_time: %.5f epoch_time: %.5f train_rmse: %.5g test_rmse: %.5g\n", 
           e, wall_clock.Read(), train_time_.value, test_time_.value, 
           epoch_time_.value, train_rmse, test_rmse);
*/
   
    printf("epoch: %d wall_clock: %.5f train_time!!!: %.5f epoch_time: %.5f train_acc: %.5g test_acc: %.5g\n",
           e, wall_clock.Read(), train_time_.value,
           epoch_time, f1_train, f1_test);
    fflush(stdout);
    time_s += epoch_time;
    if (f1_test >= target_accuracy) {
      epoch = e;
      stop = true;
      break;
    }
  }
  if (stop) {
    printf("threads: %d epoch: %d train_time: %.5f\n", tpool_.ThreadCount(), epoch, time_s);
    fflush(stdout);
  }
  return stop;
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


