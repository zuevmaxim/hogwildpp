// Copyright 2012 Victor Bittorf, Chris Re
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Hogwild!, part of the Hazy Project
// Author : Victor Bittorf (bittorf [at] cs.wisc.edu)
// Original Hogwild! Author: Chris Re (chrisre [at] cs.wisc.edu)             

#ifndef HAZY_HOGWILD_INSTANCES_SVM_SVM_EXEC_H
#define HAZY_HOGWILD_INSTANCES_SVM_SVM_EXEC_H

#include <cmath>

#include "hazy/hogwild/hogwild_task.h"
#include "hazy/vector/dot-inl.h"
#include "hazy/vector/scale_add-inl.h"
#include "hazy/hogwild/tools-inl.h"
#include "hazy/util/clock.h"

#include <numa.h>
#include <sched.h>
#include <cstdio>

#include "svmmodel.h"
#include "../../hazytl/include/hazy/vector/fvector.h"

namespace hazy {
namespace hogwild {
namespace svm {

//! Changes the model using the given example 
void inline ModelUpdate(const SVMExample& examp, const SVMParams& params,
                        MyNumaSVMModel* model, size_t& updates, size_t& count);

//! Returns the loss for the given example and model
fp_type inline ComputeLoss(const SVMExample& e, const MyNumaSVMModel& model);

//! Container for methods to train and test an SVM
class MyNumaSVMExec {
public:
  /// Preforms updates to the model
  /*! Updates by scanning over examples, uses the thread id and total
   * number of threads to determine which chunk of examples to work on.
   * \param task container of model, params, and examples
   * \param tid the thread ID; 0 <= tid < total
   * \param total the total number of threads working on updating
   */
  static double UpdateModel(MySVMTask& task, unsigned tid, unsigned total);

  /// Compute error of the task's model and the task's examples
  /*! Computes the error of each example with given the model. Uses the
   * number of threads to determine which chunk of examples to work on.
   * TODO XXX Needs to aggregate the RMSE
   * \param task container of model, params, and examples
   * \param tid the thread ID; 0 <= tid < total
   * \param total the total number of threads working on updating
   */
  static double TestModel(MySVMTask& task, unsigned tid, unsigned total);

  //! Invoked after each training epoch, causes the stepsize to decay
  static void PostUpdate(MyNumaSVMModel& model, SVMParams& params);

  static void PostEpoch(MyNumaSVMModel& model, SVMParams& params) {
  }

  static double ModelObj(MySVMTask& task, unsigned tid, unsigned total);

  static double ModelAccuracy(MySVMTask& task, unsigned tid, unsigned total);

private:
  static int GetNumaNode();

  static int GetLatestModel(MySVMTask& task, unsigned tid, unsigned total);
};

} // namespace svm
} // namespace hogwild

} // namespace hazy


namespace hazy {
namespace hogwild {
namespace svm {

fp_type inline ComputeLoss(const SVMExample& e, const MyNumaSVMModel& model) {
  // determine how far off our model is for this example
  vector::FVector <fp_type> const& w = model.weights;
  fp_type dot = vector::Dot(w, e.vector);
  return std::max(1 - dot * e.value, static_cast<fp_type>(0.0));
}

int inline ComputeAccuracy(const SVMExample& e, const MyNumaSVMModel& model) {
  // determine how far off our model is for this example
  vector::FVector <fp_type> const& w = model.weights;
  fp_type dot = vector::Dot(w, e.vector);
  return !!std::max(dot * e.value, static_cast<fp_type>(0.0));
}

  void PerformAveraging(MyNumaSVMModel* model, MyNumaSVMModel* next_model) {
    vector::FVector <fp_type>& w = model->weights;
    fp_type* const vals = w.values;
    fp_type* const next_vals = next_model->weights.values;
    fp_type lambda = 0.5;
    for (unsigned i = 0; i < w.size; ++i) {
      fp_type wi = vals[i];
      fp_type next = next_vals[i];
      fp_type new_wi = next * lambda + wi * (1 - lambda);
      vals[i] += new_wi - wi;
      next_vals[i] += new_wi - next;
    }
  }


bool CheckSync(MyNumaSVMModel* model, MyNumaSVMModel* next_model) {
  MyNumaSVMModel *a = model, *b = next_model;
  if (a > b) std::swap(a, b);
  if (!a->TryLock()) return false;
  if (!b->TryLock()) {
    a->Unlock();
    return false;
  }
  if (model->HasSynced()) {
      model->SetSynced(false);
  } else {
      PerformAveraging(model, next_model);
      next_model->SetSynced(true);
  }
  b->Unlock();
  a->Unlock();
  return true;
}

/* this is the core function, for updating the model */
int inline ModelUpdate(const SVMExample& examp, const SVMParams& params,
                       MyNumaSVMModel* model, MyNumaSVMModel* models, int tid, int weights_index, int iter, int& update_atomic_counter,
                       bool can_sync) {
  int sync_counter = 0;
  vector::FVector <fp_type>& w = model->weights;

  // evaluate this example
  fp_type wxy = vector::Dot(w, examp.vector);
  wxy = wxy * examp.value;

  if (wxy < 1) { // hinge is active.
    fp_type const e = params.step_size * examp.value;
    vector::ScaleAndAdd(w, examp.vector, e);
  }

  fp_type* const vals = w.values;
  unsigned const* const degs = params.degrees;
  size_t const size = examp.vector.size;
  // update based on the evaluation
  fp_type const scalar = params.step_size * params.mu;
  for (int i = size; i-- > 0;) {
    int const j = examp.vector.index[i];
    unsigned const deg = degs[j];
    vals[j] *= (1 - scalar / deg);
  }

  if (can_sync) {
      if (update_atomic_counter < 0 && model->IsOwner()) {
          int peer = model->RandomPeer();
          MyNumaSVMModel* next_model = &models[peer];
          CheckSync(model, next_model);
          model->SetNextOwner();
          update_atomic_counter = params.update_delay * model->cluster_size;
      }
      update_atomic_counter--;
  }
  return sync_counter;
}



void MyNumaSVMExec::PostUpdate(MyNumaSVMModel& model, SVMParams& params) {
  // Reduce the step size to encourage convergence
  params.step_size *= params.step_decay;
  // printf("Step size = %f\n", params.step_size);
}

int MyNumaSVMExec::GetNumaNode() {
  int cpu = sched_getcpu();
  return numa_node_of_cpu(cpu);
}

double MyNumaSVMExec::UpdateModel(MySVMTask& task, unsigned tid, unsigned total) {
  util::Clock clock;
  clock.Start();
  int node = GetNumaNode();
  // TODO: per core model vector
  MyNumaSVMModel& model = *task.model;

  SVMParams const& params = *task.params;
  // Select the example vector array based on current node
  vector::FVector <SVMExample> const& exampsvec = task.block[node].ex;
  // calculate which chunk of examples we work on
  size_t start = hogwild::GetStartIndex(exampsvec.size, tid, total);
  size_t end = hogwild::GetEndIndex(exampsvec.size, tid, total);
  // optimize for const pointers
  // Seclect the pointers based on current node
  size_t* perm = task.block[node].perm.values;
  SVMExample const* const examps = exampsvec.values;
  // individually update the model for each example
  int weights_index = model.thread_to_weights_mapping[tid];
  MyNumaSVMModel* const m = &task.model[weights_index];
  int update_atomic_counter = m->update_atomic_counter;
  int sync_counter = 0;
  bool canSync = m->RandomPeer() != -1;
  for (unsigned i = start; i < end; i++) {
    size_t indirect = perm[i];
    sync_counter += ModelUpdate(examps[indirect], params, m, task.model, tid, weights_index, i - start, update_atomic_counter, canSync);
  }
  // Save states
  m->update_atomic_counter = update_atomic_counter;
  // printf("%d: %d\n", tid, update_atomic_counter);
  // printf("UpdateModel: thread %d, %d/%lu elements copied.\n", tid, sync_counter, model.weights.size);

  return clock.Stop();
}

int MyNumaSVMExec::GetLatestModel(MySVMTask& task, unsigned tid, unsigned total) {
    MyNumaSVMModel* models = task.model;
    SVMParams* params = task.params;
    int max_value = 0;
    int max_index = 0;
    for (int i = 0; i < params->weights_count; ++i) {
        MyNumaSVMModel& model = models[i];
        if (model.HasSynced()) return i;
        if (model.update_atomic_counter > max_value) {
            max_value = model.update_atomic_counter;
            max_index = i;
        }
    }
    return max_index;
}

double MyNumaSVMExec::TestModel(MySVMTask& task, unsigned tid, unsigned total) {
  int node = GetNumaNode();
  MyNumaSVMModel const& model = task.model[GetLatestModel(task, tid, total)];

  //SVMParams const &params = *task.params;
  // Select the example vector array based on current node
  vector::FVector <SVMExample> const& exampsvec = task.block[node].ex;

  // calculate which chunk of examples we work on
  size_t start = hogwild::GetStartIndex(exampsvec.size, tid, total);
  size_t end = hogwild::GetEndIndex(exampsvec.size, tid, total);

  // keep const correctness
  SVMExample const* const examps = exampsvec.values;
  fp_type loss = 0.0;
  // compute the loss for each example
  for (unsigned i = start; i < end; i++) {
    fp_type l = ComputeLoss(examps[i], model);
    loss += l;
  }
  // return the number of examples we used and the sum of the loss
  //counted = end-start;
  return loss;
}

double MyNumaSVMExec::ModelAccuracy(MySVMTask& task, unsigned tid, unsigned total) {
  int node = GetNumaNode();
  MyNumaSVMModel const& model = task.model[GetLatestModel(task, tid, total)];

  //SVMParams const &params = *task.params;
  // Select the example vector array based on current node
  vector::FVector <SVMExample> const& exampsvec = task.block[node].ex;

  // calculate which chunk of examples we work on
  size_t start = hogwild::GetStartIndex(exampsvec.size, tid, total);
  size_t end = hogwild::GetEndIndex(exampsvec.size, tid, total);

  // keep const correctness
  SVMExample const* const examps = exampsvec.values;
  // return the number of examples we used and the sum of the loss
  int correct = 0;
  // compute the loss for each example
  for (unsigned i = start; i < end; i++) {
    int l = ComputeAccuracy(examps[i], model);
    correct += l;
  }
  //counted = end-start;
  return correct;
}

double MyNumaSVMExec::ModelObj(MySVMTask& task, unsigned tid, unsigned total) {
  int node = GetNumaNode();
  MyNumaSVMModel const& model = task.model[GetLatestModel(task, tid, total)];

  //SVMParams const &params = *task.params;
  // Select the example vector array based on current node
  vector::FVector <SVMExample> const& exampsvec = task.block[node].ex;

  // calculate which chunk of examples we work on
  size_t start = hogwild::GetStartIndex(exampsvec.size, tid, total);
  size_t end = hogwild::GetEndIndex(exampsvec.size, tid, total);

  // keep const correctness
  SVMExample const* const examps = exampsvec.values;
  fp_type loss = 0.0;
  // compute the loss for each example
  for (unsigned i = start; i < end; i++) {
    fp_type l = ComputeLoss(examps[i], model);
    loss += l;
  }
  start = hogwild::GetStartIndex(model.weights.size, tid, total);
  end = hogwild::GetEndIndex(model.weights.size, tid, total);
  double const* const weights = model.weights.values;
  fp_type reg = 0.0;
  // compute the regularization term
  for (unsigned i = start; i < end; ++i) {
    reg += weights[i] * weights[i];
  }
  // return the number of examples we used and the sum of the loss
  //counted = end-start;
  return loss + 0.5 * reg;
}


} // namespace svm
} // namespace hogwild

} // namespace hazy


#endif
