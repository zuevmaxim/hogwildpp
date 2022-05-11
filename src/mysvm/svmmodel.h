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

#ifndef MY_SVM_MODEL_H
#define MY_SVM_MODEL_H

#include "hazy/util/simple_random-inl.h"
#include "hazy/vector/fvector.h"
#include "hazy/vector/svector.h"
#include "hazy/vector/operations-inl.h"

#include "hazy/hogwild/hogwild_task.h"
#include "hazy/thread/thread_pool.h"
#include "../numasvm/svmmodel.h"

#include <cstdio>
#include <cstdlib>

namespace hazy {
namespace hogwild {
//! Sparse SVM implementation
namespace svm {

//! The mutable model for a sparse SVM.
struct MyNumaSVMModel {
  //! The weight vector that is trained
  vector::FVector<fp_type> weights;
  int id;
  int next_id;
  int * has_synced;
  int * lock;
  int * owner;
  vector::FVector<int> peers;
  int update_atomic_counter;
  int * thread_to_weights_mapping;


  //! Construct a weight vector of length dim backed by the buffer
  /*! A new model backed by the buffer.
   * \param buf the backing memory for the weight vector
   * \param dim the length of buf
   */
  explicit MyNumaSVMModel() {
    update_atomic_counter = -1;
  }

  void AllocateModel(unsigned dim) {
    has_synced = new int(0);
    lock = new int(0);
    owner = new int();
    weights.size = dim;
    weights.values = new fp_type[dim];
    for (unsigned i = dim; i-- > 0; ) {
      weights.values[i] = 0;
    }
  }

  void MirrorModel(MyNumaSVMModel const &m) {
    has_synced = m.has_synced;
    lock = m.lock;
    owner = m.owner;
    weights.size = m.weights.size;
    weights.values = m.weights.values;
    peers.size = m.peers.size;
    peers.values = m.peers.values;
  }

  inline bool IsOwner() {
    return *owner == id;
  }

  inline void SetNextOwner() {
    *owner = next_id;
  }

  inline bool HasSynced() {
      return *has_synced == 1;
  }

  inline void SetSynced(bool value) {
      *has_synced = value;
  }

  inline bool TryLock() {
      return __sync_bool_compare_and_swap(lock, 0, 1);
  }

  inline void Unlock() {
      __sync_bool_compare_and_swap(lock, 1, 0);
  }

  inline bool IsLocked() {
    return *lock == 1;
  }

  inline int RandomPeer() const {
    if (peers.size == 0) return -1;
    return peers.values[util::SimpleRandom::GetInstance().RandInt(peers.size)];
  }
};

typedef HogwildTask<MyNumaSVMModel, SVMParams, SVMExample> MySVMTask;

} // namespace svm
} // namespace hogwild

} // namespace hazy

#endif
