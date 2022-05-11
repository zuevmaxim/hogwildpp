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
#include <cstdlib>
#include <cstring>
#include <set>
#include "../hogwildtl/include/numa.h"

#include "hazy/hogwild/hogwild-inl.h"
#include "hazy/hogwild/numa_memory_scan.h"
#include "hazy/scan/tsvfscan.h"
#include "hazy/scan/binfscan.h"

#include "frontend_util.h"

#include "mysvm/svmmodel.h"
#include "svm/svm_loader.h"
#include "mysvm/svm_exec.h"
#include "../hazytl/include/hazy/thread/thread_pool.h"


// Hazy imports
using namespace hazy;
using namespace hazy::hogwild;
using scan::TSVFileScanner;
using scan::MatlabTSVFileScanner;

using hazy::hogwild::svm::fp_type;


using namespace hazy::hogwild::svm;


template <class Scan>
size_t NumaLoadSVMExamples(Scan &scan, vector::FVector<SVMExample> * nodeex, unsigned nnodes) { 
  size_t nfeats = 0;
#if 0
  for (unsigned i = 0; i < nnodes; ++i) {
    scan.Reset();
    numa_run_on_node(i);
    numa_set_preferred(i);
    nfeats = LoadSVMExamples<Scan>(scan, nodeex[i]);
  }
#else
  numa_run_on_node(0);
  numa_set_preferred(0);
  // The examples on the first node will be loaded from the input file
  nfeats = LoadSVMExamples<Scan>(scan, nodeex[0]);
  // Other nodes need a local copy
  for (unsigned n = 1; n < nnodes; ++n) {
    // scan.Reset();
    numa_run_on_node(n);
    numa_set_preferred(n);
    nodeex[n].size = nodeex[0].size;
    nodeex[n].values = new SVMExample[nodeex[n].size];
    for (size_t i = 0; i < nodeex[n].size; i++) {
      size_t size = nodeex[0].values[i].vector.size;
      nodeex[n].values[i].value = nodeex[0].values[i].value;
      nodeex[n].values[i].vector.size = size;
      nodeex[n].values[i].vector.index = new int[size];
      nodeex[n].values[i].vector.values = new fp_type[size];
      std::memcpy((void *)nodeex[n].values[i].vector.index, nodeex[0].values[i].vector.index, size * sizeof(int));
      std::memcpy((void *)nodeex[n].values[i].vector.values, nodeex[0].values[i].vector.values, size * sizeof(fp_type));
      for (size_t j = 0; j < nodeex[n].values[i].vector.size; j++) {
	// assert(nodeex[n].values[i].vector.index[j] >= 0);
	// assert(nodeex[n].values[i].vector.index[j] <= (long int)nfeats - 1);
      }
    }
  }
#endif
  numa_run_on_node(-1);
  // numa_set_preferred(-1);
  numa_set_localalloc();
  return nfeats;
}


/* this function creates models in a ring manner and groups cluster_size threads into a cluster, which shares a single model.
   the cluster_size variable is the "c" in HogWild++ paper.
*/
int CreateNumaClusterRoundRobinRingSVMModel(MyNumaSVMModel * &node_m, size_t nfeats, hazy::thread::ThreadPool &tpool, unsigned nthreads, unsigned cluster_size, int update_delay) {
  /* determine which w to access for each thread */
  int * thread_to_weights_mapping = new int[nthreads];
  unsigned phycpu_count = tpool.PhyCPUCount();
  int weights_count = nthreads > phycpu_count ? phycpu_count : nthreads;
  int cluster_count = weights_count / cluster_size;
  assert((nthreads % cluster_size == 0) && "Total number of threads must be a multiple of cluster size\n");
  assert((nthreads > phycpu_count ? (phycpu_count % cluster_size == 0) : 1) && 
         "When total number threads is greater than core count, core count must by a multiple of cluster size\n");
  /* weight update policy: Each cluster has a separated model data structure  */
  /* Build the weight update chain */
  for (unsigned i = 0; i < nthreads; ++i) {
    thread_to_weights_mapping[i] = ((i % phycpu_count) % cluster_size) * cluster_count + ((i % phycpu_count) / cluster_size);
//    printf("Thread %d is mapped to model %d\n", i, thread_to_weights_mapping[i]);
  }


 /* Now create the Model array: per-cluster, ring 
  */
  numa_run_on_node(0);
  numa_set_preferred(0);
  node_m = new MyNumaSVMModel[weights_count]; // some of them are just pointers to other weights
//  printf("Model array allocated at %p\n", node_m);
//  PrintNumaMemStats();
  for (int i = 0; i < weights_count; ++i) {
    int thread_id = (i % cluster_count) * cluster_size + (i / cluster_count);
    int node = tpool.GetThreadNodeAffinity(thread_id);
    numa_run_on_node(node);
    numa_set_preferred(node);
    if (i / cluster_count == 0) {
      node_m[i].AllocateModel(nfeats);
      *node_m[i].owner = i;
      node_m[i].peers.size = cluster_count - 1;
      node_m[i].peers.values = new int[cluster_count - 1];
      int k = 0;
      for (int j = 0; j < cluster_count; ++j) {
        if (i == j) continue;
        node_m[i].peers.values[k++] = j;
      }
    }
    else {
      // this thread shares the model
      node_m[i].MirrorModel(node_m[i % cluster_count]);
    }
    node_m[i].id = i;
    node_m[i].next_id = (i + cluster_count) % weights_count;
    node_m[i].thread_to_weights_mapping = thread_to_weights_mapping;
    node_m[i].update_atomic_counter = update_delay;

//    printf("Model %d runs on thread %d. Owner is %d, next is %d \n", node_m[i].id, thread_id, *node_m[i].owner, node_m[i].next_id);
//    printf("Peers (%d) are : ", node_m[i].peers.size);
//    for (int j = 0; j < node_m[i].peers.size; ++j) {
//      printf("%d ", node_m[i].peers.values[j]);
//    }
//    printf("\n");
  }
  numa_run_on_node(-1);
  numa_set_localalloc();
  return weights_count;
}

int main(int argc, char** argv) {
  hazy::util::Clock wall_clock;
  wall_clock.Start();
  //Benchmark::StartExperiment(argc, argv);

  bool matlab_tsv = false;
  bool loadBinary = false;
  unsigned nepochs = 20;
  unsigned nthreads = 1;
  float mu = 1.0, step_size = 5e-2, step_decay = 0.8;
  int cluster_size = 0;
  int update_delay = 256;
  double tolerance = 1e-2;
  double target_accuracy = 1.0;
  static struct extended_option long_options[] = {
    {"mu", required_argument, NULL, 'u', "the maxnorm"},
    {"epochs"    ,required_argument, NULL, 'e', "number of epochs (default is 20)"},
    {"stepinitial",required_argument, NULL, 'i', "intial stepsize (default is 5e-2)"},
    {"step_decay",required_argument, NULL, 'd', "stepsize decay per epoch (default is 0.8)"},
    {"seed", required_argument, NULL, 's', "random seed (o.w. selected by time, 0 is reserved)"},
    {"splits", required_argument, NULL, 'r', "number of threads (default is 1)"},
    //{"shufflers", required_argument, NULL, 'q', "number of shufflers"},
    {"binary", required_argument,NULL, 'v', "load the file in a binary fashion"},
    {"matlab-tsv", required_argument,NULL, 'm', "load TSVs indexing from 1 instead of 0"},
    {"update_delay", required_argument, NULL, 't', "Number of iterations before pass the token to the next thread (default: 256)"},
    {"cluster_size", required_argument, NULL, 'c', "Cluster size (c). Threads in a cluster share the same weights (default: #CPU in one socket)"},
    {"tolerance", required_argument, NULL, 'o', "error tolerance when doing gradient update (default 1e-2)"},
    {"target_accuracy", required_argument,NULL, 'a', "target accuracy to converge"},
    {NULL,0,NULL,0,0} 
  };

  char usage_str[] = "<train file> <test file>";
  int c = 0, option_index = 0;
  option* opt_struct = convert_extended_options(long_options);
  while( (c = getopt_long(argc, argv, "", opt_struct, &option_index)) != -1) 
  {
    switch (c) { 
      case 'v':
        loadBinary = (atoi(optarg) != 0);
        break;
      case 'm':
        matlab_tsv = (atoi(optarg) != 0);
        break;
      case 'u':
        mu = atof(optarg);
        break;
      case 'e':
        nepochs = atoi(optarg);
        break;
      case 'i':
        step_size = atof(optarg);
        break;
      case 'd':
        step_decay = atof(optarg);
        break;
      case 'r':
        nthreads = atoi(optarg);
        break;
      case 't':
        update_delay = atoi(optarg);
        break;
      case 'c':
        cluster_size = atoi(optarg);
        break;
      case 'o':
        tolerance = atof(optarg);
        break;
      case 'a':
        target_accuracy = atof(optarg);
        break;
      case ':':
      case '?':
        print_usage(long_options, argv[0], usage_str);
        exit(-1);
        break;
    }
  }

  char * szTestFile, *szExampleFile;
  
  if(optind == argc - 2) {
    szExampleFile = argv[optind];
    szTestFile  = argv[optind+1];
  } else {
    print_usage(long_options, argv[0], usage_str);
    exit(-1);
  }
  //fp_type buf[50];

  // we initialize thread pool here because we need CPU topology information
  // in HogWild++, the ThreadPool has been improved to assign CPU affinity
  hazy::thread::ThreadPool tpool(nthreads);
  tpool.Init();
  
  unsigned nnodes = tpool.UsedNodeCount();
  printf("%d threads will be running on %d nodes\n", nthreads, nnodes);
  vector::FVector<SVMExample> * node_train_examps = new vector::FVector<SVMExample>[nnodes];
  vector::FVector<SVMExample> * node_test_examps = new vector::FVector<SVMExample>[nnodes];

  size_t nfeats;

  if (loadBinary) {
    printf("Loading binary file...\n");
    scan::BinaryFileScanner scan(szExampleFile);
    nfeats = NumaLoadSVMExamples(scan, node_train_examps, nnodes);
    printf("Loaded binary file!\n");
  } else if (matlab_tsv) {
    MatlabTSVFileScanner scan(szExampleFile);
    nfeats = NumaLoadSVMExamples(scan, node_train_examps, nnodes);
  } else {
    TSVFileScanner scan(szExampleFile);
    nfeats = NumaLoadSVMExamples(scan, node_train_examps, nnodes);
  }
  if (loadBinary) {
    printf("Loading binary file...\n");
    scan::BinaryFileScanner scantest(szTestFile);
    NumaLoadSVMExamples(scantest, node_test_examps, nnodes);
    printf("Loaded binary file!\n");
  } else if (matlab_tsv) {
    MatlabTSVFileScanner scantest(szTestFile);
    NumaLoadSVMExamples(scantest, node_test_examps, nnodes);
  } else {
    TSVFileScanner scantest(szTestFile);
    NumaLoadSVMExamples(scantest, node_test_examps, nnodes);
  }

  unsigned *degs = new unsigned[nfeats];
  printf("Loaded %lu examples\n", nfeats);
  for (size_t i = 0; i < nfeats; i++) {
    degs[i] = 0;
  }
  CountDegrees(node_train_examps[0], degs);

  for (int iteration = 0; iteration < 100; ++iteration) {
    MyNumaSVMModel* node_m;
    int weights_count;
    fp_type beta = 0.0, lambda = 0.5;
    if (cluster_size <= 0) {
        cluster_size = tpool.PhyCPUCount() / tpool.NodeCount();
    }
    weights_count = CreateNumaClusterRoundRobinRingSVMModel(node_m, nfeats, tpool, nthreads, cluster_size,update_delay);
//    PrintWeights(node_m, weights_count, nthreads, tpool);
    SVMParams tp(step_size, step_decay, mu, beta, lambda, weights_count, true, update_delay, tolerance, &tpool);
    tp.degrees = degs;
    tp.ndim = nfeats;

//  hogwild::freeforall::FeedTrainTest(memfeed.GetTrough(), nepochs, nthreads);
    NumaMemoryScan<SVMExample> mscan(node_train_examps, nnodes);
    Hogwild<MyNumaSVMModel, SVMParams, MyNumaSVMExec> hw(node_m[0], tp, tpool);
    NumaMemoryScan<SVMExample> tscan(node_test_examps, nnodes);
    printf("Run experiment: threads=%d c=%d\n", nthreads, cluster_size);
    fflush(stdout);
    hw.RunExperiment(nepochs, wall_clock, mscan, tscan, target_accuracy);
  }
  return 0;
}

