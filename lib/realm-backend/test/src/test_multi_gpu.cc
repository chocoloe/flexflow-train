#include "kernels/managed_ff_stream.h"
#include "kernels/managed_per_device_ff_handle.h"
#include "local-execution/allocated_tensors.h"
#include "pcg/computation_graph.h"
#include "pcg/computation_graph_builder.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "realm-backend/driver.h"
#include "realm-backend/realm_allocator.h"
#include "realm-backend/realm_training_backing.h"
#include "test_utils.h"

using namespace ::FlexFlow;
using namespace Realm;

void top_level_task(const void *args, size_t arglen, const void *userdata,
                    size_t userlen, Realm::Processor p) {
  // initialize runtime configs
  ManagedFFStream managed_stream{};
  ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle();

  // Query all available processors and create allocators
  std::vector<Processor> worker_procs;
  std::vector<Allocator> allocators;
  Machine::ProcessorQuery pq = Machine::ProcessorQuery(Machine::get_machine())
                                   .only_kind(Processor::TOC_PROC);
  
  if (pq.count() == 0) {
    printf("No GPU processors found!\n");
    return;
  }

  printf("Found %zu GPU processors for multi-GPU training\n", pq.count());

  for (Processor proc : pq) {
    worker_procs.push_back(proc);
    allocators.push_back(create_realm_memory_allocator(proc));
  }

  AllocatedTensors allocated_tensors = make_empty_allocated_tensors();

  // construct computation graph
  ComputationGraph computation_graph = make_empty_computation_graph();

  nonnegative_int batch_size = 10_n;
  nonnegative_int data_dim = 16_n;
  nonnegative_int hidden_dim = 32_n;
  nonnegative_int output_dim = 64_n;

  // Create a deeper network to benefit from multi-GPU
  TensorShape input_tensor_shape =
      TensorShape{TensorDims{FFOrdered<nonnegative_int>{batch_size, data_dim}},
                  DataType::FLOAT};

  TensorShape weight_shape_1 =
      TensorShape{TensorDims{FFOrdered<nonnegative_int>{data_dim, hidden_dim}},
                  DataType::FLOAT};
  TensorShape weight_shape_2 =
      TensorShape{TensorDims{FFOrdered<nonnegative_int>{hidden_dim, hidden_dim}},
                  DataType::FLOAT};
  TensorShape weight_shape_3 =
      TensorShape{TensorDims{FFOrdered<nonnegative_int>{hidden_dim, output_dim}},
                  DataType::FLOAT};

  // Add layers to the computation graph
  LayerAddedResult inputs_layer =
      add_input_layer(computation_graph, input_tensor_shape);
  tensor_guid_t input_tensor_guid = get_only(inputs_layer.outputs);

  LayerAddedResult dense_layer_1 = add_dense_layer(
      computation_graph, input_tensor_guid, /*out_channels=*/hidden_dim,
      /*activation=*/Activation::RELU, /*bias=*/true);
  tensor_guid_t dense_1_output = get_only(dense_layer_1.outputs);

  LayerAddedResult dense_layer_2 = add_dense_layer(
      computation_graph, dense_1_output, /*out_channels=*/hidden_dim,
      /*activation=*/Activation::RELU, /*bias=*/true);
  tensor_guid_t dense_2_output = get_only(dense_layer_2.outputs);

  LayerAddedResult dense_layer_3 = add_dense_layer(
      computation_graph, dense_2_output, /*out_channels=*/output_dim,
      /*activation=*/Activation::NONE, /*bias=*/true);
  tensor_guid_t dense_3_output = get_only(dense_layer_3.outputs);

  // Create training backing with multi-GPU support
  RuntimeArgConfig runtime_arg_config = RuntimeArgConfig{
      DeviceSpecific<PerDeviceFFHandle>::create(managed_handle.raw_handle()),
      EnableProfiling::YES,
      ProfilingSettings{/*warmup_iters=*/0, /*measure_iters=*/1}};

  OptimizerAttrs optimizer_attrs =
      OptimizerAttrs{SGDOptimizerAttrs{/*lr=*/0.001,
                                       /*momentum=*/0.9,
                                       /*nesterov=*/false,
                                       /*weight_decay=*/0.001}};

  GradientTensorSource gradient_tensor_source;
  OptimizerTensorSource optimizer_tensor_source;

  // Create RealmTrainingBacking with multi-GPU support
  RealmTrainingBacking realm_training_backing = RealmTrainingBacking{
      p,  // master processor
      worker_procs,  // all GPU processors
      allocators,    // all GPU allocators
      allocated_tensors,
      gradient_tensor_source,
      optimizer_tensor_source,
      computation_graph,
      runtime_arg_config,
      optimizer_attrs
  };

  printf("Multi-GPU RealmTrainingBacking created successfully with %d devices\n",
         realm_training_backing.multi_gpu_resources.get_device_count());

  // Print device assignments
  for (layer_guid_t const &layer_guid : topological_ordering(computation_graph)) {
    try {
      int device_id = realm_training_backing.multi_gpu_resources.get_device_id_for_layer(layer_guid);
      printf("Layer %s assigned to device %d\n", 
             fmt::format("{}", layer_guid).c_str(), device_id);
    } catch (const std::exception& e) {
      printf("Layer %s not assigned to any device\n", 
             fmt::format("{}", layer_guid).c_str());
    }
  }

  // Test a simple forward pass
  printf("Testing multi-GPU forward pass...\n");
  std::vector<Future<float>> forward_futures;
  for (layer_guid_t const &layer_guid : topological_ordering(computation_graph)) {
    Future<float> future = execute_forward(realm_training_backing, layer_guid);
    forward_futures.push_back(future);
  }

  // Wait for all forward passes to complete
  for (Future<float> &future : forward_futures) {
    float result = future.get();
    printf("Forward pass completed with result: %f\n", result);
  }

  printf("Multi-GPU test completed successfully!\n");
} 