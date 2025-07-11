#ifndef _FLEXFLOW_REALM_BACKEND_REALM_TRAINING_BACKING_H
#define _FLEXFLOW_REALM_BACKEND_REALM_TRAINING_BACKING_H

#include "local-execution/optimizer_tensor_source.h"
#include "local-execution/task_registry.h"
#include "op-attrs/ops/loss_functions/loss_attrs.dtg.h"
#include "pcg/computation_graph.dtg.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "local-execution/allocated_tensors.h"
#include "local-execution/unallocated_tensors.h"
#include "realm-backend/driver.h"
#include "realm-backend/realm_allocator.h"
#include "realm-backend/realm_args_backing.h"
#include "realm-backend/realm_tensor_backing.h"
#include "realm-backend/task_wrapper.h"

namespace FlexFlow {

using PerLayerElapsedTime =
    std::unordered_map<layer_guid_t, std::optional<float>>;

// Device assignment policy for layers
enum class DeviceAssignmentPolicy {
  ROUND_ROBIN,  // Assign layers to devices in round-robin fashion
  LOAD_BALANCED // Assign layers based on computational load (future extension)
};

// Device assignment information
struct DeviceAssignment {
  layer_guid_t layer_guid;
  int device_id;
  Realm::Processor processor;
  Allocator allocator;
};

// Multi-GPU resource management
struct MultiGPUResources {
  std::vector<Realm::Processor> worker_procs;
  std::vector<Realm::Event> worker_events;
  std::vector<Allocator> allocators;
  std::unordered_map<layer_guid_t, DeviceAssignment> layer_device_assignments;
  DeviceAssignmentPolicy assignment_policy;
  
  MultiGPUResources(std::vector<Realm::Processor> const &procs,
                    std::vector<Allocator> const &allocs,
                    DeviceAssignmentPolicy policy = DeviceAssignmentPolicy::ROUND_ROBIN);
  
  int get_device_count() const;
  DeviceAssignment const &get_device_assignment(layer_guid_t const &layer_guid) const;
  Realm::Processor get_processor_for_layer(layer_guid_t const &layer_guid) const;
  Allocator get_allocator_for_layer(layer_guid_t const &layer_guid) const;
  int get_device_id_for_layer(layer_guid_t const &layer_guid) const;
};

struct RealmTrainingBacking {
  RealmTrainingBacking(Realm::Processor, 
    std::vector<Realm::Processor> const &, 
    std::vector<Allocator> const &,
                      AllocatedTensors const &,
                      GradientTensorSource &,
                       ComputationGraph const &, RuntimeArgConfig const &);

  RealmTrainingBacking(Realm::Processor, 
    std::vector<Realm::Processor> const &, 
    std::vector<Allocator> const &,
    AllocatedTensors const &,
    GradientTensorSource &,
    OptimizerTensorSource &,
                       ComputationGraph const &, RuntimeArgConfig const &,
                       OptimizerAttrs const &);

public:
  // runtime
  Realm::Processor master_proc;
  Realm::Event master_event;
  Realm::Memory master_mem;
  
  // Multi-GPU resources
  MultiGPUResources multi_gpu_resources;

  ComputationGraph computation_graph;
  TaskRegistry task_registry;

  RealmTensorBacking realm_tensor_backing;
  RealmArgsBacking realm_args_backing;

private:
  // Device assignment methods
  void assign_layers_to_devices(ComputationGraph const &computation_graph);
  void assign_layer_round_robin(layer_guid_t const &layer_guid, int &current_device);
};

// Multi-GPU aware task registry and initialization
TaskRegistry construct_task_registry_and_register_tasks_for_realm_multi_gpu(
    ComputationGraph const &, MultiGPUResources const &);

RealmArgsBacking initialize_args_backing_multi_gpu(RealmTrainingBacking *,
                                                   ComputationGraph const &,
                                                   RuntimeArgConfig const &);

// Multi-GPU aware tensor backing construction
RealmTensorBacking construct_realm_tensor_backing_multi_gpu(
    AllocatedTensors const &allocated_tensors,
    UnallocatedTensors const &unallocated_tensors,
    MultiGPUResources const &multi_gpu_resources);

// Task execution methods (updated for multi-GPU)
void execute_init(RealmTrainingBacking &, layer_guid_t const &);
Future<float> execute_forward(RealmTrainingBacking &,
                              layer_guid_t const &);
Future<float> execute_backward(RealmTrainingBacking &,
                              layer_guid_t const &);
Future<void> compute_loss(RealmTrainingBacking &, LossAttrs const &,
                          tensor_guid_t const &logit_tensor,
                          loss_tensor_t const &label_tensor);
Future<void> execute_update(RealmTrainingBacking &, layer_guid_t const &,
                            OptimizerAttrs const &);

// Multi-GPU aware task argument accessor
TaskArgumentAccessor get_task_arg_accessor_multi_gpu(RealmTensorBacking const &,
                                                     RealmArgsBacking const &,
                                                     TaskInvocation const &,
                                                     MultiGPUResources const &,
                                                     layer_guid_t const &);

// Legacy single-GPU functions (kept for backward compatibility)
TaskRegistry construct_task_registry_and_register_tasks_for_realm(
    ComputationGraph const &, std::vector<Realm::Processor> const &);

RealmArgsBacking initialize_args_backing(RealmTrainingBacking *,
                                        ComputationGraph const &,
                                        RuntimeArgConfig const &);

TaskArgumentAccessor get_task_arg_accessor(RealmTensorBacking const &,
                                           RealmArgsBacking const &,
                                           TaskInvocation const &,
                                           Allocator &);

} // namespace FlexFlow

#endif
