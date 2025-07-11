#include "kernels/allocation.h"
#include "local-execution/loss_functions.h"
#include "local-execution/optimizer.h"
#include "pcg/computation_graph.dtg.h"
#include "pcg/computation_graph.h"
#include "pcg/optimizer_attrs.h"
#include "realm-backend/realm_tensor_backing.h"
#include "task-spec/op_task_to_task_invocation.h"
#include "task-spec/runtime_arg_config.h"
#include "task-spec/task_invocation.h"
#include "task-spec/task_signature_impl.h"
#include "utils/containers/contains.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/get_only.h"
#include "utils/containers/values.h"
#include "utils/exception.h"
#include "utils/fmt.h"

#include "realm-backend/realm_training_backing.h"
#include "realm-backend/task_result.h"
#include "realm-backend/task_wrapper.h"

namespace FlexFlow {

using namespace Realm;

// ============================================================================
// MultiGPUResources Implementation
// ============================================================================

MultiGPUResources::MultiGPUResources(std::vector<Realm::Processor> const &procs,
                                     std::vector<Allocator> const &allocs,
                                     DeviceAssignmentPolicy policy)
    : worker_procs(procs), 
      worker_events(std::vector<Realm::Event>(procs.size(), Realm::Event::NO_EVENT)),
      allocators(allocs),
      assignment_policy(policy) {
  if (procs.size() != allocs.size()) {
    throw mk_runtime_error("Number of processors and allocators must match");
  }
  if (procs.empty()) {
    throw mk_runtime_error("At least one processor/allocator pair required");
  }
}

int MultiGPUResources::get_device_count() const {
  return worker_procs.size();
}

DeviceAssignment const &MultiGPUResources::get_device_assignment(layer_guid_t const &layer_guid) const {
  if (!contains_key(layer_device_assignments, layer_guid)) {
    throw mk_runtime_error(fmt::format("No device assignment found for layer {}", layer_guid));
  }
  return layer_device_assignments.at(layer_guid);
}

Realm::Processor MultiGPUResources::get_processor_for_layer(layer_guid_t const &layer_guid) const {
  return get_device_assignment(layer_guid).processor;
}

Allocator MultiGPUResources::get_allocator_for_layer(layer_guid_t const &layer_guid) const {
  return get_device_assignment(layer_guid).allocator;
}

int MultiGPUResources::get_device_id_for_layer(layer_guid_t const &layer_guid) const {
  return get_device_assignment(layer_guid).device_id;
}

// ============================================================================
// RealmTrainingBacking Implementation
// ============================================================================

RealmTrainingBacking::RealmTrainingBacking(
    Processor master_proc, std::vector<Processor> const &worker_procs,
    std::vector<Allocator> const &allocators,
    AllocatedTensors const &allocated_tensors,
    GradientTensorSource &gradient_tensor_source,
    ComputationGraph const &computation_graph,
    RuntimeArgConfig const &runtime_arg_config)
    : master_proc(master_proc), master_event(Realm::Event::NO_EVENT),
      master_mem(Machine::MemoryQuery(Machine::get_machine())
                     .only_kind(Memory::SYSTEM_MEM)
                     .best_affinity_to(master_proc)
                     .first()),
      multi_gpu_resources(worker_procs, allocators),
      computation_graph(computation_graph),
      task_registry(construct_task_registry_and_register_tasks_for_realm_multi_gpu(
          computation_graph, multi_gpu_resources)),
      realm_tensor_backing(construct_realm_tensor_backing_multi_gpu(
        allocated_tensors,
        generate_unallocated_tensors(
            allocated_tensors, get_all_tensor_attrs(computation_graph),
            gradient_tensor_source),
        multi_gpu_resources)),
      realm_args_backing(initialize_args_backing_multi_gpu(this, computation_graph, runtime_arg_config)) {
  
  // Assign layers to devices
  assign_layers_to_devices(computation_graph);
}

RealmTrainingBacking::RealmTrainingBacking(
    Processor master_proc, std::vector<Processor> const &worker_procs,
    std::vector<Allocator> const &allocators,
    AllocatedTensors const &allocated_tensors,
    GradientTensorSource &gradient_tensor_source,
    OptimizerTensorSource &optimizer_tensor_source,
    ComputationGraph const &computation_graph,
    RuntimeArgConfig const &runtime_arg_config,
    OptimizerAttrs const &optimizer_attrs)
    : master_proc(master_proc), master_event(Realm::Event::NO_EVENT),
      master_mem(Machine::MemoryQuery(Machine::get_machine())
                     .only_kind(Memory::SYSTEM_MEM)
                     .best_affinity_to(master_proc)
                     .first()),
      multi_gpu_resources(worker_procs, allocators),
      computation_graph(computation_graph),
      task_registry(construct_task_registry_and_register_tasks_for_realm_multi_gpu(
          computation_graph, multi_gpu_resources)),
      realm_tensor_backing(construct_realm_tensor_backing_multi_gpu(
        allocated_tensors,
        generate_unallocated_tensors_with_optimizer(
            allocated_tensors, get_all_tensor_attrs(computation_graph),
            gradient_tensor_source, optimizer_tensor_source,
            optimizer_attrs),
        multi_gpu_resources)),
      realm_args_backing(initialize_args_backing_multi_gpu(this, computation_graph, runtime_arg_config)) {
  
  // Assign layers to devices
  assign_layers_to_devices(computation_graph);
}

void RealmTrainingBacking::assign_layers_to_devices(ComputationGraph const &computation_graph) {
  int current_device = 0;
  std::vector<layer_guid_t> layers = topological_ordering(computation_graph);
  
  for (layer_guid_t const &layer_guid : layers) {
    switch (multi_gpu_resources.assignment_policy) {
      case DeviceAssignmentPolicy::ROUND_ROBIN:
        assign_layer_round_robin(layer_guid, current_device);
        break;
      case DeviceAssignmentPolicy::LOAD_BALANCED:
        // For now, fall back to round-robin. Load balancing can be implemented later
        assign_layer_round_robin(layer_guid, current_device);
        break;
    }
  }
}

void RealmTrainingBacking::assign_layer_round_robin(layer_guid_t const &layer_guid, int &current_device) {
  int device_id = current_device % multi_gpu_resources.get_device_count();
  DeviceAssignment assignment = {
    layer_guid,
    device_id,
    multi_gpu_resources.worker_procs[device_id],
    multi_gpu_resources.allocators[device_id]
  };
  
  multi_gpu_resources.layer_device_assignments[layer_guid] = assignment;
  current_device++;
}

// ============================================================================
// Multi-GPU Task Registry and Initialization
// ============================================================================

TaskRegistry construct_task_registry_and_register_tasks_for_realm_multi_gpu(
    ComputationGraph const &cg, MultiGPUResources const &multi_gpu_resources) {
  TaskRegistry task_registry = construct_task_registry(
    get_layer_attrs_mapping(cg));

  // Register tasks for all devices
  std::unordered_map<layer_guid_t, LayerAttrs> const &layer_attrs_mapping =
      get_layer_attrs_mapping(cg);
  for (std::pair<layer_guid_t, LayerAttrs> const &layer_attrs :
      layer_attrs_mapping) {
    ComputationGraphOpAttrs attrs = layer_attrs.second.op_attrs;
    std::vector<task_id_t> task_ids = get_task_ids(attrs);
    
    for (task_id_t task_id : task_ids) {
      TaskSignatureAndImpl task_signature_impl = get_task_sig_impl(task_id);
      
      // Register tasks on all devices (they may be needed on any device)
      for (int device_id = 0; device_id < multi_gpu_resources.get_device_count(); device_id++) {
        register_wrapper_tasks(device_id, multi_gpu_resources.worker_procs[device_id], 
                               task_id, task_signature_impl);
      }
    }
  }

  return task_registry;
}

RealmArgsBacking initialize_args_backing_multi_gpu(RealmTrainingBacking *backing,
                                                   ComputationGraph const &cg,
                                                   RuntimeArgConfig const &runtime_arg_config) {
  std::unordered_map<layer_guid_t, DeviceSpecificDeviceStates> per_device_op_states;
  TaskRegistry const &task_registry = backing->task_registry;
  RealmTensorBacking const &realm_tensor_backing = backing->realm_tensor_backing;
  MultiGPUResources const &multi_gpu_resources = backing->multi_gpu_resources;

  for (layer_guid_t const &node : topological_ordering(cg)) {
    if (registry_contains_task_for_layer(task_registry, node, OpTaskType::INIT)) {
      ComputationGraphOpAttrs attrs = get_layer_attrs(cg, node).op_attrs;

      TaskInvocation invocation = lower_to_task_invocation(
          init(attrs), node, get_incoming_inputs(cg, node),
          get_incoming_input_shapes(cg, node), get_outgoing_tensors(cg, node),
          get_incoming_weights(cg, node),
          realm_tensor_backing.tensor_gradient_mapping, std::nullopt);
      
      // Use the assigned device for this layer
      int device_id = multi_gpu_resources.get_device_id_for_layer(node);
      Processor assigned_processor = multi_gpu_resources.get_processor_for_layer(node);
      Allocator assigned_allocator = multi_gpu_resources.get_allocator_for_layer(node);

      TaskArgumentAccessor accessor = get_task_arg_accessor_multi_gpu(
          realm_tensor_backing,
          make_args_backing_with_empty_device_states(runtime_arg_config),
          invocation,
          multi_gpu_resources,
          node);
      
      task_id_t task_id = invocation.task_id;
      TaskImplFunction impl_function = task_registry.task_mapping.at(task_id).impl_function;
      
      Promise<DeviceSpecificDeviceStates> promise = Promise<DeviceSpecificDeviceStates>();
      Future<DeviceSpecificDeviceStates> future = promise.get_future();
      RealmTaskArgs<DeviceSpecificDeviceStates>* task_arg = new RealmTaskArgs<DeviceSpecificDeviceStates>{
          task_id, impl_function, accessor, std::move(promise)};
      uintptr_t args[1] = {reinterpret_cast<uintptr_t>(task_arg)};
      
      Event e = assigned_processor.spawn(get_realm_task_id(task_id),
                                        args, sizeof(uintptr_t), 
                                        backing->multi_gpu_resources.worker_events[device_id]);
      backing->multi_gpu_resources.worker_events[device_id] = e;
      future.set_event(e);
      per_device_op_states.insert({node, future.get().value()});
    }
  }

  return RealmArgsBacking{runtime_arg_config, per_device_op_states};
}

// ============================================================================
// Multi-GPU Tensor Backing Construction
// ============================================================================

RealmTensorBacking construct_realm_tensor_backing_multi_gpu(
    AllocatedTensors const &allocated_tensors,
    UnallocatedTensors const &unallocated_tensors,
    MultiGPUResources const &multi_gpu_resources) {

  std::unordered_map<tensor_guid_t, gradient_tensor_t> merged_gradient_maps =
      allocated_tensors.gradient_mapping;
  merged_gradient_maps.insert(unallocated_tensors.gradient_mapping.begin(),
                              unallocated_tensors.gradient_mapping.end());

  // For now, use the first allocator for all tensors
  // In the future, this should be enhanced to distribute tensors across devices
  // based on which layers use them
  std::unordered_map<TensorTypeVariant, GenericTensorAccessorW> all_tensor_backings = 
      allocated_tensors.tensor_type_backings;

  // Allocate unallocated tensors using round-robin allocation across devices
  int current_device = 0;
  for (std::pair<TensorTypeVariant, TensorShape> const &tensor_type_shape :
       unallocated_tensors.tensor_type_shapes) {
    int device_id = current_device % multi_gpu_resources.get_device_count();
    GenericTensorAccessorW tensor_backing =
        multi_gpu_resources.allocators[device_id].allocate_tensor(tensor_type_shape.second);
    all_tensor_backings.insert({tensor_type_shape.first, tensor_backing});
    current_device++;
  }

  return RealmTensorBacking{
      all_tensor_backings,
      merged_gradient_maps,
      merge_optimizer_mappings(allocated_tensors.optimizer_mapping,
                               unallocated_tensors.optimizer_mapping)};
}

// ============================================================================
// Multi-GPU Task Execution
// ============================================================================

Future<float> execute_forward(RealmTrainingBacking &realm_training_backing,
                              layer_guid_t const &operator_node) {
  if (registry_contains_task_for_layer(realm_training_backing.task_registry,
                                       operator_node, OpTaskType::FWD)) {
    ComputationGraphOpAttrs attrs =
        get_layer_attrs(realm_training_backing.computation_graph, operator_node)
            .op_attrs;
    std::optional<DeviceSpecificDeviceStates> device_state =
        get_per_device_op_state_if_exists(
            realm_training_backing.realm_args_backing, operator_node);
    
    TaskInvocation invocation = lower_to_task_invocation(
        forward(attrs), operator_node,
        get_incoming_inputs(realm_training_backing.computation_graph, operator_node),
        get_incoming_input_shapes(realm_training_backing.computation_graph, operator_node),
        get_outgoing_tensors(realm_training_backing.computation_graph, operator_node),
        get_incoming_weights(realm_training_backing.computation_graph, operator_node),
        realm_training_backing.realm_tensor_backing.tensor_gradient_mapping,
        device_state);
    
    // Use the assigned device for this layer
    int device_id = realm_training_backing.multi_gpu_resources.get_device_id_for_layer(operator_node);
    Processor assigned_processor = realm_training_backing.multi_gpu_resources.get_processor_for_layer(operator_node);
    
    TaskArgumentAccessor accessor = get_task_arg_accessor_multi_gpu(
        realm_training_backing.realm_tensor_backing,
        realm_training_backing.realm_args_backing, invocation,
        realm_training_backing.multi_gpu_resources,
        operator_node);
    
    task_id_t task_id = invocation.task_id;
    TaskImplFunction impl_function =
        realm_training_backing.task_registry.task_mapping.at(task_id).impl_function;
    
    Promise<float> promise(realm_training_backing.master_mem);
    Future<float> future = promise.get_future();
    RealmTaskArgs<float>* task_arg = new RealmTaskArgs<float>{
        task_id, impl_function, accessor, std::move(promise)};
    uintptr_t args[1] = {reinterpret_cast<uintptr_t>(task_arg)};
    
    Event e = assigned_processor.spawn(
        get_realm_task_id(task_id), args, sizeof(uintptr_t),
        realm_training_backing.multi_gpu_resources.worker_events[device_id]);
    realm_training_backing.multi_gpu_resources.worker_events[device_id] = e;
    future.set_event(e);
    return future;
  } else {
    return Future<float>(0.0f);
  }
}

Future<float> execute_backward(RealmTrainingBacking &realm_training_backing,
                               layer_guid_t const &operator_node) {
  if (registry_contains_task_for_layer(realm_training_backing.task_registry,
                                       operator_node, OpTaskType::BWD)) {
    ComputationGraphOpAttrs attrs =
        get_layer_attrs(realm_training_backing.computation_graph, operator_node)
            .op_attrs;
    std::optional<DeviceSpecificDeviceStates> device_state =
        get_per_device_op_state_if_exists(
            realm_training_backing.realm_args_backing, operator_node);
    
    TaskInvocation invocation = lower_to_task_invocation(
        backward(attrs), operator_node,
        get_incoming_inputs(realm_training_backing.computation_graph, operator_node),
        get_incoming_input_shapes(realm_training_backing.computation_graph, operator_node),
        get_outgoing_tensors(realm_training_backing.computation_graph, operator_node),
        get_incoming_weights(realm_training_backing.computation_graph, operator_node),
        realm_training_backing.realm_tensor_backing.tensor_gradient_mapping,
        device_state);
    
    // Use the assigned device for this layer
    int device_id = realm_training_backing.multi_gpu_resources.get_device_id_for_layer(operator_node);
    Processor assigned_processor = realm_training_backing.multi_gpu_resources.get_processor_for_layer(operator_node);
    
    TaskArgumentAccessor accessor = get_task_arg_accessor_multi_gpu(
        realm_training_backing.realm_tensor_backing,
        realm_training_backing.realm_args_backing, invocation,
        realm_training_backing.multi_gpu_resources,
        operator_node);
    
    task_id_t task_id = invocation.task_id;
    TaskImplFunction impl_function =
        realm_training_backing.task_registry.task_mapping.at(task_id).impl_function;
    
    Promise<float> promise(realm_training_backing.master_mem);
    Future<float> future = promise.get_future();
    RealmTaskArgs<float>* task_arg = new RealmTaskArgs<float>{
        task_id, impl_function, accessor, std::move(promise)};
    uintptr_t args[1] = {reinterpret_cast<uintptr_t>(task_arg)};
    
    Event e = assigned_processor.spawn(
        get_realm_task_id(task_id), args, sizeof(uintptr_t),
        realm_training_backing.multi_gpu_resources.worker_events[device_id]);
    realm_training_backing.multi_gpu_resources.worker_events[device_id] = e;
    future.set_event(e);
    return future;
  } else {
    return Future<float>(0.0f);
  }
}

Future<void> execute_update(RealmTrainingBacking &realm_training_backing,
                            layer_guid_t const &node,
                            OptimizerAttrs const &optimizer_attrs) {
  LayerAttrs layer_attrs =
      get_layer_attrs(realm_training_backing.computation_graph, node);
  if (layer_attrs.op_attrs.has<WeightAttrs>()) {
    // get tensors
    tensor_guid_t weight_tensor = get_only(
        get_outgoing_tensors(realm_training_backing.computation_graph, node));

    gradient_tensor_t weight_grad_tensor =
        realm_training_backing.realm_tensor_backing.tensor_gradient_mapping.at(
            weight_tensor);
    std::vector<optimizer_tensor_t> optimizer_buffer_tensors =
        realm_training_backing.realm_tensor_backing.tensor_optimizer_mapping.at(
            weight_tensor);

    // get invocation
    TaskInvocation invocation =
        get_update_invocation(optimizer_attrs, weight_tensor,
                              weight_grad_tensor, optimizer_buffer_tensors);

    // Use the assigned device for this layer
    int device_id = realm_training_backing.multi_gpu_resources.get_device_id_for_layer(node);
    Processor assigned_processor = realm_training_backing.multi_gpu_resources.get_processor_for_layer(node);

    TaskArgumentAccessor accessor = get_task_arg_accessor_multi_gpu(
        realm_training_backing.realm_tensor_backing,
        realm_training_backing.realm_args_backing, invocation,
        realm_training_backing.multi_gpu_resources,
        node);
    
    task_id_t task_id = invocation.task_id;
    register_wrapper_tasks_generic(device_id, assigned_processor, task_id);
    TaskImplFunction update_impl_fn = get_update_task_impl(optimizer_attrs);
    
    Promise<void> promise;
    Future<void> future = promise.get_future();
    RealmTaskArgs<void>* task_arg = new RealmTaskArgs<void>{
        task_id, update_impl_fn, accessor, std::move(promise)};
    uintptr_t args[1] = {reinterpret_cast<uintptr_t>(task_arg)};
    
    Event e = assigned_processor.spawn(
        get_realm_task_id(task_id), args, sizeof(uintptr_t),
        realm_training_backing.multi_gpu_resources.worker_events[device_id]);
    realm_training_backing.multi_gpu_resources.worker_events[device_id] = e;
    future.set_event(e);
    return future;
  } else {
    return Future<void>();
  }
}

Future<void> compute_loss(RealmTrainingBacking &realm_training_backing,
                          LossAttrs const &loss_attrs,
                          tensor_guid_t const &logit_tensor,
                          loss_tensor_t const &label_tensor) {
  TaskInvocation loss_invocation = backward(
      loss_attrs, logit_tensor,
      realm_training_backing.realm_tensor_backing.tensor_gradient_mapping.at(
          logit_tensor),
      label_tensor);

  // Loss computation typically happens on the same device as the last layer
  // For now, use the first device (this can be optimized later)
  int device_id = 0;
  Processor assigned_processor = realm_training_backing.multi_gpu_resources.worker_procs[device_id];
  
  TaskArgumentAccessor loss_accessor = get_task_arg_accessor(
      realm_training_backing.realm_tensor_backing,
      realm_training_backing.realm_args_backing, loss_invocation,
      realm_training_backing.multi_gpu_resources.allocators[device_id]);
  
  task_id_t task_id = loss_invocation.task_id;
  register_wrapper_tasks_generic(device_id, assigned_processor, task_id);
  TaskImplFunction loss_impl_fn = get_loss_bwd_task_impl();
  
  Promise<void> promise;
  Future<void> future = promise.get_future();
  RealmTaskArgs<void>* task_arg = new RealmTaskArgs<void>{
      task_id, loss_impl_fn, loss_accessor, std::move(promise)};
  uintptr_t args[1] = {reinterpret_cast<uintptr_t>(task_arg)};
  
  Event e = assigned_processor.spawn(
      get_realm_task_id(task_id), args, sizeof(uintptr_t),
      realm_training_backing.multi_gpu_resources.worker_events[device_id]);
  realm_training_backing.multi_gpu_resources.worker_events[device_id] = e;
  future.set_event(e);
  return future;
}

TaskArgumentAccessor get_task_arg_accessor_multi_gpu(
    RealmTensorBacking const &realm_tensor_backing,
    RealmArgsBacking const &realm_args_backing,
    TaskInvocation const &invocation,
    MultiGPUResources const &multi_gpu_resources,
    layer_guid_t const &layer_guid) {
  
  TensorSlotsBacking tensor_slots_backing =
      construct_tensor_slots_backing(realm_tensor_backing, invocation.binding);
  ArgSlotsBacking arg_slots_backing = construct_arg_slots_backing(
      invocation.binding, realm_args_backing.runtime_arg_config);
  
  // Use the allocator assigned to this layer
  Allocator assigned_allocator = multi_gpu_resources.get_allocator_for_layer(layer_guid);
  
  return TaskArgumentAccessor::create<RealmTaskArgumentAccessor>(
      assigned_allocator, tensor_slots_backing, arg_slots_backing);
}

// ============================================================================
// Legacy Single-GPU Functions (for backward compatibility)
// ============================================================================

TaskRegistry construct_task_registry_and_register_tasks_for_realm(
    ComputationGraph const &cg, std::vector<Realm::Processor> const &worker_procs) {
  TaskRegistry task_registry = construct_task_registry(
    get_layer_attrs_mapping(cg));

  // register tasks for realm
  std::unordered_map<layer_guid_t, LayerAttrs> const &layer_attrs_mapping =
      get_layer_attrs_mapping(cg);
  for (std::pair<layer_guid_t, LayerAttrs> const &layer_attrs :
      layer_attrs_mapping) {
    ComputationGraphOpAttrs attrs = layer_attrs.second.op_attrs;
    std::vector<task_id_t> task_ids = get_task_ids(attrs);
    for (task_id_t task_id : task_ids) {
        TaskSignatureAndImpl task_signature_impl = get_task_sig_impl(task_id);
        // Legacy: single gpu
        register_wrapper_tasks(0, worker_procs[0], task_id, task_signature_impl);
    }
  }

  return task_registry;
}

RealmArgsBacking initialize_args_backing(RealmTrainingBacking *backing,
                                        ComputationGraph const &cg,
                                        RuntimeArgConfig const &runtime_arg_config) {
  std::unordered_map<layer_guid_t, DeviceSpecificDeviceStates> per_device_op_states;
  TaskRegistry const &task_registry = backing->task_registry;
  RealmTensorBacking const &realm_tensor_backing = backing->realm_tensor_backing;
  Processor master_proc = backing->master_proc;
  Memory master_mem = backing->master_mem;
  std::vector<Processor> &worker_procs = backing->multi_gpu_resources.worker_procs;
  std::vector<Event> &worker_events = backing->multi_gpu_resources.worker_events;
  // Legacy: single gpu
  Allocator &allocator = backing->multi_gpu_resources.allocators[0];

  for (layer_guid_t const &node : topological_ordering(cg)) {
    if (registry_contains_task_for_layer(task_registry, node, OpTaskType::INIT)) {
      ComputationGraphOpAttrs attrs = get_layer_attrs(cg, node).op_attrs;

      TaskInvocation invocation = lower_to_task_invocation(
          init(attrs), node, get_incoming_inputs(cg, node),
          get_incoming_input_shapes(cg, node), get_outgoing_tensors(cg, node),
          get_incoming_weights(cg, node),
          realm_tensor_backing.tensor_gradient_mapping, std::nullopt);
      TaskArgumentAccessor accessor = get_task_arg_accessor(
          realm_tensor_backing,
          make_args_backing_with_empty_device_states(runtime_arg_config),
          invocation,
          allocator);
      task_id_t task_id = invocation.task_id;
      TaskImplFunction impl_function = task_registry.task_mapping.at(task_id).impl_function;
      // Legacy: single gpu launching
      Promise<DeviceSpecificDeviceStates> promise = Promise<DeviceSpecificDeviceStates>();
      Future<DeviceSpecificDeviceStates> future = promise.get_future();
      RealmTaskArgs<DeviceSpecificDeviceStates>* task_arg = new RealmTaskArgs<DeviceSpecificDeviceStates>{
          task_id, impl_function, accessor, std::move(promise)};
      uintptr_t args[1] = {reinterpret_cast<uintptr_t>(task_arg)};
      Event e = worker_procs[0].spawn(get_realm_task_id(task_id),
                                args, sizeof(uintptr_t), worker_events[0]);
      worker_events[0] = e;
      future.set_event(e);
      per_device_op_states.insert({node, future.get().value()});
    }
  }

  return RealmArgsBacking{runtime_arg_config, per_device_op_states};
}

TaskArgumentAccessor get_task_arg_accessor(RealmTensorBacking const &realm_tensor_backing,
                                           RealmArgsBacking const &realm_args_backing,
                                           TaskInvocation const &invocation,
                                           Allocator &allocator) {
  TensorSlotsBacking tensor_slots_backing =
      construct_tensor_slots_backing(realm_tensor_backing, invocation.binding);
  ArgSlotsBacking arg_slots_backing = construct_arg_slots_backing(
      invocation.binding, realm_args_backing.runtime_arg_config);
  // Legacy: single gpu
  return TaskArgumentAccessor::create<RealmTaskArgumentAccessor>(
      allocator, tensor_slots_backing, arg_slots_backing);
}

} // namespace FlexFlow
