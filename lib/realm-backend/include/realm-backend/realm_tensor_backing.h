
#ifndef _FLEXFLOW_REALM_BACKEND_REALM_TENSOR_BACKING_H
#define _FLEXFLOW_REALM_BACKEND_REALM_TENSOR_BACKING_H

#include "realm-backend/realm_tensor_backing.struct.dtg.h"
#include "task-spec/task_binding.dtg.h"
#include "task-spec/task_argument_accessor.h"
#include "task-spec/tensor_type_t.dtg.h"
#include "local-execution/allocated_tensors.h"
#include "local-execution/unallocated_tensors.h"
#include "kernels/allocation.h"

namespace FlexFlow {

GenericTensorAccessorW get_tensor(RealmTensorBacking const &, TensorTypeVariant const &);

std::unordered_map<tensor_guid_t, std::vector<optimizer_tensor_t>>
merge_optimizer_mappings(
    std::unordered_map<tensor_guid_t, std::vector<optimizer_tensor_t>> const &,
    std::unordered_map<tensor_guid_t, std::vector<optimizer_tensor_t>> const &);

std::unordered_map<TensorTypeVariant, GenericTensorAccessorW>
get_tensor_backings(
    std::unordered_map<TensorTypeVariant, GenericTensorAccessorW> const &,
    std::unordered_map<TensorTypeVariant, TensorShape> const &,
    Allocator &);

// Multi-GPU version
std::unordered_map<TensorTypeVariant, GenericTensorAccessorW>
get_tensor_backings_multi_gpu(
    std::unordered_map<TensorTypeVariant, GenericTensorAccessorW> const &,
    std::unordered_map<TensorTypeVariant, TensorShape> const &,
    std::vector<Allocator> const &);

RealmTensorBacking construct_realm_tensor_backing(
    AllocatedTensors const &, UnallocatedTensors const &, Allocator &);

TensorSlotsBacking construct_tensor_slots_backing(
    RealmTensorBacking const &, TaskBinding const &);

} // namespace FlexFlow

#endif