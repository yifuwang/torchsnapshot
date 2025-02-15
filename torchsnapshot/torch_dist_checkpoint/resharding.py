# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

import torch
from torch.distributed._shard.sharded_tensor import Shard
from torch.distributed._shard.sharding_spec import ShardMetadata
from torch.distributed._shard.sharding_spec._internals import (
    _check_shard_metadata_pair_overlap,
)

from torchsnapshot.torch_dist_checkpoint.metadata import (
    ExtendedTensorMetadata,
    TensorReadRequest,
)


def _shards_get_overlap_region_wrt_saved_tensor(
    saved_shard: ShardMetadata, current_shard: ShardMetadata
) -> List[Tuple[int, int, int, int]]:
    """
    Return the overlapping region of saved_shard and current_shard on each dimention.
    """
    narrows = []
    for dim, (
        saved_shard_offset,
        current_shard_offset,
        saved_shard_size,
        current_shard_size,
    ) in enumerate(
        zip(
            saved_shard.shard_offsets,
            current_shard.shard_offsets,
            saved_shard.shard_sizes,
            current_shard.shard_sizes,
        )
    ):
        min_range_end = min(
            saved_shard_offset + saved_shard_size,
            current_shard_offset + current_shard_size,
        )

        length = min_range_end - max(current_shard_offset, saved_shard_offset)

        if saved_shard_offset > current_shard_offset:
            offset_for_saved_tensor = 0
            offset_for_current_tensor = saved_shard_offset - current_shard_offset
        else:
            offset_for_saved_tensor = current_shard_offset - saved_shard_offset
            offset_for_current_tensor = 0

        narrows.append(
            (dim, offset_for_saved_tensor, offset_for_current_tensor, length)
        )

    return narrows


def prepare_sharded_tensor_read(
    metadata: ExtendedTensorMetadata, local_shards_out: List[Shard]
) -> List[TensorReadRequest]:
    """
    Prepare sharded tensor read.

    Args:
        metadata: Metadata describing the persisted sharded tensor. Normally,
                  this is generated by func::`prepare_sharded_tensor_write`.
        local_shards_out: The shards of the dest sharded tensor.

    Returns:
        A list of class::`TensorReadRequest`. When fullfilled, All shards in
        `local_shards_out` load from the persisted sharded tensor.
    """
    read_reqs = []
    for shard in local_shards_out:
        # scan all mds looking for chunks
        for storage_md in metadata.storage_metadata:
            shard_md_from_storage = storage_md.shard_metadata
            tensor = shard.tensor.detach()
            assert shard_md_from_storage is not None
            # this is a naive quadratic algo that can later be optimized by sorting metadata and the shards md
            # FIXME what does it mean for offset > 0? just add it to read request offset?
            assert (
                storage_md.offset == 0
            ), f"Storage at key {storage_md.storage_key} is saved with an offset, we cannot load this yet"

            # do they overlap?
            if not _check_shard_metadata_pair_overlap(
                shard.metadata, shard_md_from_storage
            ):
                continue

            storage_key = storage_md.storage_key

            target_tensor = tensor
            offsets = []
            lengths = []
            for (
                dim,
                offset_for_saved_tensor,
                offset_for_current_tensor,
                length,
            ) in _shards_get_overlap_region_wrt_saved_tensor(
                saved_shard=shard_md_from_storage, current_shard=shard.metadata
            ):
                # Note that we do NOT want to make any tensor copy.
                # all operation must be view only
                target_tensor = torch.narrow(
                    target_tensor, dim, offset_for_current_tensor, length
                )
                offsets.append(offset_for_saved_tensor)
                lengths.append(length)

            read_reqs.append(
                TensorReadRequest(
                    tensor=target_tensor,
                    storage_key=storage_key,
                    offsets=tuple(offsets),
                    lengths=tuple(lengths),
                )
            )
    return read_reqs
