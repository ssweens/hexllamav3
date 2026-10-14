"""ROCm half of :mod:`exllamav3.model.hetero`.

This module is executed by the ROCm virtualenv, never imported into the CUDA
parent.  Keep messages small: tensors are descriptors into HeteroSharedArena.
"""
from __future__ import annotations

import argparse
import traceback
from multiprocessing.connection import Connection

import torch

from .hetero_shared import HeteroSharedArena


class RocmWorker:
    def __init__(self, conn: Connection, arena_export: dict, device: int):
        if not torch.version.hip:
            raise RuntimeError("hetero ROCm worker requires a ROCm PyTorch build")
        if not torch.cuda.is_available():
            raise RuntimeError("hetero ROCm worker cannot see a HIP device")
        self.conn = conn
        self.device = torch.device(f"cuda:{device}")  # PyTorch's ROCm device spelling
        torch.cuda.set_device(self.device)
        self.arena = HeteroSharedArena(arena_export["size"], arena_export["name"], create=False)
        self.arena.register()  # hipHostRegister(PORTABLE | MAPPED) via ROCm wheel extension
        self.model = None
        self.modules = []
        self.caches: dict[int, object] = {}
        # Parent recurrent state objects own CUDA allocations and cannot cross the
        # virtualenv boundary. This maps their stable (parent-cache-id, slot) identity
        # to a state backed only by this worker's cache layers.
        self.recurrent_states: dict[tuple[int, int], object] = {}
        self.recurrent_checkpoints: dict[object, dict] = {}

    @staticmethod
    def _cache_kwargs(spec: dict) -> dict:
        kwargs = dict(spec.get("kwargs", {}))
        return kwargs

    def alloc_cache(self, message: dict) -> None:
        """Construct the child model and matching child-only cache objects before load."""
        from exllamav3 import Cache, Config, Model
        from exllamav3.cache.fp16 import CacheLayer_fp16
        from exllamav3.cache.quant import CacheLayer_quant

        if self.model is not None:
            raise RuntimeError("alloc_cache may only be called once per worker model")
        config = Config.from_directory(message["model_dir"])
        self.model = Model.from_config(config)
        type_by_name = {"CacheLayer_fp16": CacheLayer_fp16, "CacheLayer_quant": CacheLayer_quant}
        for spec in message.get("caches", []):
            layer_type = type_by_name.get(spec["layer_type"])
            if layer_type is None:
                raise NotImplementedError(f"heterogeneous cache layer type {spec['layer_type']} is unsupported")
            self.caches[spec["parent_id"]] = Cache(
                self.model,
                max_num_tokens=spec["max_num_tokens"],
                layer_type=layer_type,
                max_batch_size=spec["max_batch_size"],
                max_history=spec["max_history"],
                **self._cache_kwargs(spec),
            )

    def load_range(self, message: dict) -> None:
        if self.model is None:
            raise RuntimeError("load_range requires alloc_cache first")
        start = int(message["start_layer"])
        self.modules = []
        seen_block = False
        for entry in self.model.fwd_modules:
            module, _instance, _idx = entry
            if module.layer_idx is not None:
                seen_block = True
            if (module.layer_idx is not None and module.layer_idx >= start) or (seen_block and module.layer_idx is None):
                self.modules.append(entry)
        if not self.modules:
            raise ValueError(f"no worker modules at or after layer {start}")
        self.model._load_single(False, self.device, self.model.config, [m for m, _, _ in self.modules], False)
        self.model.active_devices = [self.device.index]
        self.model.output_device = self.device
        worker_layer_indices = {
            module.layer_idx for module, _, _ in self.modules if module.layer_idx is not None
        }
        for cache in self.caches.values():
            # Cache construction only creates meta recurrent-state buffers. Allocate the
            # states for this worker's layers here; parent-owned CUDA layers stay metadata-only.
            for state_layer in cache.get_all_recurrent_layers().values():
                if state_layer.module.layer_idx in worker_layer_indices and state_layer.device is None:
                    state_layer.alloc(self.device)
            cache.initialized = True

    def _params(self, message: dict) -> dict:
        def import_nested(value):
            if isinstance(value, list):
                return [import_nested(v) for v in value]
            if isinstance(value, dict):
                if "__exl3_hetero_tensor__" in value:
                    return self.arena.get(value["__exl3_hetero_tensor__"], self.device)
                if "__exl3_hetero_tuple__" in value:
                    return tuple(import_nested(v) for v in value["__exl3_hetero_tuple__"])
                return {k: import_nested(v) for k, v in value.items()}
            return value

        params = {name: import_nested(value) for name, value in message["params"].items()}
        for name, descriptor in message.get("tensor_params", {}).items():
            params[name] = self.arena.get(descriptor, self.device)
        cache_id = message.get("cache_id")
        if cache_id is not None:
            try:
                params["cache"] = self.caches[cache_id]
            except KeyError as exc:
                raise RuntimeError("worker has no matching cache; create Cache before model.load_heterogeneous") from exc
        return params

    def _prepare_recurrent_states(self, specs: list[dict] | None, params: dict, x: torch.Tensor) -> torch.Tensor | None:
        if specs is None:
            return None
        from exllamav3.cache.recurrent_util import prepare_for_recurrence

        states = []
        for spec in specs:
            cache_id, slot, position = spec["cache_id"], spec["slot"], spec["position"]
            try:
                cache = self.caches[cache_id]
            except KeyError as exc:
                raise RuntimeError("worker has no matching cache for recurrent state") from exc
            key = (cache_id, slot)
            state = self.recurrent_states.get(key)
            if state is None or (position == 0 and state.position != 0):
                if position != 0:
                    raise RuntimeError(
                        "worker first saw a recurrent state after position zero; "
                        "replay its prefill through the heterogeneous model"
                    )
                state = self.model.recurrent_state_cls(cache, slot, 0, clear=True)
                self.recurrent_states[key] = state
            elif state.position > position:
                # Generator speculative rollback changes the parent state before the next
                # call. Replay the same rewind locally rather than moving any state tensor.
                state.rewind(state.position - position)
            elif state.position != position:
                raise RuntimeError(
                    f"heterogeneous recurrent state desynchronized for cache {cache_id}, slot {slot}: "
                    f"worker={state.position}, parent={position}"
                )
            states.append(state)

        params["recurrent_states"] = states
        # prepare_for_recurrence owns recurrent_slots (including the ROCm-device copy),
        # while the values of input IDs are irrelevant to recurrence bookkeeping.
        input_ids = torch.empty(x.shape[:2], dtype=torch.long)
        prepare_for_recurrence(input_ids, params, self.model)
        return input_ids

    def _run(self, message: dict, prefill: bool) -> dict:
        x = self.arena.get(message["x"], self.device)
        params = self._params(message)
        recurrent_input_ids = self._prepare_recurrent_states(message.get("recurrent_states"), params, x)
        for module, instance, idx in self.modules:
            params["layer_instance"] = instance
            if not prefill and module.caps.get("logits_output") and (num := params.get("last_tokens_only")):
                x = x[..., -num:, :].contiguous()
            if prefill:
                stop = (idx, instance) == self.model.last_kv_module_idx_instance
                params["prefill"] = stop
            x = module.prepare_for_device(x, params)
            x = module.forward(x, params)
            if prefill and stop:
                break
        params.pop("prefill", None)
        if recurrent_input_ids is not None:
            from exllamav3.cache.recurrent_util import advance_recurrent_states
            advance_recurrent_states(recurrent_input_ids, params, self.model)
        torch.cuda.synchronize(self.device)
        # Request tensors have been copied to worker-local storage and consumed.
        # Reuse the arena for this response rather than accumulating decode outputs.
        self.arena.reset()
        return {"x": self.arena.put(x) if not prefill else None}

    def forward(self, message: dict) -> dict:
        return self._run(message, prefill=False)

    def prefill(self, message: dict) -> dict:
        self._run(message, prefill=True)
        return {}

    def memory_info(self, _message: dict) -> dict:
        free, total = torch.cuda.mem_get_info(self.device)
        return {
            "free": free,
            "total": total,
            "allocated": torch.cuda.memory_allocated(self.device),
            "reserved": torch.cuda.memory_reserved(self.device),
        }

    def reset_cache(self, _message: dict) -> None:
        for cache in self.caches.values():
            for tensor in cache.get_all_tensors():
                if tensor is not None:
                    tensor.zero_()
        self.recurrent_checkpoints.clear()
        torch.cuda.synchronize(self.device)

    def stash_recurrent(self, message: dict) -> None:
        cache_id, slot, key = message["cache_id"], message["slot"], message["key"]
        try:
            state = self.recurrent_states[(cache_id, slot)]
        except KeyError as exc:
            raise RuntimeError(f"worker has no recurrent state for cache {cache_id}, slot {slot}") from exc
        if state.position <= 0:
            raise RuntimeError("cannot checkpoint an empty worker recurrent state")
        self.recurrent_checkpoints[key] = state.stash()

    def restore_recurrent(self, message: dict) -> None:
        cache_id, slot, key = message["cache_id"], message["slot"], message["key"]
        try:
            stashed = self.recurrent_checkpoints[key]
            cache = self.caches[cache_id]
        except KeyError as exc:
            raise RuntimeError(f"worker recurrent checkpoint {key!r} is unavailable") from exc
        state = self.model.recurrent_state_cls(
            cache, slot, stashed["position"], clear=False, stashed=stashed
        )
        self.recurrent_states[(cache_id, slot)] = state

    def delete_recurrent(self, message: dict) -> None:
        self.recurrent_checkpoints.pop(message["key"], None)

    def copy_cache_page(self, message: dict) -> None:
        cache = self.caches[message["cache_id"]]
        for layer in cache.layers.values():
            # Only layers loaded in this process have backing tensors.  Parent-owned layers
            # exist in the matching Cache metadata but deliberately remain unallocated here.
            tensors = layer.get_tensors()
            if any(tensor is not None for tensor in tensors):
                layer.copy_page(layer, message["from_page"], message["to_page"], message["num_tokens"])
        torch.cuda.synchronize(self.device)

    def smoke_add(self, message: dict) -> dict:
        """Run an actual exllamav3 HIP extension kernel for the no-model-load smoke test."""
        from ..ext import exllamav3_ext as ext
        x = self.arena.get(message["x"], self.device)
        y = torch.empty_like(x)
        ext.add(x, x, y)
        torch.cuda.synchronize(self.device)
        self.arena.reset()
        return {"x": self.arena.put(y)}

    def unload(self, _message: dict) -> None:
        if self.model is not None:
            self.model.unload()
        self.modules = []
        self.caches.clear()
        self.recurrent_states.clear()
        self.recurrent_checkpoints.clear()
        self.model = None

    def close(self) -> None:
        try:
            self.unload({})
        finally:
            self.arena.close()
            self.conn.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fd", required=True, type=int)
    parser.add_argument("--arena-name", required=True)
    parser.add_argument("--arena-size", required=True, type=int)
    parser.add_argument("--device", default=0, type=int)
    args = parser.parse_args()
    worker = RocmWorker(Connection(args.fd), {"name": args.arena_name, "size": args.arena_size}, args.device)
    try:
        while True:
            message = worker.conn.recv()
            command = message.get("command")
            if command == "quit":
                worker.conn.send({"ok": True})
                break
            try:
                handler = getattr(worker, command)
                result = handler(message)
                worker.conn.send({"ok": True, "result": result})
            except Exception as exc:
                worker.conn.send({"ok": False, "error": repr(exc), "traceback": traceback.format_exc()})
    finally:
        worker.close()


if __name__ == "__main__":
    main()
