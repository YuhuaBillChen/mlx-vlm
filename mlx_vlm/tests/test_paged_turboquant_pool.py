"""Generator-scoped paged TurboQuant storage registry tests."""

import mlx.core as mx
import pytest

from mlx_vlm.paged_turboquant_kernel import PAGED_TURBOQUANT_PAGE_SIZE
from mlx_vlm.paged_turboquant_pool import (
    PagedTurboQuantLayerSpec,
    PagedTurboQuantPoolRegistry,
)


H_KV = 2
D = 256
PAGE = PAGED_TURBOQUANT_PAGE_SIZE


def _kv(length):
    return (
        mx.random.normal((1, H_KV, length, D)).astype(mx.bfloat16),
        mx.random.normal((1, H_KV, length, D)).astype(mx.bfloat16),
    )


def _registry(capacity_pages=8):
    return PagedTurboQuantPoolRegistry(
        {
            0: PagedTurboQuantLayerSpec(capacity_pages, H_KV),
            (3, "attention"): PagedTurboQuantLayerSpec(capacity_pages, H_KV),
        }
    )


def test_registry_eagerly_binds_independent_prompt_caches_to_stable_storage():
    registry = _registry()
    storage = registry.storage_for((3, "attention"))
    payload_ids = tuple(
        id(array) for state in (storage.keys, storage.values) for array in state
    )

    first = registry.new_cache((3, "attention"))
    second = registry.new_cache((3, "attention"))

    assert first.storage is second.storage is storage
    assert first.key_codec is second.key_codec
    assert first.value_codec is second.value_codec
    assert (
        tuple(id(array) for state in (storage.keys, storage.values) for array in state)
        == payload_ids
    )

    first.update_and_fetch(*_kv(PAGE + 3))
    second.update_and_fetch(*_kv(17))
    first_pages = first._rows.rows[0].page_ids
    second_pages = second._rows.rows[0].page_ids
    first.extend(second)

    # Admission changes only row metadata. The physical tensors and both
    # requests' page mappings remain exactly where prefill wrote them.
    assert first.storage is storage
    assert (
        tuple(id(array) for state in (storage.keys, storage.values) for array in state)
        == payload_ids
    )
    assert tuple(row.page_ids for row in first._rows.rows) == (
        first_pages,
        second_pages,
    )
    assert first.sequence_lengths == (PAGE + 3, 17)


def test_cache_sets_preserve_opaque_model_leaf_keys_and_share_per_leaf_only():
    registry = _registry()
    first_prompt = registry.new_cache_set()
    second_prompt = registry.new_cache_set(reversed(registry.leaf_keys))

    assert tuple(first_prompt) == (0, (3, "attention"))
    assert tuple(second_prompt) == ((3, "attention"), 0)
    for leaf_key in registry.leaf_keys:
        assert first_prompt[leaf_key].storage is registry.storage_for(leaf_key)
        assert second_prompt[leaf_key].storage is registry.storage_for(leaf_key)
    assert first_prompt[0].storage is not first_prompt[(3, "attention")].storage

    with pytest.raises(KeyError, match="unknown paged cache leaf"):
        registry.new_cache((99, "attention"))
    before = registry.stats().live_facades
    with pytest.raises(KeyError, match="unknown paged cache leaf"):
        registry.new_cache_set((0, "missing"))
    assert registry.stats().live_facades == before


def test_registry_reports_aggregate_capacity_usage_and_releases_every_row():
    registry = _registry(capacity_pages=4)
    first = registry.new_cache_set()
    second = registry.new_cache_set()
    for caches, length in ((first, PAGE + 1), (second, 1)):
        for cache in caches.values():
            cache.update_and_fetch(*_kv(length))

    stats = registry.stats()
    assert stats.capacity_layer_pages == 8
    assert stats.used_layer_pages == 6
    assert stats.free_layer_pages == 2
    assert stats.used_token_slots == 6 * PAGE
    assert stats.capacity_token_slots == 8 * PAGE
    assert stats.pool_nbytes > 0
    assert stats.live_facades == 4
    assert not stats.closed

    final = registry.release()
    assert final.used_layer_pages == 0
    assert final.free_layer_pages == 8
    assert final.live_facades == 0
    assert final.closed
    with pytest.raises(RuntimeError, match="released"):
        registry.new_cache(0)


def test_registry_enforces_uniform_capacity_and_exact_kernel_geometry():
    with pytest.raises(ValueError, match="same capacity_pages"):
        PagedTurboQuantPoolRegistry(
            {
                0: PagedTurboQuantLayerSpec(4, H_KV),
                1: PagedTurboQuantLayerSpec(5, H_KV),
            }
        )
    with pytest.raises(ValueError, match="page_size=256"):
        PagedTurboQuantLayerSpec(4, H_KV, page_size=128)
    with pytest.raises(ValueError, match="head_dim=256"):
        PagedTurboQuantLayerSpec(4, H_KV, head_dim=128)
    with pytest.raises(ValueError, match="Q4"):
        PagedTurboQuantLayerSpec(4, H_KV, bits=8)
    with pytest.raises(ValueError, match="float16 norms"):
        PagedTurboQuantLayerSpec(4, H_KV, norm_dtype=mx.float32)
    with pytest.raises(ValueError, match="uint32 packed"):
        PagedTurboQuantLayerSpec(4, H_KV, index_dtype=mx.uint16)


def test_registry_restores_packed_apc_row_directly_into_existing_page_pool():
    from mlx_vlm.turboquant import TurboQuantKVCache

    registry = PagedTurboQuantPoolRegistry(
        {0: PagedTurboQuantLayerSpec(4, H_KV)}
    )
    source = TurboQuantKVCache(bits=4)
    source.update_and_fetch(*_kv(PAGE + 7))
    mx.eval(source.state)

    restored = registry.restore_cache_list([source])

    assert len(restored) == 1
    assert restored[0].storage is registry.storage_for(0)
    assert restored[0].sequence_lengths == (PAGE + 7,)
    restored_keys, restored_values = restored[0].materialize(0)
    assert bool(mx.array_equal(restored_keys.norms, source.keys.norms).item())
    assert bool(mx.array_equal(restored_keys.indices, source.keys.indices).item())
    assert bool(mx.array_equal(restored_values.norms, source.values.norms).item())
    assert bool(mx.array_equal(restored_values.indices, source.values.indices).item())


def test_cross_layer_reservation_exhaustion_is_atomic_before_forward():
    registry = _registry(capacity_pages=1)
    caches = registry.new_cache_set()
    blocker = registry.new_cache(0)
    blocker.update_and_fetch(*_kv(1))
    before = registry.stats()

    with pytest.raises(Exception, match="only 0 free"):
        with registry.reserve_append(caches, PAGE + 1):
            pytest.fail("forward must not start after reservation failure")

    after = registry.stats()
    assert after.used_layer_pages == before.used_layer_pages
    assert all(cache.sequence_lengths == (0,) for cache in caches.values())


def test_cross_layer_forward_exception_rolls_back_every_layer():
    registry = _registry(capacity_pages=4)
    caches = registry.new_cache_set()

    with pytest.raises(RuntimeError, match="synthetic forward failure"):
        with registry.reserve_append(caches, PAGE + 1):
            caches[0].update_and_fetch(*_kv(PAGE + 1))
            raise RuntimeError("synthetic forward failure")

    assert all(cache.sequence_lengths == (0,) for cache in caches.values())
    assert registry.stats().used_layer_pages == 0

    # The exact pages are reusable immediately; no layer retained a partial
    # reservation after the failed forward.
    with registry.reserve_append(caches, 1):
        for cache in caches.values():
            cache.update_and_fetch(*_kv(1))
    assert all(cache.sequence_lengths == (1,) for cache in caches.values())
    assert registry.stats().used_layer_pages == 2


def test_cross_layer_reservation_requires_every_layer_to_consume_append():
    registry = _registry(capacity_pages=2)
    caches = registry.new_cache_set()

    with pytest.raises(RuntimeError, match="did not consume"):
        with registry.reserve_append(caches, 1):
            caches[0].update_and_fetch(*_kv(1))

    assert all(cache.sequence_lengths == (0,) for cache in caches.values())
    assert registry.stats().used_layer_pages == 0
