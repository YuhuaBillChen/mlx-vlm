from bisect import bisect_left
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures
from ..qwen3_vl import Model as Qwen3VLModel
from ..qwen3_vl import processing_qwen3_vl  # noqa: F401
from ..qwen3_vl.qwen3_vl import masked_scatter
from .config import ModelConfig
from .fp8 import convert_qwen_fp8_weights
from .language import LanguageModel
from .vision import VisionModel


class ChunkedInputEmbeddingProvider:
    """Build Qwen3.5 multimodal embeddings one prefill chunk at a time."""

    def __init__(
        self,
        embed_tokens,
        image_features,
        visual_positions,
        image_token_index,
        video_token_index,
    ):
        self.embed_tokens = embed_tokens
        self.image_features = image_features
        self.visual_positions = tuple(visual_positions)
        self.image_token_index = image_token_index
        self.video_token_index = video_token_index

    def __call__(self, input_ids: mx.array, *, start: int) -> mx.array:
        inputs_embeds = self.embed_tokens(input_ids)
        stop = start + input_ids.shape[1]
        first = bisect_left(self.visual_positions, start)
        last = bisect_left(self.visual_positions, stop)
        if first == last:
            return inputs_embeds
        inputs_embeds, _ = Model.merge_input_ids_with_image_features(
            self.image_features[first:last],
            inputs_embeds,
            input_ids,
            self.image_token_index,
            self.video_token_index,
        )
        return inputs_embeds


def sanitize_key(key):
    if key.startswith("model.language_model.visual"):
        key = key.replace("model.language_model.visual", "vision_tower", 1)
    elif key.startswith("model.language_model"):
        key = key.replace("model.language_model", "language_model.model", 1)
    elif key.startswith("model.visual"):
        key = key.replace("model.visual", "vision_tower", 1)
    elif key.startswith("lm_head"):
        key = key.replace("lm_head", "language_model.lm_head", 1)
    return key


NORM_WEIGHT_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    "model.norm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
)


def should_shift_norm_weights(weights):
    has_mtp_weights = any("mtp." in key for key in weights)
    has_unsanitized_conv1d = any(
        "conv1d.weight" in key and value.shape[-1] != 1
        for key, value in weights.items()
    )
    return has_mtp_weights or has_unsanitized_conv1d


def should_offset_norm_weight(original_key, shift_norm_weights):
    return shift_norm_weights or not original_key.startswith("language_model.")


class Model(Qwen3VLModel):

    def __init__(self, config: ModelConfig):
        # only initialize nn.Module, skip the initialization of vision_tower and language_model in the parent class
        nn.Module.__init__(self)
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config, config)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        if pixel_values is None:
            pixel_values = kwargs.get("pixel_values_videos", None)

        image_grid_thw = kwargs.get("image_grid_thw", None)
        video_grid_thw = kwargs.get("video_grid_thw", None)
        mask = kwargs.get("mask", None)
        chunked = bool(kwargs.pop("chunked", False))
        if chunked and input_ids.shape[0] != 1:
            raise ValueError(
                "chunk-local Qwen3.5 input embeddings require batch size 1"
            )
        grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw

        if pixel_values is None:
            position_ids, rope_deltas = self.language_model.get_rope_index(
                input_ids, attention_mask=mask
            )
            return InputEmbeddingsFeatures(
                inputs_embeds=(
                    None
                    if chunked
                    else self.language_model.model.embed_tokens(input_ids)
                ),
                input_embedding_provider=(
                    ChunkedInputEmbeddingProvider(
                        self.language_model.model.embed_tokens,
                        None,
                        (),
                        self.config.image_token_index,
                        self.config.video_token_index,
                    )
                    if chunked
                    else None
                ),
                position_ids=position_ids,
                rope_deltas=rope_deltas,
            )

        dtype = self.vision_tower.patch_embed.proj.weight.dtype
        pixel_values = pixel_values.astype(dtype)

        vision_cache = kwargs.get("vision_cache", None)
        cached = kwargs.get("cached_image_features", None)
        if cached is None and vision_cache is not None:
            cached = vision_cache.get(kwargs.get("_image_key"))
        if cached is not None:
            hidden_states = cached
        else:
            # Get the ouptut hidden states from the vision model
            hidden_states, _ = self.vision_tower(pixel_values, grid_thw)
            if vision_cache is not None and kwargs.get("_image_key") is not None:
                mx.eval(hidden_states)
                vision_cache.put(kwargs["_image_key"], hidden_states)

        if chunked:
            token_ids = input_ids[0].tolist()
            visual_positions = [
                index
                for index, token_id in enumerate(token_ids)
                if token_id
                in (self.config.image_token_index, self.config.video_token_index)
            ]
            if len(visual_positions) != hidden_states.shape[0]:
                raise ValueError(
                    "Image features and image tokens do not match: "
                    f"tokens: {len(visual_positions)}, features {hidden_states.shape[0]}"
                )
            position_ids, rope_deltas = self.language_model.get_rope_index(
                input_ids, image_grid_thw, video_grid_thw, mask
            )
            return InputEmbeddingsFeatures(
                inputs_embeds=None,
                input_embedding_provider=ChunkedInputEmbeddingProvider(
                    self.language_model.model.embed_tokens,
                    hidden_states,
                    visual_positions,
                    self.config.image_token_index,
                    self.config.video_token_index,
                ),
                position_ids=position_ids,
                rope_deltas=rope_deltas,
            )

        # Get the input embeddings from the language model
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        # Insert special image tokens in the input_ids
        inputs_embeds, _ = self.merge_input_ids_with_image_features(
            hidden_states,
            inputs_embeds,
            input_ids,
            self.config.image_token_index,
            self.config.video_token_index,
        )

        position_ids, rope_deltas = self.language_model.get_rope_index(
            input_ids, image_grid_thw, video_grid_thw, mask
        )

        return InputEmbeddingsFeatures(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            rope_deltas=rope_deltas,
        )

    @staticmethod
    def merge_input_ids_with_image_features(
        image_features, inputs_embeds, input_ids, image_token_index, video_token_index
    ):
        special_image_mask = input_ids == image_token_index
        special_video_mask = input_ids == video_token_index
        special_image_mask = special_image_mask | special_video_mask
        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask[..., None]
        special_image_mask = mx.broadcast_to(special_image_mask, inputs_embeds.shape)

        n_image_features = image_features.shape[0]
        n_image_mask_elements = special_image_mask.sum()
        if n_image_mask_elements != image_features.size:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        inputs_embeds = masked_scatter(
            inputs_embeds, special_image_mask, image_features
        )

        return inputs_embeds, special_image_mask

    def sanitize(self, weights):
        # The MTP draft shard is separate from the base model. Its presence
        # must not select the base model's RMSNorm loading convention.
        weights = {key: value for key, value in weights.items() if "mtp." not in key}
        weights = convert_qwen_fp8_weights(weights)
        shift_norm_weights = should_shift_norm_weights(weights)

        if self.config.text_config.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        sanitized_weights = {}
        for key, value in weights.items():
            original_key = key
            key = sanitize_key(key)

            if "conv1d.weight" in key and value.shape[-1] != 1:
                value = value.moveaxis(2, 1)
            if any(key.endswith(sfx) for sfx in NORM_WEIGHT_SUFFIXES):
                if value.ndim == 1 and should_offset_norm_weight(
                    original_key, shift_norm_weights
                ):
                    value += 1.0

            sanitized_weights[key] = value

        return sanitized_weights

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate
