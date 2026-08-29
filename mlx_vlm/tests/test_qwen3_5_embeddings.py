import mlx.core as mx

from mlx_vlm.models.qwen3_5.qwen3_5 import ChunkedInputEmbeddingProvider


def test_chunked_provider_matches_full_visual_embedding_merge():
    image_token = 99
    video_token = 100

    def embed_tokens(input_ids):
        return mx.repeat(input_ids[..., None].astype(mx.float32), 3, axis=-1)

    input_ids = mx.array([[1, image_token, 2, video_token, 3]])
    image_features = mx.array([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]])
    provider = ChunkedInputEmbeddingProvider(
        embed_tokens,
        image_features,
        (1, 3),
        image_token,
        video_token,
    )

    chunked = mx.concatenate(
        [
            provider(input_ids[:, :2], start=0),
            provider(input_ids[:, 2:4], start=2),
            provider(input_ids[:, 4:], start=4),
        ],
        axis=1,
    )

    expected = mx.array(
        [
            [
                [1.0, 1.0, 1.0],
                [10.0, 11.0, 12.0],
                [2.0, 2.0, 2.0],
                [20.0, 21.0, 22.0],
                [3.0, 3.0, 3.0],
            ]
        ]
    )
    assert mx.array_equal(chunked, expected).item()
