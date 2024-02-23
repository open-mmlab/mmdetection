# CLIP models

## RN50

```python
clip_cfg=dict(
    type='CLIP',
    image_encoder=dict(
        type='CLIPResNet',
        layers=[3, 4, 6, 3],
        output_dim=1024,
        heads=32,
        input_resolution=224,
        width=64,
        init_cfg=dict(
            type='Pretrained',
            prefix='visual',
            checkpoint='checkpoints/clip_r50.pth')
    ),
    text_encoder=dict(
        type='CLIPTextEncoder',
        embed_dim=1024,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,    # also the word embedding dim
        transformer_heads=8,
        transformer_layers=12,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='checkpoints/clip_r50.pth')
    )
)
```

## ViT-B/32

```python
clip_cfg=dict(
    type='CLIP',
    image_encoder=dict(
        type='CLIPViT',
        input_resolution=224,
        patch_size=32,
        width=768,
        layers=12,
        heads=12,
        output_dim=512,
        init_cfg=dict(
            type='Pretrained',
            prefix='visual',
            checkpoint='checkpoints/clip_vitb32.pth')
    ),
    text_encoder=dict(
        type='CLIPTextEncoder',
        embed_dim=512,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,    # also the word embedding dim
        transformer_heads=8,
        transformer_layers=12,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='checkpoints/clip_vitb32.pth')
    )
)
```
