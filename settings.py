SETTINGS = dict(
    train_y='data/train/measurement', train_depth='data/train/depth',
    val_y='data/val/measurement', val_depth='data/val/depth',
    real_dir='data/real', prepared_dir='data/prepared_12', output='output',
    expected_hw=[1536, 2500],  # training only; None keeps the file grid
    phase_mask_mode='fixed', phase_mask_seed=41,
    black_level=0., white_level=None, gain=1., input_encoding='linear',
    depth_scale=1., depth_min=.5, depth_max=5.,
    encoder='vits', metric_max_depth=20.,
    da_weights='weights/depth_anything_v2_metric_hypersim_vits.pth',
    use_coordinate_prior=False, use_refiner=False,
    freeze_encoder=False, freeze_da=False, attention_backend='sdpa',
    device='cuda', batch_size=2, inference_batch_size=1, epochs=20,
    lr_encoder=1e-6, lr_decoder=5e-6, lr_extra=1e-5, seed=41,
    checkpoint_da_activations=False,  # recompute DA microbatches on backward to reduce peak memory
    amp=True, amp_dtype='float16', depth_l1_weight=1., depth_edge_weight=.1,
    real_each_epoch=False, train_save_every=200,
)
