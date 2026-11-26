SETTINGS = dict(
    train_y='data/train/measurement', train_depth='data/train/depth',
    val_y='data/val/measurement', val_depth='data/val/depth',
    real_dir='data/real', prepared_dir='data/prepared_12', output='output',
    train_image_extensions=['.tif', '.tiff'],
    val_image_extensions=['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'],
    real_image_extensions=['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'],
    depth_extensions=['.tif', '.tiff'],  # paired train/validation depth labels
    expected_hw=[1536, 2500],  # H,W; None keeps native resolution
    center_crop_hw=[1036, 2072],  # H,W after resize; None keeps the full grid
    resize_before_inference_crop=False,  # test/real: center crop native input by default
    phase_grid_xy=[4, 2],  # horizontal, vertical; produces 4*2 subimages
    training_input='raw',  # 'raw' or existing 'prepared' fixed-mask NPZ
    phase_mask_mode='fixed', phase_mask_seed=41,  # 'fixed' or 'learned'
    mask_tile_hw=[16, 16],  # in grid cells; learned switches shared across frames
    lr_mask=1e-3,
    save_sampling_preview=True,  # first raw test frame of each epoch

    black_level=0., white_level=None, gain=1., input_encoding='linear',
    min_train_depth_percent=80.,  # train and test: minimum percent of crop pixels with in-range GT (inclusive)
    dataset_audit=False,  # True generates the full pre-training depth-range report
    depth_scale=1., depth_min=.5, depth_max=5.,
    da_model_type='metric',  # 'metric' or 'relative'
    relative_da_weights='weights/depth_anything_v2_vits.pth', lr_calibration=1e-3,
    encoder='vits', metric_max_depth=20.,
    da_weights='weights/depth_anything_v2_metric_hypersim_vits.pth',
    checkpoint=None,  # optional My_code_12 training checkpoint (.pt)
    da_weight_priority=True,  # with both files, official DA overrides checkpoint DA
    strict_weight_load=False,  # training: copy tensors with matching names and shapes
    strict_inference_weights=True,  # standalone evaluation/real: set False to allow partial replay
    use_coordinate_prior=False, use_refiner=False,
    freeze_encoder=False, freeze_da=False, attention_backend='sdpa',
    device='cuda', batch_size=2, inference_batch_size=1, epochs=20,
    inference_precision='float32',  # 'float16' / 'bfloat16' enable CUDA autocast for reconstruction
    fast_reconstruction_export=False,  # uncompressed NPZ and low-compression full-resolution PNG
    profile_train_steps=0,  # e.g. 10: synchronized per-stage times for first N training updates
    train_prefetch=False,  # overlap one CPU frame read with current training step
    profile_reconstruction=False,  # per-frame read, inference, metrics and export wall times
    lr_encoder=1e-6, lr_decoder=5e-6, lr_extra=1e-5, seed=41,
    checkpoint_da_activations=False,  # recompute DA microbatches on backward to reduce peak memory
    amp=True, amp_dtype='float16', depth_l1_weight=1., depth_edge_weight=.1,
    test_export_every=1,  # full-resolution test files every N epochs and the final epoch
    real_each_epoch=False, train_save_every=200,
)
