# optimizer
optimizer = dict(type='Adam', lr=0.0008, betas=(0.9, 0.999), weight_decay=0, amsgrad=False)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(policy='Plateau', mode='min', factor=0.8, patience=3)
runner = dict(type='EpochBasedRunner', max_epochs=5)