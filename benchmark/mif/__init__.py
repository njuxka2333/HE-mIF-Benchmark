"""What happens after the frozen HE→mIF generator.

* ``train_unet`` / ``train_feature`` — train a segmentation head on the stack
* ``score`` — Cell-niches virtual-staining metrics on the same stack

Uses ``models.mif_output``. Output-layer training is not in ``results/``.
"""
