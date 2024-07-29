import numpy as np

from skeleton_annotator._widget import (
    AnnotatorWidget
)

def test_annotator_widget(make_napari_viewer):
    viewer = make_napari_viewer()
    my_widget = AnnotatorWidget(viewer)
    pass