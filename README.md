# Skeleton-Annotator

Simple tool to annotator skeleton of 3D curvilinear structure


!!! Moved to [neurofly](https://github.com/beanli161514/neurofly) as a napari plugin



## Dependencies

latest version of: 
brightest_path_lib 
magicgui 
napari 
numpy 
tifffile

## Usage

```
python annotator.py
```

Click Select file button to load an 3D tiff image, then press refresh to update canvas. An example data imaged by [VISoR](https://www.nature.com/articles/s41587-021-00986-5) is at ./assets/img_20.tif

Use left and right double click to add start point and goal point. Then push f key to find a path between the two points.

<img src="assets/demo.gif" width="480">


## License

Distributed under the terms of the [BSD-3] license,

"Skeleton-Annotator" is free and open source software
