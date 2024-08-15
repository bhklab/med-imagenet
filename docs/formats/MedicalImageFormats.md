# Medical Image Formats

## Introduction

A medical image is a digital representation of the internal structure or function of an anatomic region.

- often in the form of an array of picture elements called pixels or voxels
- discrete representation that maps numerical values to positions in space
- the numerical values of each pixel or voxel are often called intensity values 
  - what they represent depends on the type of image, acquisition method, reconstruction, and post-processing

## Medical Image Metadata

Often, medical images are accompanied by metadata that provides additional information about the image.

- this information is usually stored in the beginning of the image file, typically as a "header" 
  - some common metadata fields include:
    - image dimensions (width, height, depth, etc.)
    - voxel size (spacing between voxels)
    - origin (location of the first voxel in the image)
    - orientation (direction of the x, y, and z axes)
    - pixel depth (number of bytes used to represent each voxel intensity)
    - data type (integer, floating-point, etc.)

## Pixel Data

The pixel data in a medical image file is the numerical representation of the image.

- the pixel data is stored in a format that is specific to the image file format
  - depending on the data type
- In formats that adopt a fixed-size header, the pixel data immediately follows the header.
  - otherwise, the start of the pixel data is indicated by a special marker or tag.

Pixel Data Size = Rows * Columns * Pixel Depth (Bytes) * Number of Frames

Image File Size = Header Size + Pixel Data Size


### Medical Image File Formats

- **Categories**:
  - **Standardization Formats**: Intended to standardize images from diagnostic modalities (e.g., Dicom).
  - **Post-Processing Formats**: Designed to facilitate and strengthen post-processing analysis (e.g., Analyze, Nifti, Minc).

- **Configurations for Storing**:
  - **Single File**: Contains both metadata and image data; metadata is stored at the beginning of the file.
    - Examples: Dicom, Minc, Nifti
  - **Two Files**: Metadata and image data are stored separately.
    - Example: Analyze (.hdr and .img)