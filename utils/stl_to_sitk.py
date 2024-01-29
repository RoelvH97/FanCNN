# import necessary modules
import numpy as np
import os
import time
import SimpleITK as sitk
import vtk

from glob import glob
from os.path import basename, dirname, join
from vtk.util.numpy_support import vtk_to_numpy


def sitk_to_numpy(sitk_file, swap=True):
    image = sitk.ReadImage(sitk_file)
    spacing = image.GetSpacing()
    offset = image.GetOrigin()

    image = sitk.GetArrayFromImage(image)
    if swap:
        image = np.swapaxes(image, 0, 2)
    return image, spacing, offset


def numpy_to_sitk(image, spacing, offset, filename, swap=True):
    if swap:
        image = np.swapaxes(image, 0, 2)
    image = sitk.GetImageFromArray(image.astype(np.int16))
    image.SetSpacing(spacing)
    image.SetOrigin(offset)

    writer = sitk.ImageFileWriter()
    writer.SetUseCompression(True)
    writer.SetFileName(filename)
    writer.Execute(image)


def stl_to_sitk(stl_dir, sitk_path, name_out="tree"):
    print(f"Voxelizing STL files for {basename(sitk_path)}...")

    # load the image
    image, spacing, offset = sitk_to_numpy(sitk_path)
    mask, shape = np.zeros_like(image), image.shape

    t0 = time.time()
    for stl_file in sorted(glob(join(stl_dir, "*_l.stl"))):
        # load the STL file
        reader = vtk.vtkSTLReader()
        reader.SetFileName(stl_file)
        reader.Update()

        # apply the offset to the mesh
        transform = vtk.vtkTransform()
        transform.Translate([-o for o in offset])  # back to voxel space, so invert the offset

        transform_filter = vtk.vtkTransformFilter()
        transform_filter.SetInputConnection(reader.GetOutputPort())
        transform_filter.SetTransform(transform)

        # convert the mesh into a stencil
        data_to_stencil = vtk.vtkPolyDataToImageStencil()
        data_to_stencil.SetInputConnection(transform_filter.GetOutputPort())
        data_to_stencil.SetOutputSpacing(*spacing)
        # data_to_stencil.SetOutputOrigin(*[0.5 * s for s in spacing])  # voxel center
        data_to_stencil.SetOutputOrigin(*[0 for s in spacing])  # voxel corner
        data_to_stencil.SetOutputWholeExtent(0, shape[0] - 1, 0, shape[1] - 1, 0, shape[2] - 1)

        # convert the stencil to an image
        stencil_to_image = vtk.vtkImageStencilToImage()
        stencil_to_image.SetInputConnection(data_to_stencil.GetOutputPort())
        stencil_to_image.SetOutsideValue(0)  # background value
        stencil_to_image.SetInsideValue(1)  # foreground value
        stencil_to_image.Update()

        # convert VTK image data to numpy array
        vtk_image = stencil_to_image.GetOutput()
        shape = vtk_image.GetDimensions()
        binary_mask = vtk_to_numpy(vtk_image.GetPointData().GetScalars())
        binary_mask = binary_mask.reshape(shape[2], shape[1], shape[0])
        binary_mask = np.transpose(binary_mask, (2, 1, 0))
        mask += binary_mask

    mask = np.clip(mask, 0, 1)
    numpy_to_sitk(mask, spacing, offset, f"{name_out}.nrrd")
    print(f"Voxelization took {time.time() - t0:.2f} seconds")


if __name__ == "__main__":
    dir_in = "/home/roel/data/ASOCA_Philips/images"
    dir_stl = "/home/roel/data/ASOCA_Philips/automatic_segmentations/{}/img"

    dirs = sorted(glob(join(dir_in, "*.mhd")))
    for i, file_in in enumerate(dirs[:35] + dirs[36:]):
        dir_dir_stl = dir_stl.format(file_in.split(os.sep)[-1].replace(".mhd", ""))
        stl_to_sitk(dir_dir_stl, file_in, join(dir_dir_stl, basename(file_in).replace(".mhd", "_l")))
