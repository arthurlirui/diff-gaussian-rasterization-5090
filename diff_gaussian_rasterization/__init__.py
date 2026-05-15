#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# Refactored for CUDA 13.2 / sm_120
# - Added comprehensive type annotations
# - Added depth rendering support in Python API
# - Improved error messages
#

from typing import NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
from . import _C


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)


def rasterize_gaussians(
    means3D: torch.Tensor,
    means2D: torch.Tensor,
    sh: torch.Tensor,
    colors_precomp: torch.Tensor,
    opacities: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    cov3Ds_precomp: torch.Tensor,
    raster_settings: "GaussianRasterizationSettings",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Invoke the CUDA 13.2 / sm_120 accelerated Gaussian rasterization.

    Args:
        means3D: 3D means of Gaussians, shape (N, 3)
        means2D: 2D means placeholder (not used, kept for API compatibility)
        sh: Spherical harmonics coefficients, shape (N, K, 3)
        colors_precomp: Precomputed colors, shape (N, C)
        opacities: Opacity values, shape (N, 1)
        scales: Scale parameters, shape (N, 3)
        rotations: Rotation quaternions, shape (N, 4)
        cov3Ds_precomp: Precomputed 3D covariances, shape (N, 6)
        raster_settings: Rasterization settings object

    Returns:
        Tuple of (color, radii, depth):
            - color: Rendered image, shape (C, H, W)
            - radii: Radii of each Gaussian, shape (N,)
            - depth: Rendered depth map, shape (H, W)
    """
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug
        )

        # Invoke C++/CUDA rasterizer (CUDA 13.2 / sm_120 backend)
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)  # Copy them before they can be corrupted
            try:
                num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, depth = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, depth = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)
        return color, radii, depth

    @staticmethod
    def backward(ctx, grad_out_color, _, __):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D,
                radii,
                colors_precomp,
                scales,
                rotations,
                raster_settings.scale_modifier,
                cov3Ds_precomp,
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
                raster_settings.tanfovx,
                raster_settings.tanfovy,
                grad_out_color,
                sh,
                raster_settings.sh_degree,
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)  # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads


class GaussianRasterizationSettings(NamedTuple):
    """
    Configuration settings for Gaussian rasterization.

    All tensors must be on CUDA device for the CUDA 13.2 / sm_120 backend.

    Attributes:
        image_height: Output image height in pixels
        image_width: Output image width in pixels
        tanfovx: Tangent of half horizontal field of view
        tanfovy: Tangent of half vertical field of view
        bg: Background color tensor, shape (C,)
        scale_modifier: Scale modifier for Gaussian covariance
        viewmatrix: View (camera extrinsic) matrix, shape (4, 4)
        projmatrix: Projection matrix, shape (4, 4)
        sh_degree: Spherical harmonics degree (0-3)
        campos: Camera position in world space, shape (3,)
        prefiltered: Whether points have been prefiltered for frustum culling
        debug: Enable debug mode with CUDA error checking after each kernel
    """
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    debug: bool


class GaussianRasterizer(nn.Module):
    """
    Differentiable Gaussian rasterizer backed by CUDA 13.2 / sm_120 kernels.

    This module performs alpha-blended rendering of 3D Gaussians to a 2D image
    with full support for backpropagation. The CUDA kernels are optimized for
    sm_120 architecture with:
    - Cooperative groups warp-level processing
    - Streaming loads for read-once data
    - Warp-level gradient aggregation to reduce atomicAdd contention
    - Enhanced error checking for CUDA 13.2

    Example usage:
        >>> settings = GaussianRasterizationSettings(
        ...     image_height=512, image_width=512,
        ...     tanfovx=0.5, tanfovy=0.5,
        ...     bg=torch.zeros(3, device='cuda'),
        ...     scale_modifier=1.0,
        ...     viewmatrix=viewmat, projmatrix=projmat,
        ...     sh_degree=3, campos=cam_pos,
        ...     prefiltered=False, debug=False
        ... )
        >>> rasterizer = GaussianRasterizer(settings)
        >>> color, radii, depth = rasterizer(
        ...     means3D=points, means2D=torch.zeros_like(points[:,:2]),
        ...     opacities=opacities, shs=sh_coeffs,
        ...     scales=scales, rotations=quats
        ... )
    """

    def __init__(self, raster_settings: GaussianRasterizationSettings):
        """
        Initialize the Gaussian rasterizer.

        Args:
            raster_settings: Configuration settings for rasterization
        """
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Mark which Gaussians are visible from the current camera view.

        Uses frustum culling to determine visibility.

        Args:
            positions: 3D positions of Gaussians, shape (N, 3)

        Returns:
            Boolean tensor indicating visibility, shape (N,)
        """
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)

        return visible

    def forward(
        self,
        means3D: torch.Tensor,
        means2D: torch.Tensor,
        opacities: torch.Tensor,
        shs: Optional[torch.Tensor] = None,
        colors_precomp: Optional[torch.Tensor] = None,
        scales: Optional[torch.Tensor] = None,
        rotations: Optional[torch.Tensor] = None,
        cov3D_precomp: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Render Gaussians to a 2D image with depth.

        You must provide exactly one of:
        - shs (spherical harmonics) or colors_precomp (precomputed colors)
        - scales+rotations or cov3D_precomp (precomputed 3D covariance)

        Args:
            means3D: 3D Gaussian centers, shape (N, 3), float32, CUDA
            means2D: 2D means placeholder, shape (N, 3), float32, CUDA
            opacities: Opacity values, shape (N, 1), float32, CUDA
            shs: Spherical harmonics coefficients, shape (N, K, 3), float32, CUDA
            colors_precomp: Precomputed RGB colors, shape (N, C), float32, CUDA
            scales: Scale parameters, shape (N, 3), float32, CUDA
            rotations: Rotation quaternions (w,x,y,z), shape (N, 4), float32, CUDA
            cov3D_precomp: Precomputed 3D covariance, shape (N, 6), float32, CUDA

        Returns:
            Tuple of (color, radii, depth):
                - color: Rendered image, shape (C, H, W), float32
                - radii: Screen-space radii, shape (N,), int32
                - depth: Rendered depth map, shape (H, W), float32
        """
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide exactly one of either SHs or precomputed colors!')

        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterizer (CUDA 13.2 / sm_120 backend)
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            raster_settings,
        )
