from typing import Optional, Sequence, Tuple
import os

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

try:
	import torch
except Exception:
	torch = None


def _to_display_array(img, size: Optional[Tuple[int, int]] = (224, 224)) -> np.ndarray:
	"""Convert PIL.Image, torch.Tensor, or numpy array to an HxWxC float array in [0,1].

	Args:
		img: PIL.Image, torch.Tensor, or numpy.ndarray.
		size: if PIL.Image, resize to this (width, height). If None, keep original size.

	Returns:
		numpy.ndarray shape (H, W, C) with floats in [0,1].
	"""
	# PIL Image
	if isinstance(img, Image.Image):
		if size is not None:
			img = img.resize(size)
        
		arr = np.array(img)
		if arr.dtype == np.uint8:
			arr = arr / 255.0
		else:
			arr = arr
		return arr

	# torch Tensor
	if torch is not None and isinstance(img, torch.Tensor):
		t = img.clone().detach().cpu()
		if t.dim() == 4 and t.size(0) == 1:
			t = t.squeeze(0)
		arr = t.numpy()
		if arr.ndim == 3:
			arr = np.transpose(arr, (1, 2, 0))
		elif arr.ndim == 2:
			arr = np.stack([arr] * 3, axis=-1)
		if arr.dtype == np.uint8 or arr.max() > 2.0:
			arr = arr.astype(np.float32) / 255.0
		else:
			arr = arr.astype(np.float32)
		return arr

	# numpy array
	if isinstance(img, np.ndarray):
		arr = img.astype(np.float32)
		if arr.ndim == 2:
			arr = np.stack([arr] * 3, axis=-1)
		if arr.ndim == 3 and arr.shape[2] not in (1, 3):
			if arr.shape[0] in (1, 3):
				arr = np.transpose(arr, (1, 2, 0))
		if arr.max() > 2.0:
			arr = arr / 255.0
		return arr

	raise TypeError(f"Unsupported image type: {type(img)}")


def plot_image_comparison(
	images: Sequence,
	titles: Optional[Sequence[str]] = None,
	save_path: Optional[str] = None,
	size: Optional[Tuple[int, int]] = (224, 224),
):
	"""Plot N images side-by-side and optionally save to disk.

	Supports PIL Images, torch.Tensors, and numpy arrays. Tensors are assumed to be
	either (1,C,H,W), (C,H,W), or (H,W) and will be converted to HxWxC.

	Args:
		images: sequence of images to display.
		titles: optional sequence of titles, same length as images.
		save_path: optional path to save the figure. If None, plt.show() is called.
		size: target size (width, height) to resize PIL images to. If None, keep original.
	"""
	n = len(images)
	if n == 0:
		return

	titles = list(titles) if titles is not None else [""] * n
	if len(titles) < n:
		titles += [""] * (n - len(titles))

	# convert all images
	conv = []
	for img in images:
		conv.append(_to_display_array(img, size=size))

	# compute figure size: 4 inches per image width, keep height 4
	fig_w = 4 * n
	fig_h = 4
	fig, axs = plt.subplots(1, n, figsize=(fig_w, fig_h))
	if n == 1:
		axs = [axs]

	for ax, arr, title in zip(axs, conv, titles):
		# clip to 0..1
		arr = np.clip(arr, 0.0, 1.0)
		if arr.shape[2] == 1:
			ax.imshow(arr[:, :, 0], cmap='gray')
		else:
			ax.imshow(arr)
		ax.set_title(title)
		ax.axis('off')

	plt.tight_layout()
	if save_path:
		os.makedirs(os.path.dirname(save_path), exist_ok=True)
		plt.savefig(save_path)
		plt.close()
	else:
		plt.show()


__all__ = ["plot_image_comparison"]

