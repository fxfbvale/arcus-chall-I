"""Render weight matrices as images + FFT structure stats, hunting a drawn-in watermark.
A flag drawn into a weight matrix would be visible in the raw image, or show as periodic
structure (sharp non-DC FFT peaks) vs random weights (flat spectrum).
"""
import sys; sys.path.insert(0,'solve')
import torch, numpy as np
from PIL import Image
from gen import load
model, tok = load()
sd = model.state_dict()

def norm_img(M):
    M=np.asarray(M,dtype=np.float64)
    lo,hi=np.percentile(M,1),np.percentile(M,99)
    Mc=np.clip((M-lo)/(hi-lo+1e-9),0,1)
    return (Mc*255).astype(np.uint8)

def fft_stats(M):
    M=np.asarray(M,dtype=np.float64); M=M-M.mean()
    F=np.abs(np.fft.fft2(M)); F[0,0]=0
    flat=F.flatten(); mx=flat.max(); mean=flat.mean()
    peak=np.unravel_index(F.argmax(),F.shape)
    return mx/ (mean+1e-9), peak, F.shape

targets = {
 'wte': sd['transformer.wte.weight'].numpy(),
 'wpe': sd['transformer.wpe.weight'].numpy(),
 'h0_cfc': sd['transformer.h.0.mlp.c_fc.weight'].numpy(),
 'h9_cproj': sd['transformer.h.9.mlp.c_proj.weight'].numpy(),
 'h0_attn': sd['transformer.h.0.attn.c_attn.weight'].numpy(),
}
print("=== FFT peak-to-mean ratio (random weights ~ <10; a periodic watermark spikes high) ===")
for name,M in targets.items():
    r,peak,shp=fft_stats(M)
    img=Image.fromarray(norm_img(M))
    # downscale very tall images for viewing
    if img.height>512: img=img.resize((img.width, 512))
    if img.width>1024: img=img.resize((1024, img.height))
    path=f'/tmp/wm_{name}.png'; img.save(path)
    print(f"  {name:10s} shape={M.shape} peak/mean={r:7.2f} peakfreq={peak} -> {path}")

# Also: render wte as a TIGHT 262x640 and a transposed view; and a binarized (sign) view
W=sd['transformer.wte.weight'].numpy()
Image.fromarray(norm_img(W)).save('/tmp/wm_wte_raw.png')
Image.fromarray(((W>0)*255).astype(np.uint8)).save('/tmp/wm_wte_sign.png')
# sign bit of float32 across the matrix (stego sometimes in sign bits)
Image.fromarray(norm_img(W.T)).save('/tmp/wm_wte_T.png')
print("\nsaved /tmp/wm_wte_raw.png, wm_wte_sign.png, wm_wte_T.png")
print("done")
