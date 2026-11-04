"""GSCV: generative screen content video compression with selective fidelity and interaction-aware context,
built on GVC1D.

rsm.py        window state machine: block classes F_t, window states S_t, quality offsets, side-information cost
attention.py  cross-window attention restricted to the coded windows (SDPA, true-grid RoPE)
model.py      GSCV model (subclasses of the GVC1D encoder, decoder, bottleneck and memory)
losses.py     region-adaptive rate-distortion-perception loss
data.py       training clips with synthetic interaction events and their window-state annotations
"""
