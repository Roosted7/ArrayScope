"""Shared framebuffer-to-CPU reference oracles (drivers × oracles strategy).

Oracles here judge recorded evidence — real framebuffer pixels against
CPU-computed references of the same semantic values — and are shared by the
real-GL ring (``tests/gpu_interaction``) and the default offscreen ring.
"""
