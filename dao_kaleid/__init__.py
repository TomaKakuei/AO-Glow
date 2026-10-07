"""Portable DAO/Kaleid optical simulation, trained predictors and feedback."""
import os

# Set before importing numerical libraries. Process-local, not global settings.
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_name, '1')
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

__version__ = '2026.10.06'
