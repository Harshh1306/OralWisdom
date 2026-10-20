"""Original parametric visible-enamel silhouettes; no prosthetic CAD geometry."""
import numpy as np

VERSION = 'veneer-silhouettes-1'
# Cervical, shoulder, body, incisal shoulder and edge half-widths.
PROFILES = {
    ('upper', 'central'): [.53, .83, 1., 1., .98, .87],
    ('upper', 'lateral'): [.50, .82, 1., .99, .94, .76],
    ('upper', 'canine'): [.48, .80, 1., .95, .79, .42],
    ('lower', 'central'): [.58, .86, 1., .99, .97, .88],
    ('lower', 'lateral'): [.55, .85, 1., .99, .95, .84],
    ('lower', 'canine'): [.50, .81, 1., .95, .82, .48],
}

def silhouette(ratio, arch, tooth_type, style='balanced'):
    values = np.array(PROFILES.get((arch, tooth_type), PROFILES[(arch, 'lateral')]))
    if style == 'soft':
        values[-2:] -= [.025, .07]
    return np.interp(ratio, [0, .08, .28, .76, .93, 1.], values)

def incisal_offset(arch, tooth_type):
    # Modest differentiation, in units of visible tooth height, not millimetres.
    return -.035 if arch == 'upper' and tooth_type == 'lateral' else 0.
