"""
Experiment 2 Figure Generation
4-panel figure for k=7 topological Vicsek model
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd

# Load summary data
summary_df = pd.read_csv('exp2_summary_firstpass.csv')

# Load curve data for specific noise values
noise_values = [0.005, 0.02, 0.05, 0.0738, 0.1, 0.15]
curve_data = {}
for noise in noise_values:
    filename = f'exp2_noise_{noise:.4f}.json'
    with open(filename, 'r') as f:
        curve_data[noise] = json.load(f)

# Set up the figure with 2x2 layout
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)

# Color scheme
colors = {
    0.005: '#1f77b4',   # blue
    0.02: '#ff7f0e',    # orange
    0.05: '#2ca02c',    # green
    0.0738: '#d62728',  # red
    0.1: '#9467bd',     # purple
    0.15: '#8c564b',    # brown
}

linestyles = {
    'Qn_tilde': '-',
    'Cdelta': '--',
    'MSDrel': '-.',
}

# ============================================
# Panel A: Representative Correlation Functions
# ============================================
ax_a = fig.add_subplot(gs[0, 0])

# Select three representative noise values: low, middle, high
selected_noises = [0.005, 0.0738, 0.15]

for noise in selected_noises:
    data = curve_data[noise]
    lag = np.array(data['lag'])
    
    # Qn_tilde (neighbor overlap)
    qn_mean = np.array(data['curves']['Qn_tilde_mean']['mean'])
    ax_a.plot(lag, qn_mean, color=colors[noise], linestyle='-', 
              label=f'D={noise}, Q̃ₙ', alpha=0.8, linewidth=2)
    
    # Cdelta (fluctuation autocorrelation)
    cdelta_mean = np.array(data['curves']['Cdelta_mean']['mean'])
    ax_a.plot(lag, cdelta_mean, color=colors[noise], linestyle='--', 
              label=f'D={noise}, Cδ', alpha=0.8, linewidth=2)

ax_a.set_xlabel('Lag τ', fontsize=12)
ax_a.set_ylabel('Correlation', fontsize=12)
ax_a.set_title('Panel A: Correlation Functions (k=7, N=1024)', fontsize=13, fontweight='bold')
ax_a.set_xscale('log')
ax_a.set_ylim(-0.1, 1.1)
ax_a.grid(True, alpha=0.3)
ax_a.legend(fontsize=8, ncol=2, loc='upper right')

# ============================================
# Panel B: Timescales vs Noise
# ============================================
ax_b = fig.add_subplot(gs[0, 1])

noise_arr = summary_df['noise'].values
tau_rw = summary_df['tau_rw'].values
tau_rw_se = summary_df['tau_rw_se'].values
tau_rel = summary_df['tau_rel'].values
tau_rel_se = summary_df['tau_rel_se'].values
tau_cage = summary_df['tau_cage'].values
tau_cage_se = summary_df['tau_cage_se'].values

# Plot with error bars
ax_b.errorbar(noise_arr, tau_rw, yerr=tau_rw_se, marker='o', markersize=8,
              capsize=4, linewidth=2, label=r'$\tau_{\mathrm{rw}}$ (neighbor renewal)',
              color='#1f77b4', alpha=0.8)
ax_b.errorbar(noise_arr, tau_rel, yerr=tau_rel_se, marker='s', markersize=8,
              capsize=4, linewidth=2, label=r'$\tau_{\mathrm{rel}}$ (fluctuation relaxation)',
              color='#ff7f0e', alpha=0.8)
ax_b.errorbar(noise_arr, tau_cage, yerr=tau_cage_se, marker='^', markersize=8,
              capsize=4, linewidth=2, label=r'$\tau_{\mathrm{cage}}$ (cage breaking)',
              color='#2ca02c', alpha=0.8)

ax_b.set_xlabel('Noise D', fontsize=12)
ax_b.set_ylabel('Timescale', fontsize=12)
ax_b.set_title('Panel B: Timescales vs Noise (k=7, N=1024)', fontsize=13, fontweight='bold')
ax_b.set_yscale('log')
ax_b.grid(True, alpha=0.3)
ax_b.legend(fontsize=10, loc='upper right')

# ============================================
# Panel C: MSD curves for different noise values
# ============================================
ax_c = fig.add_subplot(gs[1, 0])

for noise in selected_noises:
    data = curve_data[noise]
    lag = np.array(data['lag'])
    msd_mean = np.array(data['curves']['MSDrel_mean']['mean'])
    ell_nn2 = data['scalars']['ell_nn2']['mean']
    
    ax_c.plot(lag, msd_mean, color=colors[noise], linewidth=2, 
              label=f'D={noise}, ℓ²={ell_nn2:.3f}', alpha=0.8)
    # Mark the cage breaking threshold
    ax_c.axhline(y=ell_nn2, color=colors[noise], linestyle=':', alpha=0.5)

ax_c.set_xlabel('Lag τ', fontsize=12)
ax_c.set_ylabel('MSD_rel', fontsize=12)
ax_c.set_title('Panel C: Relative MSD (k=7, N=1024)', fontsize=13, fontweight='bold')
ax_c.set_xscale('log')
ax_c.set_yscale('log')
ax_c.grid(True, alpha=0.3)
ax_c.legend(fontsize=9, loc='lower right')

# ============================================
# Panel D: Phase diagram / Summary
# ============================================
ax_d = fig.add_subplot(gs[1, 1])

# Plot phi (polarization) vs noise
phi_mean = summary_df['phi_mean'].values
phi_se = summary_df['phi_se'].values

ax_d.errorbar(noise_arr, phi_mean, yerr=phi_se, marker='o', markersize=10,
              capsize=4, linewidth=2, label='Polarization φ', color='#d62728')

# Add vertical line at transition
ax_d.axvline(x=0.0738, color='gray', linestyle='--', alpha=0.5, label='D=0.0738')

ax_d.set_xlabel('Noise D', fontsize=12)
ax_d.set_ylabel('Polarization φ', fontsize=12)
ax_d.set_title('Panel D: Order Parameter vs Noise (k=7, N=1024)', fontsize=13, fontweight='bold')
ax_d.set_ylim(0, 1)
ax_d.grid(True, alpha=0.3)
ax_d.legend(fontsize=10, loc='upper right')

# Add annotation
ax_d.annotate('Ordered\nPhase', xy=(0.03, 0.8), fontsize=10, ha='center',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax_d.annotate('Disordered\nPhase', xy=(0.12, 0.15), fontsize=10, ha='center',
              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.suptitle('Experiment 2: Topological Vicsek Model (k=7, N=1024)', 
             fontsize=15, fontweight='bold', y=0.98)

plt.savefig('exp2_figure.png', dpi=300, bbox_inches='tight')
plt.savefig('exp2_figure.pdf', bbox_inches='tight')
print("Figure saved as exp2_figure.png and exp2_figure.pdf")

plt.show()
