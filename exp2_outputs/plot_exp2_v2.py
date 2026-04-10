"""
Experiment 2 Figure Generation - Version 2
4-panel figure with N=1024 and N=2048 size robustness comparison
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd

# Load summary data for both N values
summary_N1024 = pd.read_csv('N1024/exp2_summary_N1024.csv')
summary_N2048 = pd.read_csv('N2048/exp2_summary_N2048.csv')

# Load curve data for N=1024 (for Panels A and C)
noise_values = [0.005, 0.02, 0.05, 0.0738, 0.1, 0.15]
curve_data_N1024 = {}
for noise in noise_values:
    filename = f'exp2_noise_{noise:.4f}.json'
    with open(filename, 'r') as f:
        curve_data_N1024[noise] = json.load(f)

# Set up the figure with 2x2 layout
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)

# Color scheme for noise values
colors = {
    0.005: '#1f77b4',   # blue
    0.02: '#ff7f0e',    # orange  
    0.05: '#2ca02c',    # green
    0.0738: '#d62728',  # red
    0.1: '#9467bd',     # purple
    0.15: '#8c564b',    # brown
}

# ============================================
# Panel A: Representative Correlation Functions (N=1024)
# ============================================
ax_a = fig.add_subplot(gs[0, 0])

# Select three representative noise values: low, middle, high
selected_noises = [0.005, 0.0738, 0.15]

for noise in selected_noises:
    data = curve_data_N1024[noise]
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
# Panel B: Timescales vs Noise (Size Robustness)
# ============================================
ax_b = fig.add_subplot(gs[0, 1])

# N=1024 data
noise_1024 = summary_N1024['noise'].values
tau_rw_1024 = summary_N1024['tau_rw'].values
tau_rw_se_1024 = summary_N1024['tau_rw_se'].values
tau_rel_1024 = summary_N1024['tau_rel'].values
tau_rel_se_1024 = summary_N1024['tau_rel_se'].values
tau_cage_1024 = summary_N1024['tau_cage'].values
tau_cage_se_1024 = summary_N1024['tau_cage_se'].values

# N=2048 data
noise_2048 = summary_N2048['noise'].values
tau_rw_2048 = summary_N2048['tau_rw'].values
tau_rw_se_2048 = summary_N2048['tau_rw_se'].values
tau_rel_2048 = summary_N2048['tau_rel'].values
tau_rel_se_2048 = summary_N2048['tau_rel_se'].values
tau_cage_2048 = summary_N2048['tau_cage'].values
tau_cage_se_2048 = summary_N2048['tau_cage_se'].values

# Plot N=1024 (solid lines)
ax_b.errorbar(noise_1024, tau_rw_1024, yerr=tau_rw_se_1024, marker='o', markersize=6,
              capsize=3, linewidth=2, linestyle='-', alpha=0.7,
              label=r'$\tau_{\mathrm{rw}}$ (N=1024)', color='#1f77b4')
ax_b.errorbar(noise_1024, tau_rel_1024, yerr=tau_rel_se_1024, marker='s', markersize=6,
              capsize=3, linewidth=2, linestyle='-', alpha=0.7,
              label=r'$\tau_{\mathrm{rel}}$ (N=1024)', color='#ff7f0e')
ax_b.errorbar(noise_1024, tau_cage_1024, yerr=tau_cage_se_1024, marker='^', markersize=6,
              capsize=3, linewidth=2, linestyle='-', alpha=0.7,
              label=r'$\tau_{\mathrm{cage}}$ (N=1024)', color='#2ca02c')

# Plot N=2048 (dashed lines)
ax_b.errorbar(noise_2048, tau_rw_2048, yerr=tau_rw_se_2048, marker='o', markersize=6,
              capsize=3, linewidth=2, linestyle='--', alpha=0.7,
              label=r'$\tau_{\mathrm{rw}}$ (N=2048)', color='#1f77b4', markerfacecolor='white')
ax_b.errorbar(noise_2048, tau_rel_2048, yerr=tau_rel_se_2048, marker='s', markersize=6,
              capsize=3, linewidth=2, linestyle='--', alpha=0.7,
              label=r'$\tau_{\mathrm{rel}}$ (N=2048)', color='#ff7f0e', markerfacecolor='white')
ax_b.errorbar(noise_2048, tau_cage_2048, yerr=tau_cage_se_2048, marker='^', markersize=6,
              capsize=3, linewidth=2, linestyle='--', alpha=0.7,
              label=r'$\tau_{\mathrm{cage}}$ (N=2048)', color='#2ca02c', markerfacecolor='white')

ax_b.set_xlabel('Noise D', fontsize=12)
ax_b.set_ylabel('Timescale', fontsize=12)
ax_b.set_title('Panel B: Timescales vs Noise (Size Robustness)', fontsize=13, fontweight='bold')
ax_b.set_yscale('log')
ax_b.grid(True, alpha=0.3)
ax_b.legend(fontsize=8, loc='upper right', ncol=2)

# ============================================
# Panel C: Relative MSD (N=1024)
# ============================================
ax_c = fig.add_subplot(gs[1, 0])

for noise in selected_noises:
    data = curve_data_N1024[noise]
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
# Panel D: Order Parameter vs Noise (Both N values)
# ============================================
ax_d = fig.add_subplot(gs[1, 1])

# N=1024
phi_1024 = summary_N1024['phi_mean'].values
phi_se_1024 = summary_N1024['phi_se'].values
ax_d.errorbar(noise_1024, phi_1024, yerr=phi_se_1024, marker='o', markersize=8,
              capsize=4, linewidth=2, label='N=1024', color='#1f77b4')

# N=2048
phi_2048 = summary_N2048['phi_mean'].values
phi_se_2048 = summary_N2048['phi_se'].values
ax_d.errorbar(noise_2048, phi_2048, yerr=phi_se_2048, marker='s', markersize=8,
              capsize=4, linewidth=2, label='N=2048', color='#d62728')

# Add vertical line at transition
ax_d.axvline(x=0.0738, color='gray', linestyle='--', alpha=0.5, label='D=0.0738')

ax_d.set_xlabel('Noise D', fontsize=12)
ax_d.set_ylabel('Polarization φ', fontsize=12)
ax_d.set_title('Panel D: Order Parameter vs Noise (Size Comparison)', fontsize=13, fontweight='bold')
ax_d.set_ylim(0, 1)
ax_d.grid(True, alpha=0.3)
ax_d.legend(fontsize=10, loc='upper right')

# Add annotations
ax_d.annotate('Ordered\nPhase', xy=(0.03, 0.8), fontsize=10, ha='center',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax_d.annotate('Disordered\nPhase', xy=(0.12, 0.15), fontsize=10, ha='center',
              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.suptitle('Experiment 2: Topological Vicsek Model (k=7)', 
             fontsize=15, fontweight='bold', y=0.98)

plt.savefig('exp2_figure_v2.png', dpi=300, bbox_inches='tight')
plt.savefig('exp2_figure_v2.pdf', bbox_inches='tight')
print("Figure saved as exp2_figure_v2.png and exp2_figure_v2.pdf")

plt.show()
