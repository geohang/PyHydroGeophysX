#!/usr/bin/env python
"""
Generate placeholder thumbnails for examples that don't have auto-generated images.

This script creates simple, representative static thumbnails for use with
sphinx-gallery when plot_gallery=False (examples are not executed during build).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from matplotlib.collections import PatchCollection

# Output directory for thumbnails
THUMB_DIR = os.path.join(os.path.dirname(__file__), '..', 
                         'docs', 'source', 'auto_examples', 'images', 'thumb')
IMAGES_DIR = os.path.join(os.path.dirname(__file__), '..', 
                          'docs', 'source', 'auto_examples', 'images')

os.makedirs(THUMB_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)


def create_tdem_thumbnail():
    """Create a TDEM workflow thumbnail showing 1D layered earth and decay curve."""
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    
    # Create a simple 1D layered model representation
    layers = [0, 2, 5, 10, 15]
    colors = ['#8B4513', '#A0522D', '#CD853F', '#DEB887']
    
    for i in range(len(layers)-1):
        ax.axhspan(layers[i], layers[i+1], color=colors[i], alpha=0.7)
    
    # Add TDEM coil symbol at top
    circle = Circle((0.5, -1), 0.5, fill=False, color='red', linewidth=3)
    ax.add_patch(circle)
    
    # Add arrow for EM field
    ax.annotate('', xy=(0.5, 1), xytext=(0.5, 0),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    
    ax.set_xlim(-1, 2)
    ax.set_ylim(15, -3)
    ax.set_ylabel('Depth (m)', fontsize=10)
    ax.set_title('TDEM\nWorkflow', fontsize=12, fontweight='bold')
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    plt.tight_layout()
    
    # Save thumbnail
    thumb_path = os.path.join(THUMB_DIR, 'sphx_glr_Ex_TDEM_workflow_thumb.png')
    fig.savefig(thumb_path, dpi=100, bbox_inches='tight', facecolor='white')
    print(f"Created: {thumb_path}")
    
    # Also save a larger version for inline use
    fig_path = os.path.join(IMAGES_DIR, 'Ex_TDEM_workflow_fig_01.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Created: {fig_path}")
    
    plt.close(fig)


def create_3d_ert_thumbnail():
    """Create a 3D ERT thumbnail showing mesh/electrode concept."""
    fig = plt.figure(figsize=(4, 3), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a simple 3D mesh representation
    # Ground surface
    x = np.linspace(0, 10, 5)
    y = np.linspace(0, 10, 5)
    X, Y = np.meshgrid(x, y)
    Z = 0.5 * np.sin(X * 0.3) * np.cos(Y * 0.3)
    
    ax.plot_surface(X, Y, Z, alpha=0.3, color='brown')
    
    # Electrode points
    ex = np.linspace(1, 9, 5)
    ey = np.full(5, 5)
    ez = 0.5 * np.sin(ex * 0.3) * np.cos(ey * 0.3)
    ax.scatter(ex, ey, ez, c='red', s=50, marker='o', label='Electrodes')
    
    # Subsurface layer
    Z_sub = Z - 5
    ax.plot_surface(X, Y, Z_sub, alpha=0.2, color='blue')
    
    ax.set_xlabel('X', fontsize=8)
    ax.set_ylabel('Y', fontsize=8)
    ax.set_zlabel('Z', fontsize=8)
    ax.set_title('3D ERT\nForward', fontsize=12, fontweight='bold')
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    thumb_path = os.path.join(THUMB_DIR, 'sphx_glr_Ex_3D_ERT_forward_thumb.png')
    fig.savefig(thumb_path, dpi=100, bbox_inches='tight', facecolor='white')
    print(f"Created: {thumb_path}")
    
    # Also save inline figure
    fig_path = os.path.join(IMAGES_DIR, 'Ex_3D_ERT_forward_fig_01.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Created: {fig_path}")
    
    plt.close(fig)


def create_tdem_additional_figures():
    """Create additional TDEM figures for inline documentation."""
    # Figure 2: TDEM Decay Curve
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    
    times = np.logspace(-5, -2, 50)
    # Simulate a typical TDEM decay
    signal = 1e-9 * np.exp(-times * 1000) * (1 + 0.1 * np.random.randn(len(times)))
    
    ax.loglog(times * 1000, np.abs(signal), 'ko-', markersize=5, label='Observed')
    ax.loglog(times * 1000, 1e-9 * np.exp(-times * 1000), 'r-', lw=2, label='Predicted')
    
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('|Bz| (T)', fontsize=12)
    ax.set_title('TDEM Sounding - Synthetic Data', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, 'Ex_TDEM_workflow_fig_02.png'), 
                dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Created: Ex_TDEM_workflow_fig_02.png")
    plt.close(fig)
    
    # Figure 3: Inversion Result
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    
    depths = np.array([0, 2, 5, 10, 15, 25])
    true_cond = np.array([0.01, 0.02, 0.005, 0.001, 0.0005])
    recovered = true_cond * (1 + 0.15 * np.random.randn(len(true_cond)))
    
    for i in range(len(true_cond)):
        ax.fill_betweenx([depths[i], depths[i+1]], 1e-4, true_cond[i], 
                        alpha=0.3, color='black')
        ax.plot([true_cond[i], true_cond[i]], [depths[i], depths[i+1]], 
               'k-', lw=3, label='True' if i == 0 else '')
        ax.plot([recovered[i], recovered[i]], [depths[i], depths[i+1]], 
               'r-', lw=2, label='Recovered' if i == 0 else '')
    
    ax.set_xscale('log')
    ax.set_xlabel('Conductivity (S/m)', fontsize=12)
    ax.set_ylabel('Depth (m)', fontsize=12)
    ax.set_title('TDEM Inversion Result', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    
    plt.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, 'Ex_TDEM_workflow_fig_03.png'), 
                dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Created: Ex_TDEM_workflow_fig_03.png")
    plt.close(fig)
    
    # Figure 4: Model Comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=100)
    
    ax1 = axes[0]
    for i in range(len(true_cond)):
        ax1.fill_betweenx([depths[i], depths[i+1]], 1e-4, true_cond[i], 
                         alpha=0.3, color='black')
        ax1.plot([true_cond[i], true_cond[i]], [depths[i], depths[i+1]], 'k-', lw=3)
        ax1.plot([recovered[i], recovered[i]], [depths[i], depths[i+1]], 'r-', lw=2)
    ax1.set_xscale('log')
    ax1.set_xlabel('Conductivity (S/m)')
    ax1.set_ylabel('Depth (m)')
    ax1.set_title('Conductivity Comparison')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    residuals = np.random.randn(50)
    ax2.plot(residuals, 'ro-', markersize=4)
    ax2.axhline(0, color='k', ls='-')
    ax2.axhline(2, color='gray', ls='--')
    ax2.axhline(-2, color='gray', ls='--')
    ax2.fill_between(range(50), -2, 2, alpha=0.1, color='green')
    ax2.set_xlabel('Data Point')
    ax2.set_ylabel('Normalized Residual')
    ax2.set_title('Data Residuals')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, 'Ex_TDEM_workflow_fig_04.png'), 
                dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Created: Ex_TDEM_workflow_fig_04.png")
    plt.close(fig)
    
    # Figure 5: Data Fit
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=100)
    
    ax1 = axes[0]
    ax1.loglog(times * 1000, np.abs(signal), 'ko', markersize=6, label='Observed')
    ax1.loglog(times * 1000, 1e-9 * np.exp(-times * 1000), 'r-', lw=2, label='Predicted')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('|Bz| (T)')
    ax1.set_title('Data Fit (χ² = 1.05)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.semilogx(times * 1000, residuals[:len(times)], 'ro-', markersize=4)
    ax2.axhline(0, color='k', ls='-')
    ax2.fill_between(times * 1000, -2, 2, alpha=0.1, color='green')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Normalized Residual')
    ax2.set_title('Residuals (±2σ)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, 'Ex_TDEM_workflow_fig_05.png'), 
                dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Created: Ex_TDEM_workflow_fig_05.png")
    plt.close(fig)


def create_3d_ert_additional_figures():
    """Create additional 3D ERT figures for inline documentation."""
    # Figure 2: Mesh visualization
    fig = plt.figure(figsize=(10, 8), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    # Create mesh-like structure
    x = np.linspace(0, 100, 10)
    y = np.linspace(0, 100, 10)
    X, Y = np.meshgrid(x, y)
    
    # Surface with some topography
    Z = 5 * np.sin(X * 0.05) * np.cos(Y * 0.05) + 50
    
    # Plot surfaces at different depths
    ax.plot_surface(X, Y, Z, alpha=0.5, color='brown', label='Surface')
    ax.plot_surface(X, Y, Z - 20, alpha=0.3, color='gray')
    ax.plot_surface(X, Y, Z - 40, alpha=0.2, color='blue')
    
    # Electrodes
    ex = np.linspace(10, 90, 8)
    ey = np.full(8, 50)
    ez = 5 * np.sin(ex * 0.05) * np.cos(ey * 0.05) + 50
    ax.scatter(ex, ey, ez, c='red', s=100, marker='v')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Mesh with Topography')
    
    plt.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, 'Ex_3D_ERT_forward_fig_02.png'), 
                dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Created: Ex_3D_ERT_forward_fig_02.png")
    plt.close(fig)


if __name__ == '__main__':
    print("Generating placeholder thumbnails...")
    print("-" * 50)
    
    create_tdem_thumbnail()
    create_3d_ert_thumbnail()
    create_tdem_additional_figures()
    create_3d_ert_additional_figures()
    
    print("-" * 50)
    print("Done! Thumbnails created in:", THUMB_DIR)
    print("Figures created in:", IMAGES_DIR)
