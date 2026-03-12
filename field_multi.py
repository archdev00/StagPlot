import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from stagpy.stagyydata import StagyyData
from stagpy import field as sp_field

"""
--- REFERENCE: STAGYY FIELD EXPLANATIONS ---
PHYSICAL/THERMODYNAMIC:
    T: Temperature                  p: Pressure
    rho: Density                    eta: Viscosity (Log)
    Tcond: Thermal Conductivity     rho4rhs: Density term in RHS
    age: Material age               fSiO2, fMgO, fFeO, fXO, fFeR: Mineral/Oxide fractions

KINEMATICS & DYNAMICS:
    v1, v2, v3: Velocity (x, y, z)  edot: Strain rate (Log)
    sII: 2nd Stress Invariant (Log) s1val: Principal stress eigenvalue
    sx1, sx2, sx3: Stress vectors   stream: Stream function (2D)
    meltvel: Melt velocity (Log)

COMPOSITION & MELTING:
    c: Composition (prim/basalt)    basalt: Basalt fraction
    harzburgite: Harzburgite frac   prim: Primordial layer
    meltfrac: Current melt degree   meltrate: Melting rate
    meltcompo: Melt composition     nmelt: N melt
    cFe: FeO content                hpe: HPE content
    wtr: Water concentration (Log)  contID: ID of continents

NUMERICS:
    rs1, rs2, rs3: Momentum residue rsc: Continuity residue
"""

# --- USER INPUT ---
field_to_plot = "T"  
snap_min = 1700
snap_max = 3759

# --- TOGGLE ---
mode = "constant_frame" # Options: "constant_time" or "constant_frame"

# --- CONSTANT_TIME SETTINGS ---
dt_Gyr = 0.001 

# --- CONSTANT_FRAME SETTINGS ---
snap_step = 1   # 1 = every snapshot, 10 = every 10th snapshot, etc.

# --- CONFIGURATION ---
# Auto-detect log scale for these fields
LOG_FIELDS = ["eta", "edot", "sII", "v1", "v2", "v3", "meltvel", "wtr", "meltrate"]

# FIELD LIMITS (Min, Max)
FIELD_LIMITS = {
    "T": (300, 4000),
    "basalt": (0.0, 1.0),
    "eta": (1e18, 1e25),
    "edot": (1e-18, 1e-12),
    "meltfrac": (0.0, 0.2),
}

FIELD_LABELS = {
    "T": "Temperature", 
    "eta": "Viscosity", 
    "basalt": "Basalt Fraction",
    "v1": "Velocity (x)", 
    "edot": "Strain Rate", 
    "c": "Composition"
}

# --- PLOT SETTINGS ---
fig_width = 7
fig_height = 6

# --- DATA PATH ---
data_path = Path("/media/aritro/f522493b-003a-404d-a839-3e0925c674b6/Aritro/StagYY/runs/euler/venus_i_01/archive")
sdat = StagyyData(data_path)

# Extract folder name (e.g., 'venus_02')
folder_name = data_path.parent.name 

# Create Directory: [folder_name]_frames_[field_to_plot]_[mode]
output_dir = Path(f"{folder_name}_frames_{field_to_plot}_{mode}")
output_dir.mkdir(parents=True, exist_ok=True)

SEC_PER_GYR = 1e9 * 365.25 * 24 * 3600

# --- 1. PREPARE THE FRAME LIST ---
frames_to_render = [] 

print(f"Scanning snapshots {snap_min} to {snap_max} in {folder_name}...")
available_snaps = []
available_times = []

for n in range(snap_min, snap_max + 1):
    try:
        snap = sdat.snaps[n]
        # StagPy fix: use snap.time or fallback to snap.timeinfo["time"]
        t = snap.time
        if t is None:
            t = snap.timeinfo["time"]
            
        available_snaps.append(snap.isnap)
        available_times.append(t)
    except:
        continue

available_snaps = np.array(available_snaps)
available_times = np.array(available_times)

if len(available_snaps) == 0:
    print("Error: No data found in that snapshot range.")
else:
    if mode == "constant_time":
        t_start = available_times.min()
        t_end = available_times.max()
        target_times = np.arange(t_start, t_end, dt_Gyr * SEC_PER_GYR)
        
        for t_target in target_times:
            idx = np.abs(available_times - t_target).argmin()
            # Store (snapshot_number, actual_time_from_file)
            frames_to_render.append((int(available_snaps[idx]), available_times[idx]))
    else:
        # constant_frame mode
        for i in range(0, len(available_snaps), snap_step):
            frames_to_render.append((int(available_snaps[i]), available_times[i]))

    print(f"Mode: {mode.upper()}. Generating {len(frames_to_render)} frames.")

    # --- 2. RENDER THE FRAMES ---
    for i, (snap_number, t_val) in enumerate(frames_to_render):
        try:
            snapshot = sdat.snaps[snap_number]
            
            # Unpack limits (defaults to None, None if field not in dict)
            f_min, f_max = FIELD_LIMITS.get(field_to_plot, (None, None))
            
            if field_to_plot in LOG_FIELDS:
                # Ensure f_min is positive for LogNorm
                log_min = f_min if f_min is not None else 1e-5
                norm = colors.LogNorm(vmin=log_min, vmax=f_max)
                # Pass ONLY norm to avoid Matplotlib conflict
                fig, ax, mesh, cbar = sp_field.plot_scalar(snapshot, field_to_plot, norm=norm)
            else:
                # Linear scaling uses direct limits
                fig, ax, mesh, cbar = sp_field.plot_scalar(snapshot, field_to_plot, 
                                                         vmin=f_min, vmax=f_max)
            
            # Colorbar labels
            unit = snapshot.fields[field_to_plot].meta.dim
            display_name = FIELD_LABELS.get(field_to_plot, field_to_plot)
            cbar.set_label(f"{display_name} [{unit}]")
            
            # Time Label on Plot
            time_Gyr = t_val / SEC_PER_GYR
            label_text = f"{time_Gyr:.3f} Gyr"
            
            ax.text(0.5, 0.5, label_text, 
                    transform=ax.transAxes, ha="center", va="center", 
                    fontsize=24, color="black", 
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
            
            fig.set_size_inches(fig_width, fig_height)
            plt.tight_layout()
            
            # FILE NAMING SCHEME
            file_name = f"frame_{snap_number:05d}.png"
            fig.savefig(output_dir / file_name, dpi=300)
            plt.close(fig) 
            
            if i % 10 == 0 or i == len(frames_to_render) - 1:
                print(f"Saved: {file_name}")
                
        except Exception as e:
            print(f"Error at Snap {snap_number}: {e}")
            plt.close()