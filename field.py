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
plot_mode = "snapshot"   # Set to "time" or "snapshot" 
target_time_Gyr = 1.9    # Used if plot_mode is "time"
target_snapshot = 4191  # Used if plot_mode is "snapshot"

field_to_plot = "edot"    

# --- CONFIGURATION ---
# Auto-detect log scale for these fields
LOG_FIELDS = ["eta", "edot", "sII", "v1", "v2", "v3", "meltvel", "wtr", "meltrate"]

# FIELD LIMITS
FIELD_LIMITS = {
    "T": (300, 4000),
    "basalt": (0.0, 1.0),
    "eta": (1e18, 1e25),
    "edot": (1e-18, 1e-12),
    "meltfrac": (0.0, 0.2),
}

FIELD_LABELS = {
    "T": "Temperature", "eta": "Viscosity", "basalt": "Basalt Fraction",
    "v1": "Velocity (x)", "edot": "Strain Rate", "c": "Composition"
}

data_path = Path("/media/aritro/f522493b-003a-404d-a839-3e0925c674b6/Aritro/StagYY/runs/euler/venus_i_01/archive/")
sdat = StagyyData(data_path)
folder_name = data_path.parent.name 
SEC_PER_GYR = 1e9 * 365.25 * 24 * 3600

# --- 1. SELECTION LOGIC ---
snap_number = None
actual_time_Gyr = None

if plot_mode == "time":
    times, indices = [], []
    for snap in sdat.snaps:
        try:
            t = snap.time if snap.time is not None else snap.timeinfo["time"]
            times.append(t); indices.append(snap.isnap)
        except: continue
    
    times, indices = np.array(times), np.array(indices)
    idx = np.abs(times - (target_time_Gyr * SEC_PER_GYR)).argmin()
    snap_number, actual_time_Gyr = int(indices[idx]), times[idx] / SEC_PER_GYR
else:
    snapshot = sdat.snaps[target_snapshot]
    snap_number = target_snapshot
    t = snapshot.time if snapshot.time is not None else snapshot.timeinfo["time"]
    actual_time_Gyr = t / SEC_PER_GYR

# --- 2. GENERATE THE PLOT ---
if snap_number is not None:
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
            fig, ax, mesh, cbar = sp_field.plot_scalar(snapshot, field_to_plot, vmin=f_min, vmax=f_max)
        
        # Visual Styling
        unit = snapshot.fields[field_to_plot].meta.dim
        label = FIELD_LABELS.get(field_to_plot, field_to_plot)
        cbar.set_label(f"{label} [{unit}]")
        
        # TIME LABEL ON PLOT (Kept same as before)
        ax.text(0.5, 0.5, f"{actual_time_Gyr:.3f} Gyr", 
                transform=ax.transAxes, ha="center", va="center", 
                fontsize=20, color="black", 
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
        
        fig.set_size_inches(10, 6)
        plt.tight_layout()
        
        #NAMING SCHEME: [folder]_[field]_snap-[number]_[time]-Gyr.png
        
        save_name = f"{folder_name}_{field_to_plot}_snap-{snap_number}_{actual_time_Gyr:.1f}-Gyr.png"
        fig.savefig(save_name, dpi=300)
        plt.close(fig)
        print(f"Successfully saved: {save_name}")

    except Exception as e:
        print(f"An error occurred during plotting: {e}")