import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
from stagpy.stagyydata import StagyyData
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import LogFormatterSciNotation

# --- COLOURMAP SYSTEM ---
#  Try to import Fabio Crameri's colormaps
try:
    from cmcrameri import cm
    HAS_CRAMERI = True
except ImportError:
    HAS_CRAMERI = False


# --- FULL LIST OF RPROF PARAMETERS ---
# Use these strings in 'FIELDS_TO_PLOT' to change the data being visualized.
ALL_RPROF_FIELDS = [
    "r",            # Radial coordinate
    "Tmean",        # Temperature
    "Tmin",         # Min temperature
    "Tmax",         # Max temperature
    "vrms",         # rms velocity
    "vmin",         # Min velocity
    "vmax",         # Max velocity
    "vzabs",        # Radial velocity
    "vzmin",        # Min radial velocity
    "vzmax",        # Max radial velocity
    "vhrms",        # Horizontal velocity
    "vhmin",        # Min horiz velocity
    "vhmax",        # Max horiz velocity
    "etalog",       # Viscosity
    "etamin",       # Min viscosity
    "etamax",       # Max viscosity
    "elog",         # Strain rate
    "emin",         # Min strain rate
    "emax",         # Max strain rate
    "slog",         # Stress
    "smin",         # Min stress
    "smax",         # Max stress
    "whrms",        # Horizontal vorticity
    "whmin",        # Min horiz vorticity
    "whmax",        # Max horiz vorticity
    "wzrms",        # Radial vorticity
    "wzmin",        # Min radial vorticity
    "wzmax",        # Max radial vorticity
    "drms",         # Divergence
    "dmin",         # Min divergence
    "dmax",         # Max divergence
    "enadv",        # Advection
    "endiff",       # Diffusion
    "enradh",       # Radiogenic heating
    "enviscdiss",   # Viscous dissipation
    "enadiabh",     # Adiabatic heating
    "bsmean",       # Basalt content
    "bsmin",        # Min basalt content
    "bsmax",        # Max basalt content
    "rhomean",      # Density
    "rhomin",       # Min density
    "rhomax",       # Max density
    "airmean",      # Air
    "airmin",       # Min air
    "airmax",       # Max air
    "primmean",     # Primordial
    "primmin",      # Min primordial
    "primmax",      # Max primordial
    "ccmean",       # Continental crust
    "ccmin",        # Min continental crust
    "ccmax",        # Max continental crust
    "fmeltmean",    # Melt fraction
    "fmeltmin",     # Min melt fraction
    "fmeltmax",     # Max melt fraction
    "metalmean",    # Metal
    "metalmin",     # Min metal
    "metalmax",     # Max metal
    "gsmean",       # Grain size
    "gsmin",        # Min grain size
    "gsmax",        # Max grain
    "viscdisslog",  # Viscous dissipation
    "viscdissmin",  # Min visc dissipation
    "viscdissmax",  # Max visc dissipation
    "advtot",       # Advection
    "advdesc",      # Downward advection
    "advasc",       # Upward advection
    "tcondmean",    # Conductivity
    "tcondmin",     # Min conductivity
    "tcondmax",     # Max conductivity
    "impmean",      # Impactor fraction
    "impmin",       # Min impactor fraction
    "impmax",       # Max impactor fraction
    "hzmean",       # Harzburgite fraction
    "hzmin",        # Min harzburgite fraction
    "hzmax",        # Max harzburgite fraction
    "TTGmean",      # TTG fraction
    "TTGmin",       # Min TTG fraction
    "TTGmax",       # Max TTG fraction
    "edismean",     # Dislocation creep fraction
    "edismin",      # Min dislocation creep fraction
    "edismax",      # Max dislocation creep fraction
    "egbsmean",     # Grain boundary sliding fraction
    "egbsmin",      # Min grain boundary sliding fraction
    "egbsmax",      # Max grain boundary sliding fraction
    "ePeimean",     # Peierls creep fraction
    "ePeimin",      # Min Peierls creep fraction
    "ePeimax",      # Max Peierls creep fraction
    "eplamean",     # Plasticity fraction
    "eplamin",      # Min plasticity fraction
    "eplamax",      # Max plasticity fraction
    "dr",           # Cell thicknesses
    "diff",         # Diffusion flux
    "diffs",        # Scaled diffusion flux
    "advts",        # Scaled advection flux
    "advds",        # Scaled downward advection flux
    "advas",        # Scaled upward advection flux
    "energy"        # Total heat flux
]



# --- USER CONFIGURATION ---
# Define the path to your StagYY 'archive' directory.
DATA_ROOT = Path("/media/aritro/f522493b-003a-404d-a839-3e0925c674b6/Aritro/StagYY/runs/festus/venus_imp6/archive/")

# Fields to visualize (Y-axis = Depth, X-axis = Time, Color = Field Value)
FIELDS_TO_PLOT = ["bsmean", "fmeltmax", "elog"]   

# Manual limits for specific fields to ensure consistency across different runs.
FIELD_LIMITS = {
    "Tmax": (0, 5500),
    "vmax": (1e-6, 1.2e-3),
    "vrms": (1e-8, 1e-2), 
    "fmeltmax": (0, 1.0),
    "etalog": (1e18, 1e24),
}

# Downsampling: 1 = every step, 10 = every 10th step.
SAMPLE_STEP = 1  

# Colormap Preferences
USE_CRAMERI = True
SEQUENTIAL_MAP = "batlow"   # Good for T, viscosity, composition
DIVERGING_MAP  = "roma"  # Good for velocity, divergence, flux

def run_visualizer():
    # --- 1. INITIALIZATION ---
    print(f"{'='*60}\n RPROF-TIME \n{'='*60}")
    
    if not DATA_ROOT.exists():
        print(f"CRITICAL ERROR: The path '{DATA_ROOT}' does not exist.")
        print("Please check your 'DATA_ROOT' configuration at the top of the script.")
        return

    print(f"[*] Loading StagYY Data at: {DATA_ROOT.name}")
    sdat = StagyyData(DATA_ROOT)
    
    snaps_to_process = sdat.snaps[::SAMPLE_STEP]

    # Data structures for plotting
    times, depths = [], None
    plot_data = {f: [] for f in FIELDS_TO_PLOT}
    field_meta = {f: {"title": "", "log": False} for f in FIELDS_TO_PLOT}

    # --- 2. DATA COLLECTION LOOP ---
    print(f"[*] Reading {len(FIELDS_TO_PLOT)} fields across snapshots...")

    for idx, snap in enumerate(snaps_to_process):
        try:
            # Progress update in console
            if (idx + 1) % 10 == 0:
                print(f"    > Reading Snapshot {idx+1} (Step: {snap.istep})", end='\r')

            # Unit Conversion: Seconds to Megayears (Myr)
            current_time = snap.time / (3600 * 24 * 365.25 * 1e6)
            
            # Temporary storage to ensure ALL fields exist for this snapshot before adding
            temp_field_data = {}
            for field in FIELDS_TO_PLOT:
                rprof = snap.rprofs[field] # This is where 'Details: 0' likely happened
                temp_field_data[field] = rprof.values
                
                # Capture metadata only once
                if not field_meta[field]["title"]:
                    desc, unit = rprof.meta.description, rprof.meta.dim
                    field_meta[field]["title"] = f"{desc} ({unit})" if unit else desc
                    
                    log_keywords = ["log", "eta", "slog", "visc", "vrms", "vmax", "vmin"]
                    if any(k in field.lower() for k in log_keywords):
                        field_meta[field]["log"] = True

                # Calculate depths (m to km) once
                if depths is None:
                    r_surf = np.max(rprof.rad)
                    depths = (r_surf - rprof.rad) / 1e3 

            # If we reached here, all fields were found successfully
            times.append(current_time)
            for field in FIELDS_TO_PLOT:
                plot_data[field].append(temp_field_data[field])
        
        except Exception as e:
            # More descriptive error reporting
            print(f"\n[!] WARNING: Skipping snapshot {idx} (Step {snap.istep})")
            print(f"    Error Type: {type(e).__name__} | Details: {e}")
            continue

    # --- 3. FINAL VALIDATION ---
    if not times:
        print("\n\nERROR: No data was collected. Possible reasons:")
        print(f"1. Fields {FIELDS_TO_PLOT} do not exist in these snapshots.")
        print("2. The StagYY output files are corrupted or empty.")
        return

    print(f"\n[*] Data loading complete ({len(times)} valid snapshots). Generating plots...")

    # --- 4. PLOTTING ENGINE ---
    fig, axes = plt.subplots(len(FIELDS_TO_PLOT), 1, 
                             figsize=(12, 4 * len(FIELDS_TO_PLOT)), 
                             sharex=True, squeeze=False)

    for i, field in enumerate(FIELDS_TO_PLOT):
        ax = axes[i, 0]
        data_matrix = np.array(plot_data[field]).T
        
        vmin, vmax = FIELD_LIMITS.get(field, (None, None))
        
        if field_meta[field]["log"]:
            data_matrix = np.clip(data_matrix, 1e-35, None) 
            if vmin is None or vmin <= 0: 
                vmin = np.nanmin(data_matrix[data_matrix > 0])
            if vmax is None: 
                vmax = np.nanmax(data_matrix)
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)

        diverging_keywords = ["vz", "vh", "drms", "adv"]
        is_diverging = any(k in field.lower() for k in diverging_keywords)

        if USE_CRAMERI and HAS_CRAMERI:
            cmap = getattr(cm, DIVERGING_MAP) if is_diverging else getattr(cm, SEQUENTIAL_MAP)
        else:
            cmap = 'RdBu_r' if is_diverging else 'magma'

        im = ax.pcolormesh(np.array(times), depths, data_matrix, 
                           shading='auto', cmap=cmap, norm=norm)
        
        ax.set_title(field_meta[field]["title"], fontweight='bold', fontsize=13, loc='center')
        ax.set_ylabel("Depth (km)", fontsize=11)
        ax.invert_yaxis() 
        
        if field_meta[field]["log"]:
            formatter = LogFormatterSciNotation(base=10, labelOnlyBase=False)
            cbar = plt.colorbar(im, ax=ax, pad=0.01, aspect=15, format=formatter)
        else:
            cbar = plt.colorbar(im, ax=ax, pad=0.01, aspect=15)
            if (vmax and (vmax > 1000 or vmax < 0.01)):
                cbar.formatter.set_powerlimits((0, 0))

    axes[-1, 0].set_xlabel("Time (Myr)", fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.95, 1])

    save_name = f"evol_{'_'.join(FIELDS_TO_PLOT)}.png"
    fig.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"[*] SUCCESS: Plot saved as '{save_name}'")
    
    plt.show()

if __name__ == "__main__":
    try:
        run_visualizer()
    except KeyboardInterrupt:
        print("\n[!] Execution interrupted by user.")
        sys.exit(0)