import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.ticker import LogFormatterSciNotation

# StagPy is the primary library for handling StagYY output
from stagpy.stagyydata import StagyyData

# --- 1. CONSTANTS & COMPATIBILITY ---
SECONDS_IN_YEAR = 3.15576e7
YEARS_IN_MYR = 1e6

# Try to import Crameri colormaps for better perceptual scaling
try:
    from cmcrameri import cm
    HAS_CRAMERI = True
except ImportError:
    HAS_CRAMERI = False

# --- 2. CONFIGURATION ---
# MODE: "SNAPSHOTS" (Compare different times in ONE run) 
#       "RUNS" (Compare the same time/snapshot across MULTIPLE runs)
PLOT_MODE = "RUNS" 

# TIME SELECTION:
# If TIME_TARGETS has values, the script ignores 'snapshot_list' and finds 
# the closest available data to these specific times (in Myr).
TIME_TARGETS = [2] # [1, 2, 3]
snapshot_list = [1400] # [1400, 1500] Fallback if TIME_TARGETS is empty

# DATA SOURCE:
# Provide a label and the system path to the StagYY output directory.
RUN_PATHS = {
    "Venus_Imp6": "/media/aritro/f522493b-003a-404d-a839-3e0925c674b6/Aritro/StagYY/runs/festus/venus_imp6/archive/",
    "Venus_Imp5": "/media/aritro/f522493b-003a-404d-a839-3e0925c674b6/Aritro/StagYY/runs/festus/venus_imp5/archive/", 
}

# PLOT SETTINGS:
field_to_plot = "Tmean"  # Choose from the ALL_RPROF_FIELDS list below

# MANUAL AXIS LIMITS:
FIELD_LIMITS = {
    "etalog": (1e18, 1e22), 
    "vrms": (1e-8, 1e-2),   
    "fmeltmean": (0, 1),
}

# VISUAL STYLING:
LINE_STYLES = ["-", "--", "-.", ":"]
USE_CRAMERI = True
CRAMERI_MAP = "nuuk"

# --- 3. REFERENCE: ALL AVAILABLE RPROF FIELDS ---
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


# --- 4. HELPER FUNCTIONS ---

def find_closest_snap(sdata, target_myr):
    """Finds integer snapshot index closest to target time in Myr."""
    target_sec = target_myr * YEARS_IN_MYR * SECONDS_IN_YEAR
    try:
        times = [s.time for s in sdata.snaps]
        snaps = [s.isnap for s in sdata.snaps]
        idx = np.argmin(np.abs(np.array(times) - target_sec))
        return snaps[idx]
    except Exception as e:
        print(f"   [!] Error mapping time {target_myr} Myr to a snapshot: {e}")
        return None

# --- 5. MAIN EXECUTION BLOCK ---

def main():
    print(f"{'='*60}\n       RPROF       \n{'='*60}")
    print(f"Target Field: {field_to_plot}")
    print(f"Mode:         {PLOT_MODE}")

    try:
        fig, ax = plt.subplots(figsize=(7, 9))
        labels_set = False
        
        print(f"Attempting to load {len(RUN_PATHS)} run(s)...")
        sims = {}
        for name, path in RUN_PATHS.items():
            if not Path(path).exists():
                print(f"   [!] FAILED: Path for '{name}' does not exist: {path}")
                continue
            sims[name] = StagyyData(Path(path))
            print(f"   [+] Loaded: {name}")

        if not sims:
            raise RuntimeError("No valid simulation data could be loaded. Check your RUN_PATHS.")

        # Determine Data Iterator
        iterator = []
        if PLOT_MODE == "SNAPSHOTS":
            run_name = list(sims.keys())[0]
            sdata = sims[run_name]
            active_snaps = [find_closest_snap(sdata, t) for t in TIME_TARGETS] if TIME_TARGETS else snapshot_list
            iterator = [(run_name, snap) for snap in active_snaps if snap is not None]
        else:
            for name, sdata in sims.items():
                active_snaps = [find_closest_snap(sdata, t) for t in TIME_TARGETS] if TIME_TARGETS else snapshot_list
                for s in active_snaps:
                    if s is not None:
                        iterator.append((name, s))

        num_plots = len(iterator)
        line_colors = [None] * num_plots
        if USE_CRAMERI and HAS_CRAMERI:
            cmap_obj = getattr(cm, CRAMERI_MAP)
            line_colors = [cmap_obj(i / (num_plots - 1)) if num_plots > 1 else cmap_obj(0.5) for i in range(num_plots)]

        # --- Plotting Loop ---
        print(f"\nProcessing {num_plots} profiles...")
        for idx, (run_label, isnap) in enumerate(iterator):
            try:
                # Step A: Access snapshot and extract the profile directly
                snapshot = sims[run_label].snaps[isnap]
                rprof_obj = snapshot.rprofs[field_to_plot]
                
                # Step B: Data Extraction
                time_myr = snapshot.time / (SECONDS_IN_YEAR * YEARS_IN_MYR)
                radius = rprof_obj.rad / 1e6
                values = rprof_obj.values
                
                # Step C: Styling
                l_style = LINE_STYLES[idx % len(LINE_STYLES)]
                legend_label = f"{run_label} ({time_myr:.1f} Myr)" if PLOT_MODE == "RUNS" else f"{time_myr:.1f} Myr"
                
                ax.plot(values, radius, label=legend_label, linewidth=1.8, linestyle=l_style, color=line_colors[idx])
                print(f"   [OK] {run_label} | Snap {isnap} ({time_myr:.1f} Myr)")

                # Step D: Labels & Formatting
                if not labels_set:
                    description = rprof_obj.meta.description
                    unit = rprof_obj.meta.dim
                    if "eta" in field_to_plot and unit == "Pa": unit = "Pa s"

                    ax.set_xlabel(f"{description} [{unit}]" if unit else description, fontsize=12)
                    ax.set_ylabel("Radius [10$^6$ m]", fontsize=12)
                    
                    log_keywords = ["log", "eta", "slog", "visc", "vrms", "strain"]
                    if any(k in field_to_plot.lower() for k in log_keywords):
                        ax.set_xscale('log')
                        ax.xaxis.set_major_formatter(LogFormatterSciNotation())
                    labels_set = True

            except Exception as e:
                print(f"   [!] Error: Failed to process {run_label} Snap {isnap}. Detail: {e}")
                continue

        # --- Final Polish ---
        if field_to_plot in FIELD_LIMITS:
            ax.set_xlim(FIELD_LIMITS[field_to_plot])
        
        ax.set_ylim(3.0, 6.2)
        ax.legend(loc='best', frameon=True, fontsize=10)
        ax.grid(True, which="both", ls="-", alpha=0.2)
        
        title_mode = f"Comparison of {len(RUN_PATHS)} Runs" if PLOT_MODE == "RUNS" else f"Evolution: {run_label}"
        ax.set_title(f"{title_mode}\nField: {field_to_plot}", fontsize=14)
        
        plt.tight_layout()
        save_name = f"rprof_{field_to_plot}.png"
        fig.savefig(save_name, dpi=300)
        print(f"\n[SUCCESS] Figure saved as: {save_name}")
        plt.show()

    except Exception as e:
        print(f"\n{'#'*60}\n CRITICAL ERROR IN MAIN LOOP:\n {e}\n{'#'*60}")

if __name__ == "__main__":
    main()