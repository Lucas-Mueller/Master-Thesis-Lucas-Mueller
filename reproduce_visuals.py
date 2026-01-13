
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# Add repo root to path
here = Path.cwd().resolve()
sys.path.insert(0, str(here))

from experiment_execution.utils_experiment_execution.style import apply_theme, COLORS, FONT_SIZES, FIG_SIZES
from experiment_execution.utils_experiment_execution.visualizations import (
    plot_income_preference_bars, 
    plot_rounds_to_outcome,
    plot_floor_constraint_distribution
)

def main():
    print("Applying theme...")
    apply_theme()
    
    # Create Verification Directory
    verify_dir = Path("verification_plots")
    verify_dir.mkdir(exist_ok=True)
    
    # --- Test 1: Plot Income Preference Bars (Switcher Analysis) ---
    print("Generating Income Preference Bars...")
    # Mock data structure matching the notebook
    summary_data = {
        "Income Class": ["High", "Medium-High", "Medium", "Medium-Low", "Low"],
        "Count": [5, 10, 50, 25, 10]
    }
    summary_df = pd.DataFrame(summary_data)
    
    # Create mock count_long dataframe
    # Categories: Changed Preference, Maintained Preference
    count_long = []
    classes = ["High", "Medium-High", "Medium", "Medium-Low", "Low"]
    for c in classes:
        # Simulate some switchers
        count_long.append({"Income Class": c, "Preference Group": "Changed Preference", "Count": 2})
        count_long.append({"Income Class": c, "Preference Group": "Maintained Preference", "Count": 8})
    count_long_df = pd.DataFrame(count_long)
    
    # Create mock percent_long dataframe
    percent_long = []
    for c in classes:
        percent_long.append({"Income Class": c, "Preference Group": "Changed Preference", "Percent": 20.0})
        percent_long.append({"Income Class": c, "Preference Group": "Maintained Preference", "Percent": 80.0})
    percent_long_df = pd.DataFrame(percent_long)
    
    # Override plt.show to save instead
    original_show = plt.show
    def save_plot_1():
        plt.gcf().savefig(verify_dir / "income_preference_bars.png")
        print(f"Saved {verify_dir / 'income_preference_bars.png'}")
        
    plt.show = save_plot_1
    
    try:
        plot_income_preference_bars(
            summary_df,
            count_long_df,
            percent_long_df,
            title_suffix="Verification",
            title="Income Preference Verification"
        )
    except Exception as e:
        print(f"Failed to plot income bars: {e}")
    finally:
        plt.show = original_show

    # --- Test 2: Rounds to Outcome ---
    print("\nGenerating Rounds to Outcome...")
    # Mock metrics dataframe
    metrics_data = {
        "consensus_reached": [True] * 20 + [False] * 5,
        "rounds_to_outcome": np.random.randint(1, 10, 25)
    }
    run_metrics = pd.DataFrame(metrics_data)
    
    def save_plot_2():
        plt.gcf().savefig(verify_dir / "rounds_to_outcome.png")
        print(f"Saved {verify_dir / 'rounds_to_outcome.png'}")
        
    plt.show = save_plot_2
    
    try:
        plot_rounds_to_outcome(
            run_metrics,
            title_suffix="Verification",
            title="Rounds Verification"
        )
    except Exception as e:
        print(f"Failed to plot rounds: {e}")
    finally:
        plt.show = original_show

    # --- Test 3: Floor Constraint Distribution (LaTeX Check) ---
    print("\nGenerating Floor Constraint Distribution...")
    # Mock vote rounds
    vote_data = {
        "floor_constraint": np.random.randint(0, 50000, 100),
        "consensus_reached": [True] * 80 + [False] * 20,
        "agreed_constraint": np.random.randint(0, 50000, 100)
    }
    vote_rounds = pd.DataFrame(vote_data)
    
    def save_plot_3():
        plt.gcf().savefig(verify_dir / "floor_constraints.png")
        print(f"Saved {verify_dir / 'floor_constraints.png'}")
        
    plt.show = save_plot_3
    
    try:
        plot_floor_constraint_distribution(
            vote_rounds,
            title_suffix="Verification",
            title="Floor Constraints Verification"
        )
        print("Success: Floor constraint plot with $ labels rendered correctly.")
    except Exception as e:
        print(f"FAILED: Floor constraint plot error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        plt.show = original_show

if __name__ == "__main__":
    main()
