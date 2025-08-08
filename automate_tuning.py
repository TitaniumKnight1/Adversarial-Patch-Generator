# =================================================================================================
#           AUTOMATED HYPERPARAMETER TUNING SCRIPT
# =================================================================================================
#
# This script automates the process of finding the optimal noise parameters for
# generating adversarial patches. It systematically iterates through a predefined
# grid of parameters, runs a shortened training and evaluation for each combination,
# and records the results.
#
# How it Works:
# 1. Defines a `PARAMETER_GRID` with ranges of values for noise parameters.
# 2. For each combination of parameters:
#    a. It dynamically updates the `config.json` file.
#    b. It calls the main training script (`train_patch.py`) as a subprocess.
#       The training script's interactive UI is displayed directly in the console.
#    c. After training, it finds the path to the newly generated `best_patch.png`.
#    d. It calls the evaluation script (`evaluate_patch.py`) as a subprocess,
#       capturing its output to parse the final success rate.
#    e. It stores the parameters and the resulting success rate.
# 3. After all tests are complete, it prints a summary table, ranked by success rate,
#    to easily identify the best-performing parameter set.
#
# =================================================================================================

import json
import os
import subprocess
import itertools
import time
import re
from datetime import timedelta
from operator import itemgetter
import gc
import torch

from rich.console import Console
from rich.table import Table

# --- Configuration for the Parameter Search ---
CONFIG_FILE = 'config.json'
TRAIN_SCRIPT = 'train_patch.py'
EVAL_SCRIPT = 'evaluate_patch.py'
# Specify which training mode in the config file to modify and test.
TRAINING_MODE_TO_TEST = 'normal' 
# Number of epochs for each test run. Lower than a full run to save time.
EPOCHS_PER_RUN = 80 

# --- Define the search space for the noise parameters ---
# This grid defines all combinations of parameters that will be tested.
# The values below are a recommended starting point that covers the key ranges
# from the README.md without taking an excessive amount of time to run.
#
# For a more exhaustive search, you can add more values to these lists.
# For example, to test more `octaves`, you could change it to:
#   'octaves': [4, 6, 8, 10, 12]
#
# WARNING: Adding more values will significantly increase the total runtime.
# The current configuration results in 36 test runs (2*2*3*3).
PARAMETER_GRID = {
    'persistence': [0.5, 0.6],
    'lacunarity': [1.8, 2.0],
    'scale': [25.0, 50.0, 100.0],
    'octaves': [6, 8, 10]
}

class AutomationController:
    """
    Manages the automated testing workflow, orchestrating the training and
    evaluation scripts for each parameter combination.
    """

    def __init__(self, param_grid):
        self.console = Console()
        self.param_grid = param_grid
        self.all_combinations = list(itertools.product(*self.param_grid.values()))
        
        # This list can be pre-populated with results from previous sessions
        # to avoid re-running tests that have already completed successfully.
        self.results = []
        
        # A set is used for efficient checking of already tested parameter combinations.
        self.tested_params_set = {tuple(sorted(d['params'].items())) for d in self.results}

    def _update_config_file(self, params):
        """
        Dynamically reads, modifies, and writes the config.json file for the current test run.
        """
        self.console.log(f"📝 Updating '{CONFIG_FILE}' with new parameters...")
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)

            # Update the noise parameters and epoch count for the specified training mode.
            config['training_modes'][TRAINING_MODE_TO_TEST]['noise_parameters'] = params
            config['training_modes'][TRAINING_MODE_TO_TEST]['hyperparameters']['max_epochs'] = EPOCHS_PER_RUN
            
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=4)
            self.console.log(f"✅ Config file updated successfully (epochs: {EPOCHS_PER_RUN}).")
            return True
        except (IOError, KeyError) as e:
            self.console.print(f"💥 [bold red]Error updating config file: {e}[/bold red]")
            return False

    def _run_interactive_command(self, command):
        """
        Executes a shell command and allows its UI (e.g., the Rich progress bars
        from the training script) to be displayed directly in the current console.
        """
        try:
            # Note: We use the base train_patch.py script here.
            subprocess.run(command, check=True, text=True)
            return True
        except subprocess.CalledProcessError as e:
            self.console.print(f"\n💥 [bold red]Interactive subprocess '{' '.join(command)}' failed with return code {e.returncode}.[/bold red]")
            return False
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Training subprocess interrupted by user.[/yellow]")
            raise
        except Exception as e:
            self.console.print(f"\n💥 [bold red]An error occurred while running interactive command: {e}[/bold red]")
            return False

    def _run_capture_command(self, command):
        """
        Executes a shell command non-interactively to capture its standard output,
        which is then used for parsing results (e.g., from the evaluation script).
        """
        try:
            # Set environment variables to prevent Rich from trying to render complex UI
            child_env = os.environ.copy()
            child_env['NO_COLOR'] = '1'
            child_env['TERM'] = 'dumb'

            result = subprocess.run(
                command, capture_output=True, text=True, encoding='utf-8',
                env=child_env, check=False
            )

            if result.returncode != 0:
                self.console.print(f"\n💥 [bold red]Capture subprocess '{' '.join(command)}' failed with return code {result.returncode}.[/bold red]")
                self.console.print(result.stdout)
                self.console.print(result.stderr)
                return None
            
            return result.stdout
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Evaluation subprocess interrupted by user.[/yellow]")
            raise
        except Exception as e:
            self.console.print(f"\n💥 [bold red]An error occurred while running capture command: {e}[/bold red]")
            return None

    def _find_latest_run_dir(self):
        """Finds the most recently created directory in the 'runs/' folder."""
        runs_dir = 'runs'
        if not os.path.isdir(runs_dir): return None
        all_subdirs = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
        if not all_subdirs: return None
        return max(all_subdirs, key=os.path.getmtime)

    def _parse_evaluation_output(self, output):
        """Extracts the total success rate from the evaluation script's output using regex."""
        if not output: return 0.0
        match = re.search(r"Total Success Rate.*?(\d+\.\d+)%", output)
        if match: return float(match.group(1))
        self.console.log("[yellow]Could not parse success rate from evaluation output.[/yellow]")
        return 0.0

    def _print_final_results(self):
        """Prints a summary table of all results, sorted by performance."""
        self.console.rule("[bold magenta]Automation Complete: Final Results[/bold magenta]", style="magenta")
        
        if not self.results:
            self.console.print("[yellow]No results were recorded.[/yellow]")
            return

        table = Table(title="[bold blue]Final Test Results[/bold blue]", show_header=True, header_style="bold magenta")
        table.add_column("Rank", style="dim", width=5)
        for key in self.param_grid.keys():
            table.add_column(key.capitalize(), justify="center")
        table.add_column("Success Rate (%)", justify="right", style="green")

        sorted_results = sorted(self.results, key=itemgetter('success_rate'), reverse=True)

        for i, result in enumerate(sorted_results):
            row_data = [str(i + 1)]
            params_dict = result['params']
            for key in self.param_grid.keys():
                 row_data.append(str(params_dict.get(key, 'N/A')))
            row_data.append(f"{result['success_rate']:.2f}%")
            
            style = ""
            if i == 0: style = "bold green"
            elif i == 1: style = "yellow"
            elif i == 2: style = "cyan"
            table.add_row(*row_data, style=style)
            
        self.console.print(table)
        self.console.print("\n🏆 [bold green]Testing finished! The table above shows the best performing parameters.[/bold green]")

    def run_tests(self):
        """Main loop to execute all test combinations."""
        
        # Filter out combinations that have already been tested to allow resuming.
        combinations_to_run = [
            combo for combo in self.all_combinations 
            if tuple(sorted(dict(zip(self.param_grid.keys(), combo)).items())) not in self.tested_params_set
        ]
        
        total_runs = len(combinations_to_run)
        
        # Estimate total runtime for user convenience
        time_per_run_seconds = (EPOCHS_PER_RUN * 36) + 60 # Rough estimate
        total_estimated_seconds = total_runs * time_per_run_seconds
        estimated_duration = str(timedelta(seconds=total_estimated_seconds))
        
        self.console.rule("[bold magenta]Starting Adversarial Patch Parameter Automation[/bold magenta]")
        self.console.print(f"Found [bold cyan]{total_runs}[/bold cyan] new parameter combinations to test.")
        self.console.print(f"Skipping [bold yellow]{len(self.tested_params_set)}[/bold yellow] previously completed runs.")
        self.console.print(f"Training will run for [bold cyan]{EPOCHS_PER_RUN}[/bold cyan] epochs per test.")
        self.console.print(f"Batch size will be determined by the training script's autotuner.")
        self.console.print(f"Estimated time for remaining runs: [bold yellow]{estimated_duration}[/bold yellow]")
        
        for i, combo in enumerate(combinations_to_run):
            current_params = dict(zip(self.param_grid.keys(), combo))
            
            self.console.rule(f"[bold]Starting Run {i + 1} of {total_runs}[/bold]", style="blue")
            self.console.print(f"Parameters: {current_params}")
            
            if not self._update_config_file(current_params):
                self.console.print("[bold red]Skipping run due to config update failure.[/bold red]")
                continue

            self.console.print("\n--- [cyan]Initiating Training (UI will take over)[/cyan] ---")
            train_command = ['python3', TRAIN_SCRIPT, '--training_mode', TRAINING_MODE_TO_TEST, '--no-patience', '--patches', '1']
            if not self._run_interactive_command(train_command):
                self.console.print("[bold red]Training failed. Halting automation.[/bold red]")
                break
            self.console.print("\n--- [green]Training Complete[/green] ---\n")

            latest_run_dir = self._find_latest_run_dir()
            if not latest_run_dir:
                self.console.print("[bold red]Could not find training output directory. Skipping evaluation.[/bold red]")
                continue
            
            patch_path = os.path.join(latest_run_dir, 'best_patch.png')
            if not os.path.exists(patch_path):
                self.console.print(f"[bold red]Could not find 'best_patch.png' in '{latest_run_dir}'.[/bold red]")
                continue
            
            self.console.log(f"✅ Found patch for evaluation: [green]{patch_path}[/green]")

            self.console.print("\n--- [cyan]Initiating Evaluation (running in background)[/cyan] ---")
            with open(CONFIG_FILE, 'r') as f: config = json.load(f)
            model_to_target = config['models_to_target'][0]

            eval_command = ['python3', EVAL_SCRIPT, '--model', model_to_target, '--patch', patch_path]
            eval_output = self._run_capture_command(eval_command)
            if eval_output is None:
                self.console.print("[bold red]Evaluation failed. Halting automation.[/bold red]")
                break
            self.console.print("--- [green]Evaluation Complete[/green] ---\n")

            success_rate = self._parse_evaluation_output(eval_output)
            self.console.log(f"📈 Run {i + 1} Complete. Success Rate: [bold green]{success_rate:.2f}%[/bold green]")
            self.results.append({'params': current_params, 'success_rate': success_rate})
            
            # --- Memory Flushing and Delay ---
            # A short delay can help ensure GPU memory is fully released between runs.
            self.console.log("🧹 Flushing memory and pausing for 15 seconds...")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(15)
            self.console.log("✅ Resuming...")
        
        self._print_final_results()

if __name__ == '__main__':
    controller = AutomationController(PARAMETER_GRID)
    try:
        controller.run_tests()
    except KeyboardInterrupt:
        print("\n\n[bold yellow]User interrupted the process. Printing results gathered so far...[/bold yellow]")
        controller._print_final_results()
    except Exception as e:
        controller.console.print(f"\n\n[bold red]An unexpected error occurred in the main controller: {e}[/bold red]")
        controller.console.print_exception(show_locals=True)
