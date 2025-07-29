import os
import subprocess
import sys
import argparse
import re
import shutil
from datetime import datetime
import json

# --- Rich CLI Components for a clean UI ---
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# --- Configuration ---
# UPDATE THIS PATH to your new training script that uses config.json
TRAIN_SCRIPT_PATH = 'train_patch_v2.6.py' 
EVAL_SCRIPT_PATH = 'evaluate_patch_v2.3.py'

# --- Initialize Rich Console ---
console = Console()

def check_script_paths():
    """Verify that the required training and evaluation scripts exist."""
    if not os.path.exists(TRAIN_SCRIPT_PATH):
        console.print(f"❌ [bold red]Error: Training script not found at '{TRAIN_SCRIPT_PATH}'[/bold red]")
        sys.exit(1)
    if not os.path.exists(EVAL_SCRIPT_PATH):
        console.print(f"❌ [bold red]Error: Evaluation script not found at '{EVAL_SCRIPT_PATH}'[/bold red]")
        sys.exit(1)
    console.print("✅ [green]Located training and evaluation scripts.[/green]")

def modify_config_for_run(base_config_path, temp_config_path, new_patch_size, max_epochs):
    """
    Loads the base config, modifies patch_size and max_epochs, 
    and saves it to a temporary path.
    """
    try:
        with open(base_config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        console.print(f"❌ [bold red]Error: Base config file not found at '{base_config_path}'[/bold red]")
        sys.exit(1)
    except json.JSONDecodeError:
        console.print(f"❌ [bold red]Error: Could not decode JSON from '{base_config_path}'. Please check its format.[/bold red]")
        sys.exit(1)

    # Update the values for the test run
    config['patch_size'] = new_patch_size
    if 'hyperparameters' not in config:
        config['hyperparameters'] = {}
    config['hyperparameters']['max_epochs'] = max_epochs
    
    with open(temp_config_path, 'w') as f:
        json.dump(config, f, indent=4)

def run_command(command, description, progress, task):
    """Runs a command as a subprocess and handles its lifecycle."""
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, encoding='utf-8')
        
        # You can uncomment the following lines to see the live output from subprocesses
        # for line in iter(process.stdout.readline, ''):
        #     console.print(f"[grey50] > {line.strip()}[/grey50]")
        
        process.wait() # Wait for the subprocess to complete
        progress.update(task, advance=1)
        if process.returncode != 0:
            console.print(f"⚠️ [yellow]Warning: Subprocess for '{description}' exited with code {process.returncode}.[/yellow]")
        
        return process.returncode
    except FileNotFoundError:
        console.print(f"❌ [bold red]Error: Command '{command[0]}' not found. Is Python installed and in your PATH?[/bold red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"❌ [bold red]An error occurred while running command for '{description}': {e}[/bold red]")
        return -1

def find_latest_run_dir():
    """Finds the most recently created directory in the 'runs' folder."""
    runs_dir = 'runs'
    if not os.path.isdir(runs_dir):
        return None
    
    all_subdirs = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
    if not all_subdirs:
        return None
        
    latest_subdir = max(all_subdirs, key=os.path.getmtime)
    return latest_subdir

def parse_asr_from_output(output_str):
    """Parses the Attack Success Rate (ASR) from the evaluation script's output."""
    match = re.search(r"Attack Success Rate \(ASR\).*?(\d+\.\d+)%", output_str)
    if match:
        return float(match.group(1))
    
    # Fallback for table format which might also contain the ASR
    match = re.search(r"⭐?\s*[\d\.]+\s*│\s*(\d+\.\d+)%", output_str)
    if match:
        return float(match.group(1))
        
    console.print("[yellow]Warning: Could not parse ASR from evaluation output.[/yellow]")
    return 0.0

def main(args):
    """Main function to orchestrate the training and evaluation for different resolutions."""
    check_script_paths()
    
    resolutions_to_test = args.resolutions
    temp_config_path = "temp_config_for_density_test.json"
    trained_patches = {} # Dict to store {resolution: path}

    console.print(Panel(f"🚀 [bold magenta]Starting Patch Pixel Density Optimization[/bold magenta]\n"
                        f"   - [b]Base Config[/b]: [cyan]{args.base_config}[/cyan]\n"
                        f"   - [b]Resolutions to Test[/b]: [cyan]{resolutions_to_test}[/cyan]\n"
                        f"   - [b]Training Epochs per Patch[/b]: [cyan]{args.epochs}[/cyan]\n"
                        f"   - [b]Evaluation Scale[/b]: [cyan]{args.eval_scale}[/cyan]",
                        title="[yellow]Pixel Density Optimization Configuration[/yellow]", border_style="yellow"))

    # Create a dedicated subdirectory for these test runs
    test_run_dir = os.path.join('runs', 'density_tests')
    os.makedirs(test_run_dir, exist_ok=True)
    console.print(f"✅ [green]Test patches will be saved in '{test_run_dir}'[/green]")

    # --- Step 1: Train a patch for each resolution ---
    console.print("\n--- [blue]Phase 1: Training Patches[/blue] ---")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        train_task = progress.add_task("[green]Training...", total=len(resolutions_to_test))
        
        for res in resolutions_to_test:
            progress.print(f"🛠️  Training patch at [bold cyan]{res}x{res}[/bold cyan] resolution...")
            
            # Create a temporary, modified config file for the current resolution
            modify_config_for_run(args.base_config, temp_config_path, res, args.epochs)

            train_command = [
                sys.executable, TRAIN_SCRIPT_PATH, 
                '--config', temp_config_path,
            ]
            run_command(train_command, f"Training {res}x{res}", progress, train_task)

            # Find and save the path to the newly trained patch
            latest_run = find_latest_run_dir()
            if latest_run and os.path.exists(os.path.join(latest_run, 'best_patch.png')):
                new_patch_name = f"best_patch_density_{res}x{res}.png"
                # Save to the dedicated test directory
                destination_path = os.path.join(test_run_dir, new_patch_name)
                shutil.move(os.path.join(latest_run, 'best_patch.png'), destination_path)
                trained_patches[res] = destination_path
                progress.print(f"✅ Saved patch to [green]'{destination_path}'[/green]")
            else:
                progress.print(f"⚠️ [yellow]Could not find or save the patch for {res}x{res}. Skipping.[/yellow]")
    
    # --- Step 2: Evaluate each trained patch ---
    console.print("\n--- [blue]Phase 2: Evaluating Patches[/blue] ---")
    evaluation_results = []
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        eval_task = progress.add_task("[green]Evaluating...", total=len(trained_patches))
        
        for res, patch_path in trained_patches.items():
            progress.print(f"📊 Evaluating [bold cyan]{res}x{res}[/bold cyan] patch...")
            
            eval_command = [
                sys.executable, EVAL_SCRIPT_PATH,
                '--patch_path', patch_path,
                '--patch_scale', str(args.eval_scale)
            ]
            
            result = subprocess.run(eval_command, capture_output=True, text=True)
            asr = parse_asr_from_output(result.stdout)
            evaluation_results.append({'resolution': res, 'asr': asr})
            progress.update(eval_task, advance=1)

    # --- Step 3: Report the final results ---
    console.print("\n" + "="*80)
    console.print("📊 [bold green]Pixel Density Optimization Complete: Final Results[/bold green]")
    console.print("="*80)

    results_table = Table(title="Patch Pixel Density Performance", show_header=True, header_style="bold magenta")
    results_table.add_column("Resolution", style="cyan", justify="center")
    results_table.add_column("Attack Success Rate (ASR %)", style="green", justify="center")

    evaluation_results.sort(key=lambda x: x['asr'], reverse=True)

    best_resolution = None
    if evaluation_results:
        best_resolution = evaluation_results[0]['resolution']

    for res_data in evaluation_results:
        is_best = "⭐ " if res_data['resolution'] == best_resolution else ""
        results_table.add_row(
            f"{is_best}{res_data['resolution']}x{res_data['resolution']}",
            f"{res_data['asr']:.2f}%"
        )
    
    console.print(results_table)

    if best_resolution is not None:
        console.print(Panel(f"The highest Attack Success Rate was achieved with a patch trained at [bold cyan]{best_resolution}x{best_resolution}[/bold cyan] pixel density.",
                            title="[bold blue]Optimal Pixel Density Recommendation[/bold blue]", border_style="blue"))

    # Clean up the temporary config file
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Find the optimal adversarial patch pixel density by training and evaluating patches at different resolutions.")
    parser.add_argument('--base_config', type=str, default='config.json', help='Path to the base JSON configuration file.')
    parser.add_argument('--resolutions', type=int, nargs='+', default=[100, 150, 200, 250, 300], help='A list of patch resolutions (e.g., 100 200 300) to test.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train each patch for. This will override the value in the config file for the test runs.')
    parser.add_argument('--eval_scale', type=float, default=0.70, help='The patch-to-target area scale to use during evaluation.')
    
    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        console.print(f"\n💥 [bold red]An unexpected error occurred during the optimization process![/bold red]")
        console.print_exception(show_locals=False)
