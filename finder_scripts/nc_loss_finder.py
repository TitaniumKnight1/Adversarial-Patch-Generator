import os
import subprocess
import sys
import argparse
import re
import shutil
from datetime import datetime

# --- Rich CLI Components for a clean UI ---
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# --- Configuration ---
# UPDATE THIS PATH to your new training script with the NC loss term
TRAIN_SCRIPT_PATH = 'train_patch_v2.5.py' 
EVAL_SCRIPT_PATH = 'evaluate_patch_v2.2.py'

# --- Initialize Rich Console ---
console = Console()

def check_script_paths():
    """Verify that the required training and evaluation scripts exist."""
    if not os.path.exists(TRAIN_SCRIPT_PATH):
        console.print(f"❌ [bold red]Error: Training script not found at '{TRAIN_SCRIPT_PATH}'[/bold red]")
        console.print(f"   Please update the 'TRAIN_SCRIPT_PATH' variable in this script.")
        sys.exit(1)
    if not os.path.exists(EVAL_SCRIPT_PATH):
        console.print(f"❌ [bold red]Error: Evaluation script not found at '{EVAL_SCRIPT_PATH}'[/bold red]")
        sys.exit(1)
    console.print("✅ [green]Located training and evaluation scripts.[/green]")

def run_command(command, description, progress, task):
    """Runs a command as a subprocess and handles its lifecycle."""
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
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
    """Main function to orchestrate the training and evaluation for different NC weights."""
    check_script_paths()
    
    weights_to_test = args.nc_weights
    trained_patches = {} # Dict to store {weight: path}

    console.print(Panel(f"🚀 [bold magenta]Starting NC Loss Weight Optimization[/bold magenta]\n"
                        f"   - [b]NC Weights to Test[/b]: [cyan]{weights_to_test}[/cyan]\n"
                        f"   - [b]Training Epochs per Patch[/b]: [cyan]{args.epochs}[/cyan]\n"
                        f"   - [b]Constant TV Weight[/b]: [cyan]{args.tv_weight}[/cyan]\n"
                        f"   - [b]Evaluation Scale[/b]: [cyan]{args.eval_scale}[/cyan]",
                        title="[yellow]NC Weight Optimization Configuration[/yellow]", border_style="yellow"))

    # --- Step 1: Train a patch for each NC weight ---
    console.print("\n--- [blue]Phase 1: Training Patches[/blue] ---")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        train_task = progress.add_task("[green]Training...", total=len(weights_to_test))
        
        for weight in weights_to_test:
            progress.print(f"🛠️  Training patch with NC Weight = [bold cyan]{weight:.2e}[/bold cyan]...")
            
            # Run the training process by passing the weight as a command-line argument
            train_command = [
                sys.executable, TRAIN_SCRIPT_PATH, 
                '--max_epochs', str(args.epochs),
                '--nc_weight', str(weight),
                '--tv_weight', str(args.tv_weight),
                '--early_stopping_patience', '10'
            ]
            run_command(train_command, f"Training NC weight {weight}", progress, train_task)

            # Find and save the path to the newly trained patch
            latest_run = find_latest_run_dir()
            if latest_run and os.path.exists(os.path.join(latest_run, 'best_patch.png')):
                new_patch_name = f"best_patch_nc_weight_{weight:.2e}.png"
                destination_path = os.path.join(os.path.dirname(latest_run), new_patch_name)
                shutil.move(os.path.join(latest_run, 'best_patch.png'), destination_path)
                trained_patches[weight] = destination_path
                progress.print(f"✅ Saved patch to [green]'{destination_path}'[/green]")
            else:
                progress.print(f"⚠️ [yellow]Could not find or save the patch for NC weight {weight}. Skipping.[/yellow]")

    # --- Step 2: Evaluate each trained patch ---
    console.print("\n--- [blue]Phase 2: Evaluating Patches[/blue] ---")
    evaluation_results = []
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        eval_task = progress.add_task("[green]Evaluating...", total=len(trained_patches))
        
        for weight, patch_path in trained_patches.items():
            progress.print(f"📊 Evaluating patch from NC Weight = [bold cyan]{weight:.2e}[/bold cyan]...")
            
            eval_command = [
                sys.executable, EVAL_SCRIPT_PATH,
                '--patch_path', patch_path,
                '--patch_scale', str(args.eval_scale)
            ]
            
            result = subprocess.run(eval_command, capture_output=True, text=True)
            asr = parse_asr_from_output(result.stdout)
            evaluation_results.append({'weight': weight, 'asr': asr})
            progress.update(eval_task, advance=1)

    # --- Step 3: Report the final results ---
    console.print("\n" + "="*80)
    console.print("📊 [bold green]NC Weight Optimization Complete: Final Results[/bold green]")
    console.print("="*80)

    results_table = Table(title="NC Loss Weight Performance", show_header=True, header_style="bold magenta")
    results_table.add_column("NC Weight", style="cyan", justify="center")
    results_table.add_column("Attack Success Rate (ASR %)", style="green", justify="center")

    evaluation_results.sort(key=lambda x: x['asr'], reverse=True)

    best_weight = None
    if evaluation_results:
        best_weight = evaluation_results[0]['weight']

    for res_data in evaluation_results:
        is_best = "⭐ " if res_data['weight'] == best_weight else ""
        results_table.add_row(
            f"{is_best}{res_data['weight']:.2e}",
            f"{res_data['asr']:.2f}%"
        )
    
    console.print(results_table)

    if best_weight is not None:
        console.print(Panel(f"The highest Attack Success Rate was achieved with an NC Loss Weight of [bold cyan]{best_weight:.2e}[/bold cyan].",
                            title="[bold blue]Optimal NC Weight Recommendation[/bold blue]", border_style="blue"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Find the optimal Non-Classifiability (NC) loss weight.")
    parser.add_argument('--nc_weights', type=float, nargs='+', default=[0.1, 0.5, 1.0, 2.0, 5.0], help='A list of NC loss weights to test.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train each patch for.')
    parser.add_argument('--tv_weight', type=float, default=1e-5, help='Constant Total Variation weight to use during tests.')
    parser.add_argument('--eval_scale', type=float, default=0.70, help='The patch-to-target area scale to use during evaluation.')
    
    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        console.print(f"\n💥 [bold red]An unexpected error occurred during the optimization process![/bold red]")
        console.print_exception(show_locals=False)
