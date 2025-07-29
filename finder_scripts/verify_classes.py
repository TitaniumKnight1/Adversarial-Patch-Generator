from ultralytics import YOLO
import os
from rich.console import Console
from rich.table import Table

# --- Initialize Rich Console for pretty printing ---
console = Console()

# --- Configuration ---
# IMPORTANT: Set this to the path of your model file.
# This should be one of the models you are targeting, 
# for example, the one from your config.json.
MODEL_PATH = "yolov11n.pt" 
OUTPUT_FILENAME = "class_reference.txt"

def verify_model_classes(model_path):
    """
    Loads a YOLO model, prints its class names to the console,
    and saves them to a text file.
    """
    if not os.path.exists(model_path):
        console.print(f"💥 [bold red]Error: Model file not found at '{model_path}'[/bold red]")
        console.print("Please make sure the MODEL_PATH variable in the script is correct.")
        return

    try:
        # Load the YOLO model
        console.print(f"⏳ [cyan]Loading model from '{model_path}'...[/cyan]")
        model = YOLO(model_path)
        
        # The class names are stored in the 'names' attribute of the model
        class_names = model.names
        
        if not class_names:
            console.print("🤔 [yellow]The model loaded, but no class names were found.[/yellow]")
            return

        console.print(f"✅ [bold green]Successfully loaded model. Found {len(class_names)} classes.[/bold green]")

        # --- Create and save the .txt reference file ---
        with open(OUTPUT_FILENAME, 'w') as f:
            f.write("# YOLO Model Class Reference\n")
            f.write(f"# Generated from model: {os.path.basename(model_path)}\n\n")
            f.write("# Use the numerical IDs below in the 'target_classes' array of your config.json\n")
            f.write("# Example: 'target_classes': [2, 5, 7] will target cars, buses, and trucks.\n\n")
            for class_id, class_name in class_names.items():
                f.write(f"{class_id}: {class_name}\n")
        
        console.print(f"💾 [bold blue]Class reference saved to '{OUTPUT_FILENAME}'[/bold blue]")


        # --- Create and print a table to the console ---
        table = Table(title=f"YOLO Model Class Reference ({os.path.basename(model_path)})")
        table.add_column("Class ID", justify="right", style="cyan", no_wrap=True)
        table.add_column("Class Name", style="magenta")

        # Populate the table
        for class_id, class_name in class_names.items():
            table.add_row(str(class_id), class_name)
            
        console.print(table)

    except Exception as e:
        console.print(f"💥 [bold red]An error occurred while loading the model or reading classes:[/bold red]")
        console.print(e)

if __name__ == "__main__":
    verify_model_classes(MODEL_PATH)
