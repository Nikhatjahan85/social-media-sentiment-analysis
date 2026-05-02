import subprocess
import os
import sys

def run_script(script_path):
    print(f"\n{'='*50}\nExecuting {script_path}...\n{'='*50}")
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error executing {script_path}:")
        print(result.stderr)
        sys.exit(1)
    else:
        print(result.stdout)
        print(f"Successfully executed {script_path}")

def main():
    print("Starting Social Media Sentiment Analysis Pipeline...\n")
    
    # Define execution order
    scripts = [
        "data/generate_dataset.py",
        "src/preprocess.py",
        "src/train_model.py",
        "src/predict.py"
    ]
    
    for script in scripts:
        if os.path.exists(script):
            run_script(script)
        else:
            print(f"Error: Could not find script {script}")
            sys.exit(1)
            
    print("\n" + "="*50)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*50)
    print("\nTo launch the dashboard, run the following command:")
    print("streamlit run app/dashboard.py\n")

if __name__ == "__main__":
    main()