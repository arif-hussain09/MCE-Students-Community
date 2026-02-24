import os
import subprocess
import sys

def run_test(test_file, description):
    """Run a test file and report results"""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✓ {description} - PASSED")
            if result.stdout:
                print("Output:", result.stdout[-500:])  # Last 500 chars
        else:
            print(f"✗ {description} - FAILED")
            if result.stderr:
                print("Error:", result.stderr[-500:])
                
    except subprocess.TimeoutExpired:
        print(f"✗ {description} - TIMEOUT")
    except Exception as e:
        print(f"✗ {description} - ERROR: {e}")

def main():
    print("Running comprehensive test suite...")
    
    tests = [
        ("test_installation.py", "Installation Test"),
        ("test_basic.py", "Basic Functionality Test"),
        ("test_full_processing.py", "Full Processing Test")
    ]
    
    for test_file, description in tests:
        if os.path.exists(test_file):
            run_test(test_file, description)
        else:
            print(f"✗ {test_file} not found - SKIPPED")
    
    print(f"\n{'='*50}")
    print("Test suite completed!")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()