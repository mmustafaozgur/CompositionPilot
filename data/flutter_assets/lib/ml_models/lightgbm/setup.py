import subprocess
import sys
import os
import venv

def check_python_version():
    """Check if Python version is 3.7 or higher."""
    version = sys.version_info
    print(f"Current Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("Error: Python 3.7 or higher is required")
        return False
    return True

def get_venv_path():
    """Get the path to the virtual environment."""
    return os.path.join(os.path.dirname(__file__), 'venv')

def get_venv_python():
    """Get the path to the virtual environment's Python executable."""
    if sys.platform == 'win32':
        return os.path.join(get_venv_path(), 'Scripts', 'python.exe')
    return os.path.join(get_venv_path(), 'bin', 'python')

def get_venv_pip():
    """Get the path to the virtual environment's pip executable."""
    if sys.platform == 'win32':
        return os.path.join(get_venv_path(), 'Scripts', 'pip.exe')
    return os.path.join(get_venv_path(), 'bin', 'pip')

def create_venv():
    """Create a virtual environment if it doesn't exist."""
    venv_path = get_venv_path()
    if not os.path.exists(venv_path):
        print("Creating virtual environment...")
        venv.create(venv_path, with_pip=True)
        print("Virtual environment created successfully")
    else:
        print("Virtual environment already exists")

def install_package(package):
    """Install a package using pip in the virtual environment."""
    pip_path = get_venv_pip()
    try:
        print(f"Attempting to install {package}...")
        subprocess.check_call([pip_path, 'install', package])
        print(f"Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing {package}: {e}")
        return False

def main():
    print("Starting LightGBM setup process...")
    print(f"Python executable path: {sys.executable}")
    print(f"Current working directory: {os.getcwd()}")
    
    if not check_python_version():
        sys.exit(1)

    # Create virtual environment
    create_venv()

    required_packages = [
        'flask',
        'flask-cors',
        'lightgbm',
        'pandas',
        'numpy',
        'scikit-learn',
        'joblib',
    ]

    print("\nChecking and installing required packages...")
    missing_packages = []
    pip_path = get_venv_pip()

    for package in required_packages:
        try:
            subprocess.check_call([pip_path, 'show', package], 
                                stdout=subprocess.DEVNULL, 
                                stderr=subprocess.DEVNULL)
            print(f"Package found: {package}")
        except subprocess.CalledProcessError:
            missing_packages.append(package)
            print(f"Missing package: {package}")

    if missing_packages:
        print("\nInstalling required packages...")
        for package in missing_packages:
            if not install_package(package):
                print(f"\nFailed to install required packages. Please install them manually using:")
                print(f"{pip_path} install {' '.join(missing_packages)}")
                sys.exit(1)
        print("\nAll required packages installed successfully!")
    else:
        print("\nAll required packages are already installed!")

    # Verify that the model files exist
    model_files = [
        'lightgbm_delta_e_model.txt',
        'lightgbm_scaler.joblib',
        'lightgbm_model_features.joblib'
    ]
    
    print("\nChecking for required model files...")
    missing_files = []
    for file in model_files:
        file_path = os.path.join(os.path.dirname(__file__), file)
        if os.path.exists(file_path):
            print(f"Found: {file}")
        else:
            missing_files.append(file)
            print(f"Missing: {file}")
    
    if missing_files:
        print(f"\nWarning: Missing model files: {missing_files}")
        print("Please ensure these files are present before running the service.")
    else:
        print("\nAll required model files are present!")

if __name__ == "__main__":
    main() 