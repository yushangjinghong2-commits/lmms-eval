#!/usr/bin/env python3
"""
UniPic2 Integration Configuration Checker

This script checks if the UniPic2 integration is properly configured.
"""

import os
import sys
from pathlib import Path


def check_unipic2_path():
    """Check if UniPic-2 repository is accessible"""
    print("=" * 60)
    print("1. Checking UniPic-2 Repository Location")
    print("=" * 60)

    # Expected path
    wd = Path(__file__).parent.resolve()
    unipic2_path = os.path.join(str(wd), "..", "UniPic", "UniPic-2")
    unipic2_path = os.path.abspath(unipic2_path)

    print(f"Expected path: {unipic2_path}")

    if os.path.exists(unipic2_path):
        print("✓ UniPic-2 repository found!")

        # Check for key files
        key_files = [
            "unipicv2/pipeline_stable_diffusion_3_kontext.py",
            "unipicv2/transformer_sd3_kontext.py",
            "unipicv2/stable_diffusion_3_conditioner.py",
        ]

        all_found = True
        for file in key_files:
            file_path = os.path.join(unipic2_path, file)
            if os.path.exists(file_path):
                print(f"  ✓ Found: {file}")
            else:
                print(f"  ✗ Missing: {file}")
                all_found = False

        if all_found:
            print("✓ All required UniPic-2 modules found!")
            return True
        else:
            print("✗ Some UniPic-2 modules are missing!")
            return False
    else:
        print(f"✗ UniPic-2 repository not found at {unipic2_path}")
        print("\nPlease ensure UniPic-2 is cloned at:")
        print(f"  {unipic2_path}")
        print("\nOr adjust the path in the integration code.")
        return False


def check_model_registration():
    """Check if models are properly registered"""
    print("\n" + "=" * 60)
    print("2. Checking Model Registration")
    print("=" * 60)

    try:
        # Add lmms-eval to path
        lmms_eval_path = Path(__file__).parent.resolve()
        if str(lmms_eval_path) not in sys.path:
            sys.path.insert(0, str(lmms_eval_path))

        from lmms_eval.models import AVAILABLE_SIMPLE_MODELS

        models_to_check = ["unipic2", "unipic2_visual_cot"]
        all_registered = True

        for model_name in models_to_check:
            if model_name in AVAILABLE_SIMPLE_MODELS:
                print(f"✓ Model '{model_name}' is registered")
            else:
                print(f"✗ Model '{model_name}' is NOT registered")
                all_registered = False

        if all_registered:
            print("✓ All UniPic2 models are properly registered!")
            return True
        else:
            print("✗ Some models are not registered!")
            return False

    except Exception as e:
        print(f"✗ Error checking model registration: {e}")
        return False


def check_model_files():
    """Check if model files exist"""
    print("\n" + "=" * 60)
    print("3. Checking Model Implementation Files")
    print("=" * 60)

    lmms_eval_path = Path(__file__).parent.resolve()
    models_path = os.path.join(lmms_eval_path, "lmms_eval", "models", "simple")

    model_files = ["unipic2.py", "unipic2_visual_cot.py"]

    all_found = True
    for file in model_files:
        file_path = os.path.join(models_path, file)
        if os.path.exists(file_path):
            print(f"✓ Found: {file}")
        else:
            print(f"✗ Missing: {file}")
            all_found = False

    if all_found:
        print("✓ All model files are present!")
        return True
    else:
        print("✗ Some model files are missing!")
        return False


def check_dependencies():
    """Check if required dependencies are installed"""
    print("\n" + "=" * 60)
    print("4. Checking Python Dependencies")
    print("=" * 60)

    required_packages = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("diffusers", "diffusers"),
        ("accelerate", "accelerate"),
        ("pillow", "PIL"),  # Pillow is imported as PIL
        ("tqdm", "tqdm"),
        ("loguru", "loguru"),
    ]

    all_installed = True
    for display_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"✓ {display_name} is installed")
        except ImportError:
            print(f"✗ {display_name} is NOT installed")
            all_installed = False

    if all_installed:
        print("✓ All required dependencies are installed!")
        return True
    else:
        print("✗ Some dependencies are missing!")
        print("\nInstall missing dependencies with:")
        print("  pip install torch transformers diffusers accelerate pillow tqdm loguru")
        return False


def check_task_files():
    """Check if illusionbench task files exist"""
    print("\n" + "=" * 60)
    print("5. Checking illusionbench Task Files")
    print("=" * 60)

    lmms_eval_path = Path(__file__).parent.resolve()
    tasks_path = os.path.join(
        lmms_eval_path, "lmms_eval", "tasks", "illusionbench"
    )

    if not os.path.exists(tasks_path):
        print(f"✗ illusionbench tasks directory not found at {tasks_path}")
        return False

    # Check for key files
    key_files = ["arshia_utils.py", "__init__.py"]

    all_found = True
    for file in key_files:
        file_path = os.path.join(tasks_path, file)
        if os.path.exists(file_path):
            print(f"✓ Found: {file}")
        else:
            print(f"✗ Missing: {file}")
            all_found = False

    # Count task YAML files
    import glob

    yaml_files = glob.glob(os.path.join(tasks_path, "*.yaml"))
    print(f"✓ Found {len(yaml_files)} task YAML files")

    if all_found:
        print("✓ illusionbench task files are present!")
        return True
    else:
        print("✗ Some task files are missing!")
        return False


def main():
    """Run all checks"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║  UniPic2 Integration Configuration Checker              ║")
    print("╚" + "=" * 58 + "╝")
    print("\n")

    checks = [
        check_unipic2_path,
        check_model_registration,
        check_model_files,
        check_dependencies,
        check_task_files,
    ]

    results = []
    for check in checks:
        results.append(check())

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(f"Checks passed: {passed}/{total}")

    if all(results):
        print("\n✓ All checks passed! Integration is properly configured.")
        print("\nYou can now use UniPic2 models with lmms-eval:")
        print("  python -m lmms_eval --model unipic2 ...")
        print("  python -m lmms_eval --model unipic2_visual_cot ...")
        return 0
    else:
        print("\n✗ Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
