"""
Advanced Dynamic Teaching System

Based on β-VAE + Contrastive Learning
Supports task-conditional priors and domain disentanglement
"""

__version__ = "1.0.0"
__author__ = "Advanced Dynamic Teaching Team"
__description__ = "Advanced Dynamic Teaching System based on β-VAE + Contrastive Learning"

# Project information
PROJECT_NAME = "AdvancedDynamicTeaching"
SYSTEM_TYPE = "Advanced Student VAE"

# Version information
VERSION_INFO = {
    'major': 1,
    'minor': 0,
    'patch': 0,
    'release': 'stable'
}

def get_version_string():
    """Get version string"""
    return f"{VERSION_INFO['major']}.{VERSION_INFO['minor']}.{VERSION_INFO['patch']}-{VERSION_INFO['release']}"

def print_system_info():
    """Print system information"""
    print(" Advanced Dynamic Teaching System")
    print(f"   Version: {get_version_string()}")
    print(f"   System type: {SYSTEM_TYPE}")
    print(f"   Key features: β-VAE + Task-conditional prior + Contrastive learning")
    print("=" * 50) 