import os
import importlib

# Get all Python files in the current directory
module_files = [f[:-3] for f in os.listdir(os.path.dirname(__file__)) 
                if f.endswith('.py') and f != '__init__.py']

# Import all modules
for module in module_files:
    importlib.import_module(f'.{module}', package=__package__)

# Export all modules
__all__ = module_files