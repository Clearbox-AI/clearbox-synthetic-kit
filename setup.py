import shutil
import os
import numpy as np
from pathlib import Path
from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

# Read the content of the README.md for the long_description metadata
with open("README.md", "r") as readme:
    long_description = readme.read()

# Parse the requirements.txt file to get a list of dependencies
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# List of files to exclude from Cythonization
EXCLUDE_FILES = [
    "clearbox_engine/dataset/dataset.py",
    "clearbox_engine/VAE/tabular_vae.py",
]

def get_extensions_paths(root_dir, exclude_files):
    """
    Retrieve file paths for compilation.

    Parameters
    ----------
    root_dir : str
        Root directory to start searching for files.
    exclude_files : list of str
        List of file paths to exclude from the result.

    Returns
    -------
    list of str or Extension
        A list containing file paths and/or Extension objects.

    """
    paths = []

    # Walk the directory to find .py and .pyx files
    for root, _, files in os.walk(root_dir):
        for filename in files:
            if (
                os.path.splitext(filename)[1] != ".py"
                and os.path.splitext(filename)[1] != ".pyx"
            ):
                continue

            if os.path.splitext(filename)[1] == ".pyx":
                file_path = Extension(
                    root.replace("/", "."),
                    [os.path.join(root, filename)],
                    include_dirs=[np.get_include()],
                )
            else:
                file_path = os.path.join(root, filename)

            if file_path in exclude_files:
                continue

            paths.append(file_path)

    return paths

class CustomBuild(build_ext):
    """
    Custom build class that inherits from Cython's build_ext.

    This class is created to override the default build behavior.
    Specifically, it ensures certain non-Cython files are copied
    over to the build output directory after the Cythonization process.
    """

    def run(self):
        """Override the run method to copy specific files after build."""
        # Run the original run method
        build_ext.run(self)

        build_dir = Path(self.build_lib)
        root_dir = Path(__file__).parent
        target_dir = build_dir if not self.inplace else root_dir

        # List of files to copy after the build process
        files_to_copy = [
            Path("clearbox_engine/dataset") / "__init__.py",
            Path("clearbox_engine/dataset") / "dataset.py",
            Path("clearbox_engine/preprocessor") / "__init__.py",
            Path("clearbox_engine/transformers") / "__init__.py",
            Path("clearbox_engine/VAE") / "__init__.py",
            Path("clearbox_engine/VAE") / "tabular_vae.py",
            Path("clearbox_engine/VAE") / "timeseries_vae.py",
            Path("clearbox_engine/engine") / "__init__.py",
            Path("clearbox_engine/autoconfig") / "__init__.py",
            Path("clearbox_engine/synthesizer") / "__init__.py",
            Path("clearbox_engine/anomalies") / "__init__.py",
            Path("clearbox_engine/metrics") / "__init__.py",
            Path("clearbox_engine/metrics/privacy") / "__init__.py",
            Path("clearbox_engine/metrics/privacy") / "gower_matrix_c.pyx"
        ]

        for file in files_to_copy:
            self.copy_file(file, root_dir, target_dir)

    def copy_file(self, path, source_dir, destination_dir):
        """
        Utility method to copy files from source to destination.

        Parameters
        ----------
        path : Path
            Path of the file to be copied.
        source_dir : Path
            Directory where the source file resides.
        destination_dir : Path
            Directory where the file should be copied to.

        """
        if not (source_dir / path).exists():
            return

        shutil.copyfile(str(source_dir / path), str(destination_dir / path))


# Main setup configuration
setup(
    # Metadata about the package
    name="clearbox-engine",
    version="1.0.3",
    author="Clearbox AI",
    author_email="info@clearbox.ai",
    description="The engine of all Clearbox AI tools, which provides an easy to use Dataset class, a dynamic data preprocessor and an advanced VAE.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Clearbox-AI/engine",
    install_requires=requirements,
    python_requires=">=3.7.0",
    
    # Cython modules compilation
    ext_modules=cythonize(
        get_extensions_paths("clearbox_engine", EXCLUDE_FILES),
        build_dir="build",
        compiler_directives=dict(language_level=3, always_allow_keywords=True),
    ),
    
    # Override the build command with our custom class
    cmdclass=dict(build_ext=CustomBuild),

    # List of packages included in the distribution (empty in this case)
    packages=[],
)