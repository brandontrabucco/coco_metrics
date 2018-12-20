"""Setup script for detailed_captioning."""

from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = ['numpy', 'matplotlib', 'scikit-image']

setup(
    name='coco_metrics',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    packages=[p for p in find_packages() if p.startswith('coco_metrics')],
    description='COCO Caption Eval Utility',
)