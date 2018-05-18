try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'My useful python tools.',
    'author': 'Soren Wacker',
    'url': 'https://github.com/soerendip42',
    'download_url': 'https://github.com/soerendip42/myToolbox',
    'author_email': 'swacker@ucalgary.ca',
    'version': '0.1',
    'install_requires': [''],
    'packages': [''],
    'scripts': [],
    'name': 'myToolbox'
}

setup(**config)
