from distutils.core import setup

setup(
  name='quntile regression',
  url='https://github.com/aalling93/',
  author='Kristian Soerensen',
  author_email='kaaso@space.dtu.dk',
  packages=['namees',],
  install_requires=['numpy','scikit-learn','pandas','tensorflow',' tensorflow-probability','matplotlib'],
  version='0.1',
  license='MIT',
  description='',
  long_description=open('README.md').read()
)