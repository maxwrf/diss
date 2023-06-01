from distutils.core import setup, Extension

module = Extension('Example', sources=['example.C'])

setup(name="PackageName", version="0.01",
      description="Example module", ext_modules=[module])
