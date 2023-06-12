from distutils.core import setup, Extension

module = Extension('GNMC', sources=['gnmc.cpp'])

setup(name="GNMC",
      version="0.01",
      include_dirs=["/Users/maxwuerfek/code/diss/diss-env/include/python3.8",
                    "/Users/maxwuerfek/code/diss/diss-env/lib/python3.8/site-packages/numpy/core/include"
                    ],
      description="Example module", ext_modules=[module]
      )
