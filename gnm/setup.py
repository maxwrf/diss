from distutils.core import setup, Extension

module = Extension('GNM', sources=['bridge.cpp',
                                   'gnm.cpp'])

setup(name="GNM",
      version="0.1",
      include_dirs=["/Users/maxwuerfek/code/diss/diss-env/include/python3.8",
                    "/Users/maxwuerfek/code/diss/diss-env/lib/python3.8/site-packages/numpy/core/include"
                    ],
      description="Example module", ext_modules=[module]
      )
