from setuptools import setup, find_packages

setup(
    name='randomized-based-feedforward-neural-network',  # Cambia esto por el nombre de tu paquete
    version='1.0.1',  # Cambia esto por la versión de tu paquete
    packages=find_packages(where='src'),  # Busca paquetes en el directorio src
    package_dir={'': 'src'},  # Indica que los paquetes están en el directorio src
    install_requires=[
        # Aquí puedes agregar las dependencias de tu paquete
        'torch==2.5.1',
        'torchvision==0.20.1',
        'numpy==2.1.2',
        'scikit-learn==1.5.2',
    ],
    author='Emilio Rodrigo Carreira Villalta',
    author_email='ercarreira2@gmail.com',
    description=f"Package for feedforward neural network with randomized-based algorithms",  
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/rorro6787/randomized-based-feedforward-neural-network',  # URL de tu repositorio
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Cambia esto si usas otra licencia
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
