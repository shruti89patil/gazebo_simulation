from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'mobile_robot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # ament index
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        # package.xml
        ('share/' + package_name, ['package.xml']),
        # launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
        # model files
        (os.path.join('share', package_name, 'model'), glob('model/*')),
        # parameter files
        (os.path.join('share', package_name, 'parameters'), glob('parameters/*')),

        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
    ],
    install_requires=['setuptools'], 
    zip_safe=True,
    maintainer='shruti',
    maintainer_email='shruti@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'publisher = mobile_robot.test:main',   # also fixed typo here
            'skid_teleop = mobile_robot.skid_teleop:main',
            'controller_final = mobile_robot.controller_final:main',
        ],
    },
)
