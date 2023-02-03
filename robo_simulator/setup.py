from setuptools import setup

package_name = 'robo_simulator'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='muyejia1202',
    maintainer_email='muyejia2023@u.northwestern.edu',
    description='simple robot soccer simulator',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'robo_sim = robo_simulator.simulator:main',
            'soccer_field = robo_simulator.field:main',
            'rs_env = robo_simulator.rs_env:main',
            'eval_rs = robo_simulator.evaluate:main',
        ],
    },
)
