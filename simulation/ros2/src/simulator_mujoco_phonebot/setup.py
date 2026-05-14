from setuptools import find_packages, setup

package_name = 'simulator_mujoco_phonebot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hrc',
    maintainer_email='houruochen@g.ucla.edu',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Torque-aware realtime node (torque controller in code).
            'phonebot_realtime_keyboard_torque = simulator_mujoco_phonebot.realtime_keyboard_torque_awared_node:main',
            # Position-control realtime node (direct position targets to ctrl).
            'phonebot_realtime_keyboard_position = simulator_mujoco_phonebot.realtime_keyboard_position_node:main',
            # Backward-compatible alias (kept): points to torque-aware node.
            'phonebot_realtime_keyboard = simulator_mujoco_phonebot.realtime_keyboard_torque_awared_node:main',

            # Cross-repo realtime policy runners (MuJoCo CPU).
            'realtime_simulation_booster_gym_policy = simulator_mujoco_phonebot.realtime_simulation_booster_gym_policy:main',
            'realtime_simulation_hermes_hi_policy = simulator_mujoco_phonebot.realtime_simulation_hermes_hi_policy:main',
            'realtime_simulation_playground_t1_policy = simulator_mujoco_phonebot.realtime_simulation_playground_t1_policy:main',
        ],
    },
)
