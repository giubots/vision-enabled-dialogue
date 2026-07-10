from setuptools import find_packages, setup

package_name = "vision_enabled_dialogue"

setup(
    name=package_name,
    version="0.3.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Giulio Antonio Abbo",
    maintainer_email="giubots@protonmail.com",
    description="A ROS 2 package for an LLM-based dialogue system that incorporates visual inputs to enhance contextual understanding.",
    license="Apache-2.0",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": ["main = vision_enabled_dialogue.main_ros:main"],
    },
)
