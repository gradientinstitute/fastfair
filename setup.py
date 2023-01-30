#!/usr/bin/env python
# Copyright 2023 Gradient Institute
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Setup environment for the fast fair regression package."""

from setuptools import setup

exec(open("fastfair/__version__.py").read())
readme = open("README.md").read()

setup(
    name="fastfair",
    version=__version__,
    description="Fast fair regression models",
    long_description=readme,
    author="Gradient Institute",
    author_email="info@gradientinstitute.org",
    url="https://github.com/gradientinstitute/fastfair",
    packages=["fastfair"],
    package_dir={"fastfair": "fastfair"},
    include_package_data=True
)