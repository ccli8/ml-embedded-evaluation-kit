#  SPDX-FileCopyrightText:  Copyright 2025-2026 Arm Limited and/or its
#  affiliates <open-source-office@arm.com>
#  SPDX-License-Identifier: Apache-2.0
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""NPU hardware descriptor dataclasses."""
import itertools
import typing
from dataclasses import dataclass


@dataclass(frozen=True)
class NpuConfig:
    """Hardware descriptor for a single Ethos-U NPU configuration."""
    name_prefix: str
    macs: int
    processor_id: str
    prefix_id: str
    memory_mode: str
    system_config: str

    @property
    def config_name(self) -> str:
        """Vela accelerator-config string, e.g. ``"ethos-u55-128"``."""
        return f"{self.name_prefix}-{self.macs}"

    @property
    def config_id(self) -> str:
        """Short ID used in output file names, e.g. ``"H128"``."""
        return f"{self.prefix_id}{self.macs}"


@dataclass(frozen=True)
class NpuConfigs:
    """Indexed collection of :class:`NpuConfig` objects."""
    configs: typing.Dict[str, typing.Dict[int, NpuConfig]]

    @staticmethod
    def create(*configs: NpuConfig) -> "NpuConfigs":
        """Build a collection from an arbitrary number of :class:`NpuConfig` instances."""
        _configs: typing.Dict[str, typing.Dict[int, NpuConfig]] = {}
        for c in configs:
            if c.name_prefix not in _configs:
                _configs[c.name_prefix] = {}
            _configs[c.name_prefix][c.macs] = c
        return NpuConfigs(configs=_configs)

    def get(self, name_prefix: str, macs: typing.Union[int, str]) -> typing.Optional[NpuConfig]:
        """Return the config matching ``name_prefix`` and ``macs``, or ``None``."""
        configs_for_name = self.configs.get(name_prefix)
        if not configs_for_name:
            return None
        return configs_for_name.get(int(macs))

    def get_by_name(self, name: str) -> typing.Optional[NpuConfig]:
        """Return the config matching a full name such as ``"ethos-u55-128"``, or ``None``."""
        name_prefix, macs = name.rsplit("-", 1)
        return self.get(name_prefix, macs)

    @property
    def names(self) -> typing.List[str]:
        """All config names in this collection, e.g. ``["ethos-u55-128", ...]``."""
        return list(itertools.chain.from_iterable([
            [f"{c.name_prefix}-{c.macs}" for c in config.values()]
            for config in self.configs.values()
        ]))


_u85_macs_to_system_configs = {
    128: "Ethos_U85_SYS_DRAM_Low",
    256: "Ethos_U85_SYS_DRAM_Low",
    512: "Ethos_U85_SYS_DRAM_Mid_512",
    1024: "Ethos_U85_SYS_DRAM_Mid_1024",
    2048: "Ethos_U85_SYS_DRAM_High_2048",
}

#: All supported Ethos-U NPU configurations.
valid_npu_configs = NpuConfigs.create(
    *(
        NpuConfig(
            name_prefix="ethos-u55",
            macs=macs,
            processor_id="U55",
            prefix_id="H",
            memory_mode="Shared_Sram",
            system_config="Ethos_U55_High_End_Embedded",
        ) for macs in (32, 64, 128, 256)
    ),
    *(
        NpuConfig(
            name_prefix="ethos-u65",
            macs=macs,
            processor_id="U65",
            prefix_id="Y",
            memory_mode="Dedicated_Sram",
            system_config="Ethos_U65_High_End",
        ) for macs in (256, 512)
    ),
    *(
        NpuConfig(
            name_prefix="ethos-u85",
            macs=macs,
            processor_id="U85",
            prefix_id="Z",
            memory_mode="Dedicated_Sram",
            system_config=_u85_macs_to_system_configs[macs],
        ) for macs in (128, 256, 512, 1024, 2048)
    ),
)


def get_default_npu_config_from_name(config_name: str) -> NpuConfig:
    """Return the :class:`NpuConfig` for *config_name* from :data:`valid_npu_configs`.

    :param config_name: Ethos-U NPU configuration name, e.g. ``"ethos-u55-128"``.
    :raises ValueError:  If *config_name* is not recognised.
    """
    npu_config = valid_npu_configs.get_by_name(config_name)
    if not npu_config:
        raise ValueError(
            f"Invalid Ethos-U NPU configuration '{config_name}'. "
            f"Select one from {valid_npu_configs.names}."
        )
    return npu_config
