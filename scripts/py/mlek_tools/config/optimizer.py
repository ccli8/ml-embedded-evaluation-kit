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
"""MLEK's single source of truth for NPU compilation settings."""
import typing
from dataclasses import dataclass, field
from pathlib import Path

# Applied when memory_mode is Shared_Sram and no explicit arena_cache_size is set.
_DEFAULT_SHARED_SRAM_ARENA_SIZE = 2 * 1024 * 1024  # 2 MiB


@dataclass(frozen=True)
class OptimizerConfig:
    """MLEK's NPU optimization intent.

    All fields map directly to Vela compiler options and are applied identically
    to both the TFLM (Vela CLI) and ExecuTorch (``EthosUCompileSpec``) paths,
    ensuring a single source of truth for how models are compiled for the NPU.
    """
    vela_config_file: typing.Optional[Path] = None
    arena_cache_size: int = 0
    cop_format: str = "COP1"
    separate_io_regions: bool = False
    additional_npu_config_names: typing.List[str] = field(default_factory=list)

    def __post_init__(self):
        """
        Validate optimization parameters.

        :raises ValueError: If any option is outside the supported range.
        """
        if self.arena_cache_size < 0:
            raise ValueError("Arena cache size cannot be less than 0")

        if self.cop_format not in ("COP1", "COP2"):
            raise ValueError("COP format should be one of: COP1, COP2")

    def _effective_arena_cache_size(self, memory_mode: str) -> int:
        """Resolve the arena cache size, applying the Shared_Sram default when needed."""
        if self.arena_cache_size:
            return self.arena_cache_size
        if memory_mode == "Shared_Sram":
            return _DEFAULT_SHARED_SRAM_ARENA_SIZE
        return 0

    def to_vela_extra_flags(self, memory_mode: str = "") -> typing.List[str]:
        """Extra Vela flags beyond the standard compile-spec parameters.

        For the ExecuTorch path these go into ``EthosUCompileSpec.extra_flags``.
        ``memory_mode`` is required to resolve the Shared_Sram arena default.
        """
        args = [f"--cop-format={self.cop_format}"]
        effective_arena = self._effective_arena_cache_size(memory_mode)
        if effective_arena:
            args.append(f"--arena-cache-size={effective_arena}")
        if self.separate_io_regions:
            args.append("--separate-io-regions")
        return args

    def to_vela_cli_args(self, memory_mode: str = "") -> typing.List[str]:
        """Full flat Vela CLI arg list for the TFLM path.

        Includes ``--config`` (when :attr:`vela_config_file` is set) in addition
        to the extra flags returned by :meth:`to_vela_extra_flags`.
        """
        args = self.to_vela_extra_flags(memory_mode)
        if self.vela_config_file:
            args += ["--config", str(self.vela_config_file)]
        return args
