# Copyright (c) 2025 Synria Robotics Co., Ltd.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Author: Synria Robotics Team
# Website: https://synriarobotics.ai

import time


def precise_sleep(seconds: float, spin_threshold: float = 0.002, sleep_margin: float = 0.001):
    """
    Wait for `seconds` with better precision than time.sleep alone at the expense of more CPU usage.

    Parameters:
      - seconds: duration to wait
      - spin_threshold: if remaining <= spin_threshold -> spin; otherwise sleep (seconds). Default 2ms
      - sleep_margin: when sleeping leave this much time before deadline to avoid oversleep. Default 1ms

    Note:
        The default parameters are chosen to prioritize timing accuracy over CPU usage for high-frequency
        use cases like 200 Hz (5ms intervals). For lower frequencies, you may want to increase
        spin_threshold (e.g., 10ms for 30 FPS) for better CPU efficiency.
    """
    if seconds <= 0:
        return

    end_time = time.perf_counter() + seconds
    while True:
        remaining = end_time - time.perf_counter()
        if remaining <= 0:
            break
        if remaining > spin_threshold:
            time.sleep(max(remaining - sleep_margin, 0))
        else:
            pass
