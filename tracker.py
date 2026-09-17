"""Solar geometry and panel irradiance model for the dual-axis tracker.

Modeling assumptions:
1. Angles in dataset are degrees and converted to radians for trigonometry.
2. POA irradiance uses DNI + DHI only (no GHI and no ground-reflected term).
3. PV generation is zero when solar zenith is >= 90 degrees.
"""

from dataclasses import dataclass
import math
from typing import Dict


def wrap_azimuth_degrees(value: float) -> float:
    """Wrap azimuth angle to [0, 360) degrees."""

    return value % 360.0


def incidence_cosine(
    solar_zenith_deg: float,
    solar_azimuth_deg: float,
    panel_tilt_deg: float,
    panel_azimuth_deg: float,
) -> float:
    """Return cosine of incidence angle between sun vector and panel normal.

    cos(theta_i) = cos(theta_z)cos(theta_p) + sin(theta_z)sin(theta_p)cos(gamma_s - gamma_p)
    """

    theta_z = math.radians(max(0.0, min(180.0, solar_zenith_deg)))
    beta = math.radians(max(0.0, min(90.0, panel_tilt_deg)))
    gamma_s = math.radians(wrap_azimuth_degrees(solar_azimuth_deg))
    gamma_p = math.radians(wrap_azimuth_degrees(panel_azimuth_deg))
    cos_theta_i = (
        math.cos(theta_z) * math.cos(beta)
        + math.sin(theta_z) * math.sin(beta) * math.cos(gamma_s - gamma_p)
    )
    return max(cos_theta_i, 0.0)


@dataclass
class PanelModel:
    """Simple panel model parameters used for power conversion."""

    area_m2: float = 10.0
    efficiency: float = 0.20
    performance_ratio: float = 0.90
    temperature_coefficient: float = 0.004
    reference_temperature_c: float = 25.0
    max_power_kw: float = 0.0

    def rated_power_kw(self) -> float:
        """Approximate rated DC power at 1000 W/m^2."""

        return self.area_m2 * self.efficiency * self.performance_ratio

    def effective_max_power_kw(self) -> float:
        """Return configured max power, or derived rated power when missing."""

        if self.max_power_kw and self.max_power_kw > 0.0:
            return self.max_power_kw

        return self.rated_power_kw()


def compute_plane_of_array_irradiance(
    dni: float,
    dhi: float,
    solar_zenith_deg: float,
    solar_azimuth_deg: float,
    panel_tilt_deg: float,
    panel_azimuth_deg: float,
    dni_min_threshold: float = 0.0,
) -> Dict[str, float]:
    """Return POA irradiance using DNI + DHI isotropic diffuse model.

    I_beam = DNI * max(cos(theta_i), 0)
    I_diffuse = DHI * (1 + cos(theta_p)) / 2
    I_POA = I_beam + I_diffuse
    """

    if float(solar_zenith_deg) >= 90.0 or float(dni) < float(dni_min_threshold):
        return {
            "cos_incidence": 0.0,
            "poa_direct_w_per_m2": 0.0,
            "poa_diffuse_w_per_m2": 0.0,
            "poa_ground_w_per_m2": 0.0,
            "poa_w_per_m2": 0.0,
        }

    cos_theta_i = incidence_cosine(
        solar_zenith_deg=solar_zenith_deg,
        solar_azimuth_deg=solar_azimuth_deg,
        panel_tilt_deg=panel_tilt_deg,
        panel_azimuth_deg=panel_azimuth_deg,
    )
    tilt_rad = math.radians(max(0.0, min(90.0, panel_tilt_deg)))
    beam = max(0.0, float(dni)) * cos_theta_i
    diffuse = max(0.0, float(dhi)) * (1.0 + math.cos(tilt_rad)) / 2.0
    poa = max(0.0, beam + diffuse)

    return {
        "cos_incidence": cos_theta_i,
        "poa_direct_w_per_m2": beam,
        "poa_diffuse_w_per_m2": diffuse,
        "poa_ground_w_per_m2": 0.0,
        "poa_w_per_m2": poa,
    }


def compute_power_and_energy(
    poa_irradiance_w_per_m2: float,
    ambient_temperature_c: float,
    panel: PanelModel,
    seconds_per_time_step: float,
) -> Dict[str, float]:
    """Convert POA irradiance to power (kW) and time-step energy (kWh)."""

    poa = max(0.0, float(poa_irradiance_w_per_m2))
    base_power_kw = poa * panel.area_m2 * panel.efficiency * panel.performance_ratio / 1000.0
    temp_factor = 1.0 - panel.temperature_coefficient * max(0.0, ambient_temperature_c - panel.reference_temperature_c)
    temp_factor = max(0.0, temp_factor)
    power_kw = base_power_kw * temp_factor
    power_kw = min(power_kw, panel.effective_max_power_kw())
    energy_kwh = power_kw * max(0.0, float(seconds_per_time_step)) / 3600.0
    return {"power_kw": power_kw, "energy_kwh": energy_kwh, "temperature_factor": temp_factor}
