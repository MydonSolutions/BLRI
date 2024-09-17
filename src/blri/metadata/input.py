from typing import Optional, List

from pydantic import BaseModel

class InputMetaData(BaseModel):
    nof_antenna: int
    nof_channel: int
    nof_time: int
    nof_polarisation: int
    channel_bandwidth_mhz: float
    observed_frequency_mhz: float
    polarisation_chars: str
    phase_center_rightascension_radians: float
    phase_center_declination_radians: float
    dut1_s: float
    spectra_timespan_s: float
    telescope: str
    source_name: str
    antenna_names: Optional[List[str]]