""" Validate poliastro frames against Orekit ones """

from itertools import product

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import CartesianRepresentation
from astropy.tests.helper import assert_quantity_allclose
from orekit.pyhelpers import setup_orekit_curdir
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.bodies import CelestialBodyFactory
from org.orekit.time import AbsoluteDate
from org.orekit.utils import TimeStampedPVCoordinates
from poliastro.bodies import Earth as Earth_poliastro
from poliastro.bodies import Jupiter as Jupiter_poliastro
from poliastro.bodies import Mars as Mars_poliastro
from poliastro.bodies import Mercury as Mercury_poliastro
from poliastro.bodies import Neptune as Neptune_poliastro
from poliastro.bodies import Saturn as Saturn_poliastro
from poliastro.bodies import Sun as Sun_poliastro
from poliastro.bodies import Uranus as Uranus_poliastro
from poliastro.bodies import Venus as Venus_poliastro
from poliastro.constants import J2000 as J2000_poliastro
from poliastro.frames.equatorial import GCRS as GCRS_poliastro
from poliastro.frames.equatorial import HCRS as HCRS_poliastro
from poliastro.frames.equatorial import JupiterICRS as JupiterICRS_poliastro
from poliastro.frames.equatorial import MarsICRS as MarsICRS_poliastro
from poliastro.frames.equatorial import MercuryICRS as MercuryICRS_poliastro
from poliastro.frames.equatorial import NeptuneICRS as NeptuneICRS_poliastro
from poliastro.frames.equatorial import SaturnICRS as SaturnICRS_poliastro
from poliastro.frames.equatorial import UranusICRS as UranusICRS_poliastro
from poliastro.frames.equatorial import VenusICRS as VenusICRS_poliastro
from poliastro.frames.fixed import ITRS as ITRS_poliastro
from poliastro.frames.fixed import JupiterFixed as JupiterFixed_poliastro
from poliastro.frames.fixed import MarsFixed as MarsFixed_poliastro
from poliastro.frames.fixed import MercuryFixed as MercuryFixed_poliastro
from poliastro.frames.fixed import NeptuneFixed as NeptuneFixed_poliastro
from poliastro.frames.fixed import SaturnFixed as SaturnFixed_poliastro
from poliastro.frames.fixed import SunFixed as SunFixed_poliastro
from poliastro.frames.fixed import UranusFixed as UranusFixed_poliastro
from poliastro.frames.fixed import VenusFixed as VenusFixed_poliastro

import orekit

# Setup orekit virtual machine and associated data
VM = orekit.initVM()
setup_orekit_curdir("orekit-data.zip")

# All interesting 3D directions in first quadrant
R_SET, V_SET = [product([0, 1], repeat=3) for _ in range(2)]

# Retrieve celestial bodies from orekit
Sun_orekit = CelestialBodyFactory.getSun()
Mercury_orekit = CelestialBodyFactory.getMercury()
Venus_orekit = CelestialBodyFactory.getVenus()
Earth_orekit = CelestialBodyFactory.getEarth()
Mars_orekit = CelestialBodyFactory.getMars()
Jupiter_orekit = CelestialBodyFactory.getJupiter()
Saturn_orekit = CelestialBodyFactory.getSaturn()
Uranus_orekit = CelestialBodyFactory.getUranus()
Neptune_orekit = CelestialBodyFactory.getNeptune()

# Name of the bodies
BODIES_NAMES = [
    "Sun",
    "Mercury",
    "Venus",
    "Earth",
    "Mars",
    "Jupiter",
    "Saturn",
    "Uranus",
    "Neptune",
]

# orekit: bodies, inertial and fixed frames
OREKIT_BODIES = [
    Sun_orekit,
    Mercury_orekit,
    Venus_orekit,
    Earth_orekit,
    Mars_orekit,
    Jupiter_orekit,
    Saturn_orekit,
    Uranus_orekit,
    Neptune_orekit,
]
OREKIT_INERTIAL_FRAMES = [body.getInertiallyOrientedFrame() for body in OREKIT_BODIES]
OREKIT_FIXED_FRAMES = [body.getBodyOrientedFrame() for body in OREKIT_BODIES]

# poliastro: bodies, intertial and fixed frames
POLIASTRO_BODIES = [
    Sun_poliastro,
    Mercury_poliastro,
    Venus_poliastro,
    Earth_poliastro,
    Mars_poliastro,
    Jupiter_poliastro,
    Saturn_poliastro,
    Uranus_poliastro,
    Neptune_poliastro,
]
POLIASTRO_INERTIAL_FRAMES = [
    HCRS_poliastro,
    MercuryICRS_poliastro,
    VenusICRS_poliastro,
    GCRS_poliastro,
    MarsICRS_poliastro,
    JupiterICRS_poliastro,
    SaturnICRS_poliastro,
    UranusICRS_poliastro,
    NeptuneICRS_poliastro,
]
POLIASTRO_FIXED_FRAMES = [
    SunFixed_poliastro,
    MercuryFixed_poliastro,
    VenusFixed_poliastro,
    ITRS_poliastro,
    MarsFixed_poliastro,
    JupiterFixed_poliastro,
    SaturnFixed_poliastro,
    UranusFixed_poliastro,
    NeptuneFixed_poliastro,
]


# Collect both API data in two dicitonaries
OREKIT_BODIES_AND_FRAMES = dict(
    zip(BODIES_NAMES, zip(OREKIT_BODIES, OREKIT_INERTIAL_FRAMES, OREKIT_FIXED_FRAMES))
)
POLIASTRO_BODIES_AND_FRAMES = dict(
    zip(
        BODIES_NAMES,
        zip(POLIASTRO_BODIES, POLIASTRO_INERTIAL_FRAMES, POLIASTRO_FIXED_FRAMES),
    )
)

J2000_orekit = AbsoluteDate.J2000_EPOCH


@pytest.mark.parametrize("r_vec", R_SET)
@pytest.mark.parametrize("v_vec", V_SET)
@pytest.mark.parametrize("body_name", BODIES_NAMES)
def validate_from_body_intertial_to_body_fixed(body_name, r_vec, v_vec):

    # Unpack vectors components
    rx, ry, rz = [float(r_i) for r_i in r_vec]
    vx, vy, vz = [float(v_i) for v_i in v_vec]

    # orekit: collect body information
    (
        body_orekit,
        body_frame_inertial_orekit,
        body_fixed_frame_orekit,
    ) = OREKIT_BODIES_AND_FRAMES[body_name]

    # orekit: build r_vec and v_vec wrt inertial body frame
    xyz_orekit = Vector3D(rx, ry, rz)
    uvw_orekit = Vector3D(vx, vy, vz)
    coords_orekit_inertial = TimeStampedPVCoordinates(
        J2000_orekit, xyz_orekit, uvw_orekit
    )

    # orekit: build conversion between inertial and non-inertial frames
    body_inertial_to_fixed = body_frame_inertial_orekit.getTransformTo(
        body_fixed_frame_orekit,
        J2000_orekit,
    )

    # orekit: convert from inertial coordinates to non-inertial ones
    coords_orekit_fixed_raw = (
        body_inertial_to_fixed.transformPVCoordinates(coords_orekit_inertial)
        .getPosition()
        .toArray()
    )
    coords_orekit_fixed = np.asarray(coords_orekit_fixed_raw) * u.m

    # poliastro: collect body information
    (
        body_poliastro,
        body_frame_inertial_poliastro,
        body_fixed_frame_poliastro,
    ) = POLIASTRO_BODIES_AND_FRAMES[body_name]

    # poliastro: build r_vec and v_vec wrt inertial body frame
    xyz_poliastro = CartesianRepresentation(rx * u.m, ry * u.m, rz * u.m)
    coords_poliastro_inertial = body_frame_inertial_poliastro(xyz_poliastro)

    # poliastro: convert from inertial to fixed frame at given epoch
    coords_poliastro_fixed = (
        coords_poliastro_inertial.transform_to(
            body_fixed_frame_poliastro(obstime=J2000_poliastro)
        )
        .represent_as(CartesianRepresentation)
        .xyz
    )

    # Check if all quantities are similar
    assert_quantity_allclose(coords_poliastro_fixed, coords_orekit_fixed)
