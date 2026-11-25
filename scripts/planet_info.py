"""
Star + planet profile for one KOI, plus a Mars reference to compare against.

Kepler's transit method measures a planet's radius, orbital period and the
host star's temperature/size/mass. Everything else here (density, atmosphere
type, livability) is INFERRED from those, not measured, and the app labels it
that way. Distance is orbital distance from the star (AU), from Kepler's third
law; Kepler's table has no distance from Earth.
"""

import math

import pandas as pd

# Earth = 1 for radius/mass/insolation. Densities in g/cm^3.
EARTH_DENSITY = 5.51
EARTH_TEQ_K = 255.0     # Earth's equilibrium temperature (no greenhouse)
EARTH_VESC = 11.19      # km/s

# Mars, for the comparison.
MARS = {
    "name": "Mars",
    "radius_earth": 0.532,
    "mass_earth": 0.107,
    "density": 3.93,
    "teq_k": 210.0,
    "insolation": 0.43,
    "orbit_au": 1.524,
    "period_days": 687.0,
    "star_type": "G",
    "star_teff": 5772.0,
    "category": "Rocky planet",
    "atmosphere": "Very thin, about 95% carbon dioxide",
}


def star_class(teff):
    """Spectral class from effective temperature (K)."""
    if teff is None or pd.isna(teff):
        return "Unknown"
    for limit, name in [(3900, "M"), (5300, "K"), (6000, "G"), (7300, "F"), (10000, "A")]:
        if teff < limit:
            return name
    return "B/O"


STAR_CLASS_NOTE = {
    "M": "Red dwarf: small, cool and dim. The most common kind of star.",
    "K": "Orange dwarf: a bit cooler and smaller than the Sun, very long-lived.",
    "G": "Yellow dwarf: Sun-like.",
    "F": "Yellow-white: hotter and brighter than the Sun, shorter-lived.",
    "A": "White: hot and bright, rotates fast.",
    "B/O": "Blue giant: very hot and short-lived.",
    "Unknown": "Temperature not in NASA's table.",
}


def planet_category(radius):
    """Size class from radius in Earth radii."""
    if radius is None or pd.isna(radius):
        return "Unknown"
    if radius < 1.25:
        return "Earth-sized (rocky)"
    if radius < 2.0:
        return "Super-Earth"
    if radius < 4.0:
        return "Mini-Neptune"
    if radius < 8.0:
        return "Neptune-sized"
    return "Gas giant"


ATMOSPHERE_NOTE = {
    "Earth-sized (rocky)": "Rocky surface. Could hold a thin atmosphere of nitrogen, "
                           "carbon dioxide or water vapour, or none at all.",
    "Super-Earth": "Mostly rock and metal. May keep a heavier atmosphere; a thick "
                   "hydrogen layer is possible at the upper end.",
    "Mini-Neptune": "Likely a thick hydrogen and helium envelope over a rocky or icy "
                    "core, with water and methane.",
    "Neptune-sized": "Hydrogen and helium with a lot of water, methane and ammonia ice. "
                     "No solid surface.",
    "Gas giant": "Almost all hydrogen and helium, like Jupiter. No solid surface.",
    "Unknown": "Radius not in NASA's table.",
}


def estimate_mass(radius):
    """Rough mass (Earth masses) from radius. Transits give radius, not mass,
    so this is a typical-planet estimate, not a measurement."""
    if radius is None or pd.isna(radius):
        return None
    if radius < 1.5:                       # Weiss & Marcy 2014, rocky planets
        density = 2.43 + 3.39 * radius
        return density / EARTH_DENSITY * radius ** 3
    if radius < 4.0:                       # Weiss & Marcy 2014, volatile-rich
        return 2.69 * radius ** 0.93
    mass = 17.0 * (min(radius, 11.2) / 3.9) ** 2.78   # Neptune -> Jupiter
    return mass


def density_g_cc(radius, mass):
    if radius is None or mass is None or pd.isna(radius) or radius <= 0:
        return None
    return EARTH_DENSITY * mass / radius ** 3


def orbit_au(period_days, star_mass):
    """Orbital distance from Kepler's third law, a^3 = M * P^2 (AU, solar mass, years)."""
    if period_days is None or star_mass is None or pd.isna(period_days) or pd.isna(star_mass):
        return None
    return (star_mass * (period_days / 365.25) ** 2) ** (1 / 3)


def escape_velocity(radius, mass):
    if radius is None or mass is None or radius <= 0:
        return None
    return EARTH_VESC * math.sqrt(mass / radius)


def _esi_term(x, x0, weight):
    return (1 - abs(x - x0) / (x + x0)) ** weight


def earth_similarity(radius, density, vesc, teq):
    """Earth Similarity Index (Schulze-Makuch 2011 weights) from the four inputs
    we can estimate. 1.0 is Earth. It is a similarity score, not a promise of life."""
    if None in (radius, density, vesc, teq):
        return None
    terms = [
        _esi_term(radius, 1.0, 0.57), _esi_term(density / EARTH_DENSITY, 1.0, 1.07),
        _esi_term(vesc / EARTH_VESC, 1.0, 0.70), _esi_term(teq, EARTH_TEQ_K, 5.58),
    ]
    out = 1.0
    for t in terms:
        out *= t
    return out ** (1 / 4)


def livability(radius, insolation, teq):
    """Plain-language read. Habitable zone here means liquid water is plausible on a
    rocky surface; it does not mean the planet has water or life."""
    if insolation is None or pd.isna(insolation):
        return "Unknown", "Starlight level not in NASA's table."
    in_zone = 0.32 <= insolation <= 1.7
    rocky = radius is not None and not pd.isna(radius) and radius < 1.8
    if insolation > 1.7:
        return "Too hot", f"Gets {insolation:.1f}x Earth's starlight, hot enough to boil oceans."
    if insolation < 0.32:
        return "Too cold", f"Gets only {insolation:.2f}x Earth's starlight, so water would freeze."
    if in_zone and rocky:
        return "Possibly livable", ("In the habitable zone and small enough to be rocky, so "
                                    "liquid water could exist. Atmosphere unknown.")
    return "Right temperature, wrong planet", ("In the habitable zone, but big enough that it "
                                               "is probably a gas planet with no surface.")


def build_profile(row):
    """row: one line of the NASA KOI table (a pandas Series)."""
    radius = row.get("koi_prad")
    mass = estimate_mass(radius)
    density = density_g_cc(radius, mass)
    vesc = escape_velocity(radius, mass)
    teq = row.get("koi_teq")
    insolation = row.get("koi_insol")
    verdict, verdict_why = livability(radius, insolation, teq)
    category = planet_category(radius)
    star = star_class(row.get("koi_steff"))
    return {
        "radius_earth": radius, "mass_earth": mass, "density": density,
        "teq_k": teq, "insolation": insolation, "period_days": row.get("koi_period"),
        "orbit_au": orbit_au(row.get("koi_period"), row.get("koi_smass")),
        "category": category, "atmosphere": ATMOSPHERE_NOTE[category],
        "star_class": star, "star_note": STAR_CLASS_NOTE[star],
        "star_teff": row.get("koi_steff"), "star_radius_sun": row.get("koi_srad"),
        "star_mass_sun": row.get("koi_smass"),
        "esi": earth_similarity(radius, density, vesc, teq),
        "livability": verdict, "livability_why": verdict_why,
    }


def mars_profile():
    esi = earth_similarity(MARS["radius_earth"], MARS["density"],
                           escape_velocity(MARS["radius_earth"], MARS["mass_earth"]),
                           MARS["teq_k"])
    return {**MARS, "esi": esi}


def load_koi_table(path):
    return pd.read_csv(path, low_memory=False).set_index("kepoi_name")
