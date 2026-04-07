"""
NatureSim — Forest Quad-System Simulation
==========================================
Four interacting agentic systems:
  1. Weather  — cycles rain/drought, shifts wind, drives fire risk
  2. Vegetation — grows, spreads via seeds, fuels fire, feeds animals
  3. Fire     — spreads toward fuel, wind-biased, suppressed by rain
  4. Animals  — flee fire, seek food, reproduce, disperse seeds
"""

import math
import random
from dataclasses import dataclass
from typing import List

import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────

GRID_WIDTH  = 60
GRID_HEIGHT = 60

# Vegetation
VEG_GROWTH_BASE       = 0.008   # base regrowth per tick (logistic)
VEG_MOISTURE_BONUS    = 0.60    # extra growth multiplier when moist
VEG_SPREAD_RATE       = 0.003   # seed diffusion strength to neighbours
VEG_SEED_DROP_CHANCE  = 0.003   # chance an animal drops a seed each tick
VEG_FIRE_CONSUME_RATE = 0.18    # fraction of vegetation burned per tick

# Fire
FIRE_SPREAD_BASE      = 0.055   # baseline spread probability per neighbour
FIRE_VEG_FACTOR       = 0.35    # fuel amplifies spread
FIRE_DROUGHT_FACTOR   = 0.25    # drought amplifies spread
FIRE_DECAY_BASE       = 0.06    # burn-out rate on low fuel
FIRE_RAIN_SUPPRESS    = 0.20    # rain extinguishes this fraction per tick
FIRE_LIGHTNING_CHANCE = 0.00004 # random ignition probability per cell per tick

# Weather
RAIN_MIN_DAYS   = 3
RAIN_MAX_DAYS   = 18
DRY_MIN_DAYS    = 8
DRY_MAX_DAYS    = 55
MOISTURE_GAIN   = 0.06   # moisture added per tick while raining
MOISTURE_EVAP   = 0.008  # moisture lost per tick (evaporation)
WIND_CHANGE_P   = 0.04   # probability wind direction changes each tick

# Animals
ANIMAL_INIT_COUNT        = 18
ANIMAL_MAX_COUNT         = 180
ANIMAL_MOVE_SPEED        = 2.5   # max displacement per tick (grid cells)
ANIMAL_GRAZE_AMOUNT      = 0.006 # vegetation consumed per tick
ANIMAL_SEED_DROP_CHANCE  = VEG_SEED_DROP_CHANCE
ANIMAL_SEED_BOOST        = 0.18  # vegetation added when seed drops
ANIMAL_REPRODUCE_CHANCE  = 0.025
ANIMAL_REPRO_ENERGY_MIN  = 0.65
ANIMAL_REPRO_VEG_MIN     = 0.55
ANIMAL_FIRE_DEATH_THRESH = 0.40  # fire intensity lethal to animals
ANIMAL_STARVE_DAYS       = 25    # consecutive hungry ticks before death


# ── Animal entity ─────────────────────────────────────────────────────────────

@dataclass
class Animal:
    x: float
    y: float
    energy: float = 1.0
    hunger: int   = 0   # consecutive ticks without food
    age: int      = 0

    def gpos(self, w: int, h: int):
        """Return (grid_x, grid_y) integer cell position."""
        return int(self.x) % w, int(self.y) % h


# ── Weather system ────────────────────────────────────────────────────────────

class Weather:
    """
    Agentic weather: autonomously cycles rain/drought, shifts wind,
    and exposes a fire-risk index used by the Fire system.
    """

    def __init__(self):
        self.raining      = False
        self.rain_left    = 0
        self.drought_days = 0
        self.wind_dx      = random.choice([-1, 0, 1])
        self.wind_dy      = random.choice([-1, 0, 1])
        self.wind_speed   = random.uniform(0.8, 1.5)
        self.humidity     = 0.45
        self.temperature  = 0.50

    def step(self):
        if self.raining:
            self.rain_left   -= 1
            self.drought_days = 0
            self.humidity     = min(1.0, self.humidity + 0.04)
            self.temperature  = max(0.2, self.temperature - 0.01)
            if self.rain_left <= 0:
                self.raining = False
        else:
            self.drought_days += 1
            self.humidity      = max(0.05, self.humidity - 0.010)
            self.temperature   = min(1.00, self.temperature + 0.005)
            # Rain probability grows with drought length
            p_rain = 0.008 + min(self.drought_days / 70.0, 1.0) * 0.06
            if random.random() < p_rain:
                self.raining   = True
                self.rain_left = random.randint(RAIN_MIN_DAYS, RAIN_MAX_DAYS)

        # Wind occasionally shifts
        if random.random() < WIND_CHANGE_P:
            self.wind_dx    = random.choice([-1, 0, 1])
            self.wind_dy    = random.choice([-1, 0, 1])
            self.wind_speed = random.uniform(0.5, 2.2)

    @property
    def fire_risk(self) -> float:
        """Composite 0–1 index: drought length × heat × low humidity."""
        drought = min(self.drought_days / 50.0, 1.0)
        return drought * (1.0 - self.humidity) * (self.temperature ** 0.5)

    @property
    def status_str(self) -> str:
        if self.raining:
            return f"Rain ({self.rain_left}d left)"
        return f"Drought {self.drought_days}d | Risk {self.fire_risk:.2f}"


# ── Main simulation ───────────────────────────────────────────────────────────

class NatureSim:
    """
    Coordinates all four systems each tick.
    Call step() to advance by one day.
    """

    def __init__(self, width: int = GRID_WIDTH, height: int = GRID_HEIGHT):
        self.width  = width
        self.height = height
        self.tick   = 0

        # Grid state — float32 arrays (H × W)
        rng = np.random.default_rng()
        self.vegetation = rng.uniform(0.40, 0.90, (height, width)).astype(np.float32)
        self.fire       = np.zeros((height, width), dtype=np.float32)
        self.moisture   = rng.uniform(0.30, 0.60, (height, width)).astype(np.float32)
        self.scorched   = np.zeros((height, width), dtype=np.float32)

        self.weather: Weather        = Weather()
        self.animals:  List[Animal]  = []

        # Statistics history (lists grown each tick)
        self.history = {
            "tick":          [],
            "veg_mean":      [],
            "fire_cells":    [],
            "animal_count":  [],
            "moisture_mean": [],
            "drought_days":  [],
            "fire_risk":     [],
        }

        self._seed_animals()
        self._seed_fire()

    # ── Initialization ────────────────────────────────────────────────────────

    def _seed_animals(self):
        for _ in range(ANIMAL_INIT_COUNT):
            self.animals.append(Animal(
                x=random.uniform(0, self.width),
                y=random.uniform(0, self.height),
            ))

    def _seed_fire(self):
        """Place one natural fire cluster away from borders."""
        cx = random.randint(10, self.width  - 10)
        cy = random.randint(10, self.height - 10)
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                if (dx**2 + dy**2) ** 0.5 <= 3.0:
                    nx = (cx + dx) % self.width
                    ny = (cy + dy) % self.height
                    self.fire[ny, nx] = float(np.clip(
                        random.gauss(0.70, 0.18), 0.20, 1.00
                    ))

    # ── Public step ───────────────────────────────────────────────────────────

    def step(self):
        """Advance all four systems by one tick (one simulated day)."""
        self.tick += 1
        self.weather.step()
        self._step_moisture()
        self._step_fire()
        self._step_vegetation()
        self._step_animals()
        self._record_stats()

    # ── Moisture (shared resource) ────────────────────────────────────────────

    def _step_moisture(self):
        if self.weather.raining:
            self.moisture += MOISTURE_GAIN
        self.moisture -= MOISTURE_EVAP
        np.clip(self.moisture, 0.0, 1.0, out=self.moisture)

    # ── Fire system ───────────────────────────────────────────────────────────

    def _step_fire(self):
        fire  = self.fire
        h, w  = self.height, self.width
        risk  = self.weather.fire_risk
        rain  = self.weather.raining
        wdx   = self.weather.wind_dx
        wdy   = self.weather.wind_dy
        wspd  = self.weather.wind_speed

        # Decay / rain suppression
        if rain:
            new_fire = np.maximum(0.0, fire - FIRE_RAIN_SUPPRESS)
        else:
            decay    = FIRE_DECAY_BASE * (1.0 - self.vegetation * 0.40)
            new_fire = np.maximum(0.0, fire - decay)

        # Spread: destination-cell spread probability (vectorised over 8 dirs)
        base_prob = (
            FIRE_SPREAD_BASE
            + FIRE_VEG_FACTOR    * self.vegetation
            + FIRE_DROUGHT_FACTOR * risk
        )

        rand_buf  = np.random.random((h, w)).astype(np.float32)
        ignit_buf = np.random.uniform(0.55, 0.95, (h, w)).astype(np.float32)

        for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
            src = np.roll(np.roll(fire, -dy, axis=0), -dx, axis=1)
            active = src > 0.05

            wind_bonus = (1.0 + wspd) if (dx == wdx and dy == wdy) else 1.0
            prob = np.clip(base_prob * wind_bonus * src, 0.0, 1.0)

            spread = active & (new_fire == 0.0) & (rand_buf < prob)
            # Refresh random buffer slice for next direction (cheap approximation)
            rand_buf = np.roll(rand_buf, 7, axis=0)

            new_fire = np.where(spread, src * ignit_buf, new_fire)

        # Lightning ignitions
        if not rain and self.weather.drought_days > 5:
            p_lightning = FIRE_LIGHTNING_CHANCE * (1.0 + risk * 4.0)
            ignite = (
                (np.random.random((h, w)) < p_lightning)
                & (self.vegetation > 0.25)
                & (new_fire == 0.0)
            )
            new_fire = np.where(
                ignite,
                np.random.uniform(0.35, 0.75, (h, w)).astype(np.float32),
                new_fire,
            )

        self.fire = np.clip(new_fire, 0.0, 1.0).astype(np.float32)

    # ── Vegetation system ─────────────────────────────────────────────────────

    def _step_vegetation(self):
        veg = self.vegetation

        # Fire consumes fuel
        veg -= self.fire * VEG_FIRE_CONSUME_RATE

        # Track scorched earth (slow to recover)
        burning = self.fire > 0.05
        self.scorched = np.where(
            burning,
            np.minimum(1.0, self.scorched + 0.12),
            np.maximum(0.0, self.scorched - 0.004),
        ).astype(np.float32)

        # Natural logistic regrowth (inhibited by scorched earth)
        growth_factor  = 1.0 - self.scorched * 0.75
        moisture_bonus = 1.0 + self.moisture * VEG_MOISTURE_BONUS
        growth = VEG_GROWTH_BASE * growth_factor * moisture_bonus * (1.0 - veg)
        veg += growth

        # Seed diffusion to 4-neighbours
        neighbour_avg = (
            np.roll(veg,  1, axis=0) + np.roll(veg, -1, axis=0) +
            np.roll(veg,  1, axis=1) + np.roll(veg, -1, axis=1)
        ) * 0.25
        veg += (neighbour_avg - veg) * VEG_SPREAD_RATE

        self.vegetation = np.clip(veg, 0.0, 1.0).astype(np.float32)

    # ── Animal system ─────────────────────────────────────────────────────────

    def _step_animals(self):
        survivors: List[Animal] = []
        w, h = self.width, self.height

        for animal in self.animals:
            gx, gy = animal.gpos(w, h)

            # Fire mortality (escape chance inversely proportional to intensity)
            if self.fire[gy, gx] > ANIMAL_FIRE_DEATH_THRESH:
                if random.random() > (1.0 - self.fire[gy, gx]):
                    continue  # perishes in fire

            # Starvation
            if animal.hunger >= ANIMAL_STARVE_DAYS:
                continue

            # Movement: choose direction that maximises food & avoids fire
            dx, dy = self._choose_move(animal, gx, gy)
            animal.x = (animal.x + dx) % w
            animal.y = (animal.y + dy) % h
            gx, gy = animal.gpos(w, h)

            # Grazing
            if self.vegetation[gy, gx] > 0.08:
                self.vegetation[gy, gx] = max(
                    0.0, self.vegetation[gy, gx] - ANIMAL_GRAZE_AMOUNT
                )
                animal.hunger = 0
                animal.energy = min(1.0, animal.energy + 0.06)

                # Seed dispersal — animal spreads seeds while foraging
                if random.random() < ANIMAL_SEED_DROP_CHANCE:
                    sx = int(animal.x + random.uniform(-4.0, 4.0)) % w
                    sy = int(animal.y + random.uniform(-4.0, 4.0)) % h
                    self.vegetation[sy, sx] = min(
                        1.0, self.vegetation[sy, sx] + ANIMAL_SEED_BOOST
                    )
            else:
                animal.hunger += 1
                animal.energy  = max(0.0, animal.energy - 0.025)

            animal.age += 1

            # Reproduction
            if (
                animal.energy >= ANIMAL_REPRO_ENERGY_MIN
                and self.vegetation[gy, gx] >= ANIMAL_REPRO_VEG_MIN
                and len(survivors) < ANIMAL_MAX_COUNT
                and random.random() < ANIMAL_REPRODUCE_CHANCE
            ):
                offspring = Animal(
                    x=float((animal.x + random.uniform(-1.5, 1.5)) % w),
                    y=float((animal.y + random.uniform(-1.5, 1.5)) % h),
                    energy=0.45,
                )
                survivors.append(offspring)
                animal.energy -= 0.25   # reproduction costs energy

            survivors.append(animal)

        # Hard cap: if population exceeds maximum, cull randomly
        if len(survivors) > ANIMAL_MAX_COUNT:
            survivors = random.sample(survivors, ANIMAL_MAX_COUNT)
        self.animals = survivors

    def _choose_move(self, animal: Animal, gx: int, gy: int):
        """
        Sample 10 candidate moves; return the (dx, dy) with the best score.
        Score = food ahead − fire ahead (strongly penalised).
        """
        w, h = self.width, self.height
        best_score = -1e9
        best_dx = best_dy = 0.0
        spd = ANIMAL_MOVE_SPEED

        for _ in range(10):
            angle = random.uniform(0.0, 2.0 * math.pi)
            mag   = random.uniform(0.2, spd)
            dx    = mag * math.cos(angle)
            dy    = mag * math.sin(angle)

            nx = int(animal.x + dx)     % w
            ny = int(animal.y + dy)     % h
            lx = int(animal.x + dx * 2) % w
            ly = int(animal.y + dy * 2) % h

            veg_score  =  self.vegetation[ny, nx] * 2.0 + self.vegetation[ly, lx]
            fire_score = -(self.fire[ny, nx] * 8.0     + self.fire[ly, lx] * 4.0)
            noise      =  random.uniform(-0.10, 0.10)

            score = veg_score + fire_score + noise
            if score > best_score:
                best_score = score
                best_dx, best_dy = dx, dy

        return best_dx, best_dy

    # ── Statistics ────────────────────────────────────────────────────────────

    def _record_stats(self):
        h = self.history
        h["tick"].append(self.tick)
        h["veg_mean"].append(float(np.mean(self.vegetation)))
        h["fire_cells"].append(int(np.sum(self.fire > 0.05)))
        h["animal_count"].append(len(self.animals))
        h["moisture_mean"].append(float(np.mean(self.moisture)))
        h["drought_days"].append(self.weather.drought_days)
        h["fire_risk"].append(self.weather.fire_risk)

    # ── Rendering helpers ─────────────────────────────────────────────────────

    def get_rgb(self) -> np.ndarray:
        """
        Returns a float32 (H × W × 3) RGB array suitable for imshow.

        Color encoding:
          Green  — vegetation density
          Red    — fire intensity
          Blue   — soil moisture
          Brown  — scorched / bare earth
        """
        veg = self.vegetation
        fir = self.fire
        mst = self.moisture
        sc  = self.scorched

        # Base: brownish bare earth
        r = np.full_like(veg, 0.30)
        g = np.full_like(veg, 0.20)
        b = np.full_like(veg, 0.10)

        # Vegetation → greens
        g += veg * 0.65
        r -= veg * 0.10
        b += veg * 0.05

        # Moisture → blue tint
        b += mst * 0.20
        g += mst * 0.04

        # Scorched earth → darker, browner
        r -= sc * 0.15
        g -= sc * 0.18
        b -= sc * 0.08

        # Fire → orange/red, suppress green/blue
        r += fir * 0.70
        g += fir * veg * 0.25   # burning vegetation yields orange
        g -= fir * 0.10
        b -= fir * 0.20

        return np.clip(np.stack([r, g, b], axis=2), 0.0, 1.0).astype(np.float32)

    def print_summary(self):
        """One-line summary for headless / logging use."""
        print(
            f"Day {self.tick:4d} | "
            f"Veg {np.mean(self.vegetation):.3f} | "
            f"Fire {np.sum(self.fire > 0.05):4d} cells | "
            f"Animals {len(self.animals):3d} | "
            f"Weather: {self.weather.status_str}"
        )
