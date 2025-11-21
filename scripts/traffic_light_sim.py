"""Visualize a one-way street traffic light with random car arrivals.

Run directly:
    python scripts/traffic_light_sim.py

Adjust parameters with CLI options to see different arrival rates or
cycle timings. Close the plot window to stop the simulation.
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


@dataclass
class LightCycle:
    """Traffic light durations in seconds."""

    green: float = 12.0
    yellow: float = 3.0
    red: float = 8.0

    @property
    def period(self) -> float:
        return self.green + self.yellow + self.red

    def phase(self, t: float) -> str:
        """Return the light phase (green/yellow/red) at elapsed time t."""

        t_mod = t % self.period
        if t_mod < self.green:
            return "green"
        if t_mod < self.green + self.yellow:
            return "yellow"
        return "red"


@dataclass
class Car:
    position: float  # meters from stop line; negative means in queue
    velocity: float = 0.0


class TrafficSimulation:
    """Simulate a one-way approach to a traffic light."""

    def __init__(
        self,
        arrival_rate: float = 0.35,
        car_length: float = 4.5,
        gap: float = 2.5,
        car_speed: float = 10.0,
        exit_distance: float = 80.0,
        cycle: LightCycle | None = None,
        seed: int | None = None,
    ) -> None:
        self.arrival_rate = arrival_rate
        self.car_length = car_length
        self.gap = gap
        self.car_speed = car_speed
        self.exit_distance = exit_distance
        self.cycle = cycle or LightCycle()
        self.rng = np.random.default_rng(seed)
        self.cars: List[Car] = []
        self.time_elapsed = 0.0

    def spawn_cars(self, dt: float) -> None:
        """Spawn new cars based on a Poisson arrival process."""

        arrivals = self.rng.poisson(self.arrival_rate * dt)
        if arrivals == 0:
            return

        if self.cars:
            start_position = self.cars[-1].position - (self.car_length + self.gap)
        else:
            start_position = -6.0

        for i in range(arrivals):
            self.cars.append(Car(position=start_position - i * (self.car_length + self.gap)))

    def update(self, dt: float) -> str:
        """Advance the simulation and return the active light phase."""

        self.spawn_cars(dt)
        phase = self.cycle.phase(self.time_elapsed)

        stop_line = 0.0
        stop_buffer = 1.0
        # Move the lead car first and then enforce headway spacing for following cars
        for i, car in enumerate(self.cars):
            if i == 0:
                if phase == "green" or car.position > stop_line:
                    car.velocity = self.car_speed
                else:
                    target = stop_line - stop_buffer
                    distance_to_target = max(0.0, target - car.position)
                    car.velocity = min(self.car_speed, distance_to_target / dt)
            else:
                lead = self.cars[i - 1]
                target_position = lead.position - (self.car_length + self.gap)
                max_step = max(0.0, (target_position - car.position) / dt)
                # Followers can inch forward even on red, but never exceed the
                # spacing to the vehicle ahead and obey the cruise speed limit.
                car.velocity = min(self.car_speed, max_step)

            car.position += car.velocity * dt

        # Remove cars that cleared the intersection
        self.cars = [car for car in self.cars if car.position < self.exit_distance]
        self.time_elapsed += dt
        return phase


class Visualizer:
    """Matplotlib animation showing the queue and light state."""

    def __init__(self, sim: TrafficSimulation, dt: float, frames: int) -> None:
        self.sim = sim
        self.dt = dt
        self.frames = frames
        self.fig, self.ax = plt.subplots(figsize=(9, 3))
        self.light_patch = plt.Rectangle((1.5, -2.5), 3, 5, color="gray", zorder=1)
        self.ax.add_patch(self.light_patch)
        self.light_circle = plt.Circle((3, 0), 1, color="green", zorder=2)
        self.ax.add_patch(self.light_circle)
        self.car_artists: List[plt.Rectangle] = []

        self.ax.set_xlim(-60, 90)
        self.ax.set_ylim(-5, 5)
        self.ax.set_yticks([])
        self.ax.set_xlabel("Position along street (m)")
        self.ax.axvline(0, color="black", linestyle="--", linewidth=1, label="Stop line")
        self.ax.set_title("One-way traffic light simulation")
        self.ax.legend(loc="upper left")

    def _update_cars(self) -> None:
        """Redraw cars based on simulation state."""

        # Clear old rectangles
        for rect in self.car_artists:
            rect.remove()
        self.car_artists.clear()

        for car in self.sim.cars:
            rect = plt.Rectangle(
                (car.position - self.sim.car_length / 2, -1),
                self.sim.car_length,
                2,
                color="#1f77b4",
                alpha=0.85,
                zorder=3,
            )
            self.ax.add_patch(rect)
            self.car_artists.append(rect)

    def _update_light(self, phase: str) -> None:
        color_map = {"green": "green", "yellow": "gold", "red": "red"}
        self.light_circle.set_color(color_map.get(phase, "gray"))

    def frame(self, _frame: int):
        phase = self.sim.update(self.dt)
        self._update_light(phase)
        self._update_cars()
        return itertools.chain([self.light_circle], self.car_artists)

    def animate(self) -> animation.FuncAnimation:
        return animation.FuncAnimation(
            self.fig,
            self.frame,
            frames=self.frames,
            interval=self.dt * 1000,
            blit=True,
            repeat=False,
        )

    def show(self) -> None:
        self.animate()
        plt.tight_layout()
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrival-rate", type=float, default=0.35, help="Cars per second (lambda)")
    parser.add_argument("--car-length", type=float, default=4.5, help="Vehicle length in meters")
    parser.add_argument("--gap", type=float, default=2.5, help="Desired bumper gap in meters")
    parser.add_argument("--green", type=float, default=12.0, help="Green time in seconds")
    parser.add_argument("--yellow", type=float, default=3.0, help="Yellow time in seconds")
    parser.add_argument("--red", type=float, default=8.0, help="Red time in seconds")
    parser.add_argument("--speed", type=float, default=10.0, help="Cruise speed in m/s during green")
    parser.add_argument("--frames", type=int, default=400, help="Number of animation frames")
    parser.add_argument("--dt", type=float, default=0.2, help="Seconds per frame")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for repeatability")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cycle = LightCycle(green=args.green, yellow=args.yellow, red=args.red)
    sim = TrafficSimulation(
        arrival_rate=args.arrival_rate,
        car_length=args.car_length,
        gap=args.gap,
        car_speed=args.speed,
        cycle=cycle,
        seed=args.seed,
    )
    viz = Visualizer(sim, dt=args.dt, frames=args.frames)
    viz.show()


if __name__ == "__main__":
    main()
