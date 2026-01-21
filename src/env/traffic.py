from __future__ import annotations

import random
from typing import List, Tuple

import carla


def spawn_traffic(
    client: carla.Client,
    world: carla.World,
    traffic_cfg: dict,
    seed: int = 0,
) -> Tuple[List[carla.Actor], carla.TrafficManager]:
    """Spawn background traffic vehicles and walkers using TrafficManager.

    This is a lightweight spawner. You can replace it with ScenarioRunner or
    your own richer scenario generation.
    """
    rng = random.Random(int(seed))

    tm_port = int(traffic_cfg.get('tm_port', 8000))
    tm = client.get_trafficmanager(tm_port)
    tm.set_synchronous_mode(True)
    tm.set_hybrid_physics_mode(bool(traffic_cfg.get('hybrid_physics', True)))
    tm.set_global_distance_to_leading_vehicle(float(traffic_cfg.get('global_distance_to_leading_vehicle', 2.5)))

    actors: List[carla.Actor] = []
    blueprint_library = world.get_blueprint_library()

    # Vehicles
    num_vehicles = int(traffic_cfg.get('num_vehicles', 20))
    spawn_points = world.get_map().get_spawn_points()
    rng.shuffle(spawn_points)

    vehicle_bps = blueprint_library.filter('vehicle.*')
    for sp in spawn_points[:num_vehicles]:
        bp = rng.choice(vehicle_bps)
        if bp.has_attribute('color'):
            bp.set_attribute('color', rng.choice(bp.get_attribute('color').recommended_values))
        try:
            v = world.try_spawn_actor(bp, sp)
        except Exception:
            v = None
        if v is None:
            continue
        v.set_autopilot(True, tm_port)
        actors.append(v)

    # Walkers
    num_walkers = int(traffic_cfg.get('num_walkers', 10))
    walker_bps = blueprint_library.filter('walker.pedestrian.*')
    controller_bp = blueprint_library.find('controller.ai.walker')

    walker_spawn = []
    for _ in range(num_walkers):
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        walker_spawn.append(carla.Transform(loc))

    walkers = []
    walker_controllers = []
    for tf in walker_spawn:
        bp = rng.choice(walker_bps)
        w = world.try_spawn_actor(bp, tf)
        if w is None:
            continue
        walkers.append(w)

    for w in walkers:
        c = world.spawn_actor(controller_bp, carla.Transform(), attach_to=w)
        walker_controllers.append(c)

    for c in walker_controllers:
        c.start()
        c.go_to_location(world.get_random_location_from_navigation())
        c.set_max_speed(1.0 + rng.random())

    actors.extend(walkers)
    actors.extend(walker_controllers)

    return actors, tm
