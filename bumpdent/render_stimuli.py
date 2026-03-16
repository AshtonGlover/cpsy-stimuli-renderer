#!/usr/bin/env python3
"""Render stacked bump and dent stimuli with adjustable lighting."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def smoothstep(edge0: np.ndarray, edge1: np.ndarray, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / np.clip(edge1 - edge0, 1e-8, None), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def normalize(vectors: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.clip(lengths, 1e-8, None)


def vec_from_angles(azimuth_deg: float, elevation_deg: float) -> tuple[float, float, float]:
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    c = math.cos(el)
    x = c * math.cos(az)
    y = c * math.sin(az)
    z = math.sin(el)
    mag = math.sqrt(x * x + y * y + z * z)
    if mag <= 1e-8:
        return (0.0, 0.0, 1.0)
    return (x / mag, y / mag, z / mag)


@dataclass
class RenderParams:
    width: int = 420
    height: int = 520
    background: int = 182
    albedo: int = 196
    radius: float = 96.0
    center_x: float = 0.0
    center_y: float = 0.0
    vertical_gap: float = 38.0
    top_is_concave: bool = False
    bump_strength: float = 1.0
    dent_strength: float = 1.0
    light_x: float = 0.0
    light_y: float = 0.75
    light_z: float = 1.1
    ambient: float = 0.28
    diffuse: float = 0.82
    specular: float = 0.18
    shininess: float = 20.0
    edge_softness: float = 1.8
    use_cosine_falloff: bool = True
    use_flat_profile: bool = False
    shadow_enabled: bool = False
    shadow_azimuth: float = 220.0
    shadow_elevation: float = 30.0
    shadow_strength: float = 0.45
    shadow_softness: float = 0.9
    shadow_distance: float = 45.0


def _shade_disc(
    xx: np.ndarray,
    yy: np.ndarray,
    center_x: float,
    center_y: float,
    radius: float,
    concave: bool,
    strength: float,
    params: RenderParams,
) -> tuple[np.ndarray, np.ndarray]:
    dx = xx - center_x
    dy_up = center_y - yy
    dist2 = dx * dx + dy_up * dy_up
    radial_distance = np.sqrt(np.clip(dist2, 0.0, None))
    radial = radial_distance / radius
    inside = radial <= 1.0

    image = np.zeros_like(xx, dtype=np.float32)
    alpha = np.zeros_like(xx, dtype=np.float32)
    if not np.any(inside):
        return image, alpha

    sign = -1.0 if concave else 1.0
    if params.use_cosine_falloff:
        height_scale = radius * 0.34 * strength
        profile = np.zeros_like(xx, dtype=np.float32)
        if params.use_flat_profile:
            # A squared cosine cap keeps the center flatter and reduces the "ball" look.
            profile[inside] = (0.5 * (1.0 + np.cos(np.pi * radial[inside]))) ** 2
        else:
            profile[inside] = 0.5 * (1.0 + np.cos(np.pi * radial[inside]))

        d_profile_dr = np.zeros_like(xx, dtype=np.float32)
        if params.use_flat_profile:
            base = 0.5 * (1.0 + np.cos(np.pi * radial[inside]))
            d_base_dr = -0.5 * np.pi * np.sin(np.pi * radial[inside])
            d_profile_dr[inside] = 2.0 * base * d_base_dr
        else:
            d_profile_dr[inside] = -0.5 * np.pi * np.sin(np.pi * radial[inside])
        d_height_dr = sign * height_scale * d_profile_dr / radius

        safe_distance = np.where(radial_distance > 1e-6, radial_distance, 1.0)
        dz_dx = d_height_dr * (dx / safe_distance)
        dz_dy = d_height_dr * (dy_up / safe_distance)
        dz_dx[~inside] = 0.0
        dz_dy[~inside] = 0.0

        height = sign * height_scale * profile
        height[~inside] = 0.0

        normals = np.dstack((-dz_dx, -dz_dy, np.ones_like(xx, dtype=np.float32)))
    else:
        nx = np.zeros_like(xx, dtype=np.float32)
        ny = np.zeros_like(xx, dtype=np.float32)
        nz = np.zeros_like(xx, dtype=np.float32)

        nx[inside] = dx[inside] / radius
        ny[inside] = dy_up[inside] / radius
        nz[inside] = np.sqrt(np.clip(1.0 - nx[inside] ** 2 - ny[inside] ** 2, 0.0, 1.0))

        nx[inside] *= sign * strength
        ny[inside] *= sign * strength
        normals = np.dstack((nx, ny, nz))
        height = np.zeros_like(xx, dtype=np.float32)

    normals = normalize(normals)

    px = (xx - xx.mean()) / max(xx.shape[1], 1)
    py = -(yy - yy.mean()) / max(yy.shape[0], 1)
    pz = height / max(xx.shape[1], yy.shape[0], 1)
    points = np.dstack((px, py, pz))
    light_direction = normalize(
        np.array([[[params.light_x, params.light_y, params.light_z]]], dtype=np.float32)
    )
    lambert = np.clip(np.sum(normals * light_direction, axis=-1), 0.0, 1.0)

    view_vectors = np.zeros_like(points)
    view_vectors[..., 2] = 1.0
    half_vectors = normalize(light_direction + view_vectors)
    highlights = np.clip(np.sum(normals * half_vectors, axis=-1), 0.0, 1.0) ** params.shininess

    lit = (
        params.ambient
        + params.diffuse * lambert
        + params.specular * highlights
    )
    lit = np.clip(lit, 0.0, 1.2)

    edge_distance = radius - np.sqrt(np.clip(dist2, 0.0, None))
    alpha = smoothstep(
        np.zeros_like(edge_distance),
        np.full_like(edge_distance, params.edge_softness),
        edge_distance,
    )

    image = float(params.albedo) * lit
    return image, alpha


def _shadow_mask(
    xx: np.ndarray,
    yy: np.ndarray,
    center_x: float,
    center_y: float,
    radius: float,
    params: RenderParams,
) -> np.ndarray:
    if not params.shadow_enabled or params.shadow_strength <= 0.0:
        return np.zeros_like(xx, dtype=np.float32)

    dx = xx - center_x
    dy_up = center_y - yy
    radial_distance = np.sqrt(np.clip(dx * dx + dy_up * dy_up, 0.0, None))
    edge_distance = radial_distance - radius

    shadow_dir = vec_from_angles(params.shadow_azimuth, params.shadow_elevation)
    shadow_xy = np.array([shadow_dir[0], shadow_dir[1]], dtype=np.float32)
    shadow_xy_norm = np.linalg.norm(shadow_xy)
    if shadow_xy_norm <= 1e-8:
        return np.zeros_like(xx, dtype=np.float32)
    shadow_xy /= shadow_xy_norm

    width = radius * (0.03 + 0.17 * (params.shadow_softness / 2.5)) + params.shadow_distance
    if width <= 1e-8:
        return np.zeros_like(xx, dtype=np.float32)

    ring = (edge_distance >= 0.0) & (edge_distance <= width)
    mask = np.zeros_like(xx, dtype=np.float32)
    if not np.any(ring):
        return mask

    t = np.clip(edge_distance / width, 0.0, 1.0)
    falloff = 1.0 - t

    safe_distance = np.where(radial_distance > 1e-6, radial_distance, 1.0)
    ndx = dx / safe_distance
    ndy = dy_up / safe_distance
    light_dot = ndx * shadow_xy[0] + ndy * shadow_xy[1]
    dir_mod = np.clip(-light_dot, 0.0, 1.0)

    mask[ring] = params.shadow_strength * falloff[ring] * dir_mod[ring]
    return np.clip(mask, 0.0, 1.0)


def render_image(params: RenderParams) -> Image.Image:
    width = int(params.width)
    height = int(params.height)

    yy, xx = np.mgrid[0:height, 0:width]
    background = np.full((height, width), float(params.background), dtype=np.float32)

    cx = width * 0.5 + params.center_x
    cy = height * 0.5 + params.center_y
    top_y = cy - (params.radius + params.vertical_gap * 0.5)
    bottom_y = cy + (params.radius + params.vertical_gap * 0.5)

    if params.shadow_enabled:
        bump_shadow = _shadow_mask(xx, yy, cx, top_y, params.radius, params)
        dent_shadow = _shadow_mask(xx, yy, cx, bottom_y, params.radius, params)
        combined_shadow = np.clip(bump_shadow + dent_shadow, 0.0, 1.0)
        background *= 1.0 - combined_shadow

    top_strength = params.dent_strength if params.top_is_concave else params.bump_strength
    bottom_strength = params.bump_strength if params.top_is_concave else params.dent_strength

    top_disc, top_alpha = _shade_disc(
        xx, yy, cx, top_y, params.radius, concave=params.top_is_concave, strength=top_strength, params=params
    )
    bottom_disc, bottom_alpha = _shade_disc(
        xx, yy, cx, bottom_y, params.radius, concave=not params.top_is_concave, strength=bottom_strength, params=params
    )

    image = background.copy()
    image = image * (1.0 - top_alpha) + top_disc * top_alpha
    image = image * (1.0 - bottom_alpha) + bottom_disc * bottom_alpha
    rgb = np.clip(np.round(image), 0, 255).astype(np.uint8)
    rgb = np.repeat(rgb[..., None], 3, axis=2)
    return Image.fromarray(rgb, mode="RGB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a stacked bump-and-dent stimulus image.")
    parser.add_argument("--output", type=Path, default=Path("stimulus.png"), help="Output PNG path.")
    parser.add_argument("--width", type=int, default=420, help="Image width in pixels.")
    parser.add_argument("--height", type=int, default=520, help="Image height in pixels.")
    parser.add_argument("--background", type=int, default=182, help="Background gray level 0-255.")
    parser.add_argument("--albedo", type=int, default=196, help="Shape gray level 0-255 before shading.")
    parser.add_argument("--radius", type=float, default=96.0, help="Disc radius in pixels.")
    parser.add_argument("--vertical-gap", type=float, default=38.0, help="Gap between bump and dent discs.")
    parser.add_argument(
        "--top-is-concave",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Flip orientation so the top stimulus is concave and the bottom is convex.",
    )
    parser.add_argument("--center-x", type=float, default=0.0, help="Horizontal shift in pixels.")
    parser.add_argument("--center-y", type=float, default=0.0, help="Vertical shift in pixels.")
    parser.add_argument("--light-x", type=float, default=0.0, help="Horizontal light position.")
    parser.add_argument("--light-y", type=float, default=0.75, help="Vertical light position, from -1 bottom to 1 top.")
    parser.add_argument("--light-z", type=float, default=1.1, help="Forward light distance.")
    parser.add_argument("--ambient", type=float, default=0.28, help="Ambient contribution.")
    parser.add_argument("--diffuse", type=float, default=0.82, help="Diffuse contribution.")
    parser.add_argument("--specular", type=float, default=0.18, help="Specular contribution.")
    parser.add_argument("--shininess", type=float, default=20.0, help="Specular sharpness.")
    parser.add_argument("--edge-softness", type=float, default=1.8, help="Edge antialias width.")
    parser.add_argument(
        "--cosine-falloff",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable cosine height-map falloff for smoother edge blending.",
    )
    parser.add_argument(
        "--flat-profile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a flatter, more embedded-looking relief profile.",
    )
    parser.add_argument(
        "--shadow",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable directional contact shadows around each stimulus.",
    )
    parser.add_argument("--shadow-azimuth", type=float, default=220.0, help="Shadow direction azimuth in degrees.")
    parser.add_argument("--shadow-elevation", type=float, default=30.0, help="Shadow direction elevation in degrees.")
    parser.add_argument("--shadow-strength", type=float, default=0.45, help="Shadow darkening strength.")
    parser.add_argument("--shadow-softness", type=float, default=0.9, help="Shadow edge softness.")
    parser.add_argument("--shadow-distance", type=float, default=45.0, help="Shadow spread distance in pixels.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = RenderParams(
        width=max(32, args.width),
        height=max(32, args.height),
        background=int(clamp(args.background, 0, 255)),
        albedo=int(clamp(args.albedo, 0, 255)),
        radius=max(8.0, args.radius),
        center_x=args.center_x,
        center_y=args.center_y,
        vertical_gap=max(-args.radius * 1.5, args.vertical_gap),
        top_is_concave=args.top_is_concave,
        light_x=args.light_x,
        light_y=clamp(args.light_y, -1.0, 1.0),
        light_z=max(0.05, args.light_z),
        ambient=max(0.0, args.ambient),
        diffuse=max(0.0, args.diffuse),
        specular=max(0.0, args.specular),
        shininess=max(1.0, args.shininess),
        edge_softness=max(0.5, args.edge_softness),
        use_cosine_falloff=args.cosine_falloff,
        use_flat_profile=args.flat_profile,
        shadow_enabled=args.shadow,
        shadow_azimuth=args.shadow_azimuth % 360.0,
        shadow_elevation=clamp(args.shadow_elevation, 1.0, 89.0),
        shadow_strength=clamp(args.shadow_strength, 0.0, 1.0),
        shadow_softness=max(0.1, args.shadow_softness),
        shadow_distance=max(0.0, args.shadow_distance),
    )
    image = render_image(params)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(f"Saved stimulus to {args.output}")


if __name__ == "__main__":
    main()
