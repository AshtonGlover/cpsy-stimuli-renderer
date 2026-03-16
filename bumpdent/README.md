# BumpDent Stimulus Renderer

This folder now contains an interactive Tkinter tool that matches the pattern used by the existing renderer in the repository.

The preview shows:

- a bump stimulus on top
- a dent stimulus below it

Both share the same lighting model, and `Light Y` moves the illumination from the bottom of the scene (`-1`) to the top (`1`).
Directional shadows can be enabled independently and edited in a separate shadow-controls window.
Every slider also has a typed numeric field, and the lighting panel includes a preset for an ambiguity-neutral render.

## Launch the GUI

```bash
python3 bumpdent_gui.py
```

## Export from the command line

```bash
python3 render_stimuli.py --output stimulus.png --light-y 1
```

## Main controls

- `Light Y`: vertical light position from bottom to top
- `Light X`: horizontal light offset
- `Set Ambiguous`: zeros directional shading so convexity / concavity cues become neutral
- `Gap`: spacing or overlap between the bump and dent
- `Radius`: size of each stimulus
- `Bump` / `Dent`: curvature strength for each shape
- `Flat embedded profile`: toggles a flatter relief so the shape looks less ball-like
- `Directional shadows`: enables a separate directional contact-shadow pass
- `Shadow Controls`: opens a second window with azimuth, elevation, strength, softness, and spread controls
- `Background` / `Albedo`: gray scene styling

The renderer code is shared between the GUI and CLI, so exported images match the live preview.
