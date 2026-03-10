In run_trial.py, write code to create a gui that takes a user through an experiment using the results of sphere_renderer_gui. I want to create a new gui in run_trial.py that launches the experiement tells the user what to do and allows the user to input their response. keep the instructions to the user very brief. The plan below is the rough sketch of the experiment. Ignore the idea about different lights (point lights etc, and just use the lights). 

Research Questions
How does the visual system integrate different shading cues to infer light direction? If there are conflicted cues about lighting direction, does the visual system:
reliability-weight them (like the vector-sum / Bayesian cue combination used in the paper)
Give one type priority over others (e.g. cast shadows always win)
Do lighting conditions influence cue integration by strengthening or weakening some cues?

Hypothesis
For different lighting conditions:
Under diffuse lighting conditions, with weaker cues and more ambiguous shadows, the visual system will reliability-weight shading cues.
Under a point light, where there are strong cues, the visual system will rely primarily on the most salient cue, cast shadows.

For different lighting directions:
Top-down will heavily bias diffuse shading
Side-side will bias cast shadows

Stimuli
2 lighting conditions:
Light from all around (minimal cues)
Single point light (very strong cues)

Diffuse shading (signal convex or concave)
Cast shadows (signal convex or concave)
Specular highlights (signal convex or concave) → unsure if we will use
Light direction varies (top-down, side-side)

Congruent Stimuli
For each, we need two lights:
Diffuse light (general light pointing toward stimulus, like sun or something)
Point light

Do all of these for convex and concave (16 total stimuli)
Case 1 (weaker cues): low diffuse light, strong point light
Point light from above
Point light from below
Point light from left
Point left from right

Case 2 (stronger cues): medium diffuse light, medium point light 
Point light from above
Point light from below
Point light from left
Point left from right

Methods
Baseline (control) study where all cues agree and two lighting conditions are tested → determine baseline convex/concave bias
Present conflicting stimuli
Ask participants to rate on scale
1 - always see convex
2 - primarily see convex, less confident, switch a little
3 - see both equally
4 - primarily see concave, less confident, switch a little
5 - always see concave

Analysis
