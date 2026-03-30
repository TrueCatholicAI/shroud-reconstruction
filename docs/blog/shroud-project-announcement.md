# I Built an AI Pipeline to Study the Shroud of Turin. Here's What the Data Shows.

I'm a Catholic. I'll say that upfront, because transparency matters more than pretending I have no priors.

I believe the Shroud of Turin is the authentic burial cloth of Jesus Christ. That conviction is what motivated me to spend months building a computer vision pipeline to extract and analyze the three-dimensional information encoded in the Shroud's image. But conviction is not what drives the pipeline. The code doesn't know what I believe. It runs the same math regardless.

Here's what I built, what the data shows, and what it doesn't claim.

---

## The VP-8 Discovery That Started Everything

In 1976, physicists John Jackson, Eric Jumper, and Bill Mottern placed a photograph of the Shroud under a VP-8 Image Analyzer - a device that converts image brightness to surface height. Normally, photographs produce distorted, meaningless terrain when processed this way. The Shroud produced a coherent three-dimensional face.

That single observation has never been adequately explained by any forgery hypothesis. Brightness on the Shroud's image correlates linearly with the distance between the cloth and the body it wrapped. No painting, photograph, or rubbing technique produces this property.

I wanted to take that discovery further using modern tools.

## What the Pipeline Does

The pipeline processes two independent source photographs of the Shroud - Giuseppe Enrie's 1931 image and Vernon Miller's 1978 STURP photograph - through four stages:

1. **Depth extraction** - Converting brightness to a 3D surface using the VP-8 principle
2. **Landmark detection** - Finding anatomical features (eyes, nose, mouth, jaw) on the depth map
3. **Measurement** - Computing facial dimensions in real centimeters using scale calibration
4. **Reconstruction** - Generating sculptural renders constrained by the extracted geometry

Every step is documented. Every parameter is recorded. Every script is open source.

## The Key Findings

**The face is anatomically correct.** Across nine anthropometric measurements, 78% fall within published ranges for adult males. The face has a symmetry score of 0.989. This is not a cartoon or a crude image - it's a face with correct proportions at millimeter-scale precision.

**Two independent photographs agree.** The Enrie 1931 and Miller 1978 sources - taken 47 years apart with completely different cameras and film - produce measurements that agree. Seven of nine measurements from each study fall in normal ranges. If this were an artifact of one particular photograph, the second study would fail.

**Modern AI cannot decode the signal.** I ran two state-of-the-art neural depth estimation models - MiDaS DPT-Large and Depth Anything V2 - on the same source images. These models are trained on millions of photographs to recognize depth from shading, perspective, and texture. Neither one could recover the Shroud's 3D information. The VP-8 brightness-to-distance assumption is *required* to decode the signal. The encoding is unlike anything in modern AI training data.

**The image formation follows a strict linear function.** When we plot image intensity against cloth-to-body distance, the relationship is perfectly linear (R-squared = 1.000). Not exponential. Not inverse-square. Not power-law. This rules out most radiation-based hypotheses and strongly constrains whatever mechanism created the image.

**The full body encodes consistent information.** Height estimation from the full-body depth extraction yields 1.76 m (5'9") with proper anatomical proportions. The hands are crossed right over left with thumbs not visible - consistent with post-mortem adduction, a detail a medieval forger would have no reason to include.

## What the Data Does NOT Claim

This is important. The pipeline does not:

- Prove the Shroud is authentic
- Identify the man depicted
- Assert that the AI reconstructions are portraits of Jesus
- Make any theological argument

The reconstructions are rendered in clay, sandstone, and marble specifically to remove skin color from the output. The Shroud does not encode pigmentation. Any skin tone applied to a reconstruction would be an assumption, not a finding.

I also publish every negative and inconclusive result. The scourge mark detection didn't work well. The coin-over-eyes investigation was inconclusive. Temporal degradation analysis between 1931 and 1978 couldn't separate real changes from imaging differences. Knowing what the data *cannot* tell us is as important as knowing what it can.

## Why This Matters to Me as a Catholic

The Church has never declared the Shroud authentic. It doesn't need to. The faith doesn't depend on a piece of linen.

But I think the Shroud deserves serious, honest investigation with modern tools. Not breathless claims. Not dismissive debunking. Rigorous, reproducible analysis that lets the data speak.

If the Shroud is what I believe it is, then it can withstand any amount of scrutiny. If I'm wrong, I want to know. The worst thing a believer can do is be afraid of evidence.

Every line of code, every parameter choice, every finding - positive or negative - is published openly. Anyone can download the scripts, run them on the same source images, and verify every result. That's not faith. That's science. And I don't think the two are enemies.

## See the Full Research

The complete project with all findings, interactive 3D viewer, and downloadable data:

**[Shroud of Turin AI Forensic Reconstruction](https://truecatholicai.github.io/shroud-reconstruction/)**

Source code: **[GitHub Repository](https://github.com/TrueCatholicAI/shroud-reconstruction)**

---

*Built with Python, OpenCV, Stable Diffusion, and an RTX 2080 Ti. Published under open-source principles because truth doesn't need a paywall.*

*TrueCatholic AI - 2026*
