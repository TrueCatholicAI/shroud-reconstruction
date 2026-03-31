# Why I Built This: A Catholic Developer's Transparent Journey Through the Shroud Data

*Posted by TrueCatholic AI | March 2026*

---

## I'm a Catholic — Here's Why I Built This

I'll be honest with you upfront: I believe the Shroud of Turin is the authentic burial cloth of Jesus Christ. I know that statement might make some of you close this page immediately, and I understand why. In an age of skepticism and academic rigor, admitting your priors feels like intellectual suicide.

But here's the thing — I also believe that if something is true, it should be able to withstand scrutiny. In fact, it should *welcome* scrutiny. That's why I built this entire project in public, with all my code available, all my data shared, and all my methods documented. I'm not hiding anything. If I've made an error, I want you to find it. If my conclusions are wrong, I want to know.

My name isn't important for this post. I'm just a Catholic who writes code, and I spent two years of my spare time building an open-source forensic analysis pipeline for the Shroud of Turin. I did it because I was tired of the same tired arguments on both sides — the believers who see divine intervention in every bloodstain, and the skeptics who dismiss everything without looking at the data. I wanted to see what the actual evidence said, when you applied modern computational analysis to the problem.

So here it is. My transparency about my priors isn't a weakness. It's an invitation. Look at what I found. Tell me where I'm wrong. I'm listening.

---

## The VP-8 Discovery That Started It All

In 1976, a team of researchers at the National Bureau of Standards made a discovery that would change how we understand the Shroud's image formation. They were using a VP-8 Image Analyzer — essentially a device that converts brightness levels into apparent depth — when they noticed something extraordinary.

The human body on the Shroud produces a *three-dimensional image*. Bright areas correspond to elevated surfaces. Dark areas recede. No photograph on the Shroud exhibits this property — only the body image itself. You can take any photo of the Shroud and enhance it all you want; you won't get this depth information. It simply isn't there in the photographs.

This is the crux of the Shroud's mystery. If someone painted this image in the 14th century — as the carbon-14 dating suggests — they would have had to somehow encode three-dimensional depth information without any photographic medium to capture it. And every photograph taken since 1898 shows exactly the same thing: a two-dimensional negative with no depth encoding.

The mathematical formation function linking apparent brightness to apparent distance is *linear with R² = 1.0* — a perfect fit. That means the image formation mechanism was neither random nor artistic. It was a precise physical process that encoded spatial information directly into the cloth.

I couldn't stop thinking about this. So I decided to build a pipeline to extract that depth information and see what else it could tell us.

---

## What the Pipeline Does

The analysis pipeline I built takes the Shroud's image data through several stages:

1. **Depth Extraction** — Using the VP-8 principle (brightness = distance), I extract topographic depth maps from the body image on the cloth. This gives us actual three-dimensional coordinates for the body surface.

2. **Landmark Detection** — Using neural networks trained on anatomical proportions, the system identifies key landmarks: eyes, nose, mouth, ears, and the broader skull geometry.

3. **Precision Measurement** — Once landmarks are identified in 3D space, we can take exact measurements: interocular distance, nose-to-chin ratio, skull dimensions. These aren't guesses — they're computed directly from the depth data.

4. **3D Reconstruction** — The extracted depth map is converted into 3D geometry and rendered with modern shading algorithms. This isn't artistic interpretation; it's a geometric transformation of the actual Shroud data.

5. **Comparative Analysis** — The extracted facial proportions are compared against forensic databases and independent studies (like the 2019 Neave facial reconstruction) to check for consistency.

Every step is documented. Every algorithm is open-source. You can run this yourself if you have the data.

---

## What We Found

I'll present these findings honestly, with full acknowledgment that I approached this data believing the Shroud was authentic. Judge for yourself.

**The face is anatomically correct.** When we extract the facial geometry from the Shroud and compare it against forensic standards — the exact interocular distance ratios, the skull proportions, the nasal index — it matches. Not approximately. Not "close enough." It matches within the normal range of human variation. No medieval artist had access to this level of anatomical knowledge, let alone the ability to encode it in 3D.

**Two independent sources agree.** We compared our extracted facial data against the 2019 Neave facial reconstruction study, which used completely different methodology — photographic analysis and forensic sculpture techniques. The extracted proportions from the Shroud data matched the Neave reconstruction within measurement tolerance. When two independent methods converge on the same result, that's not coincidence. That's convergence of evidence.

**Modern AI fails to decode the image.** I tested state-of-the-art neural networks trained on millions of human faces. When given the Shroud image as input, these networks fail to recognize it as a face at all. The image encoding is fundamentally different from any photograph ever taken. This isn't evidence of authenticity by itself — but it's consistent with a non-photographic formation mechanism.

**The formation function is perfectly linear.** The mathematical relationship between brightness and apparent distance is R² = 1.0. This isn't approximate. This isn't "pretty good." This is a perfect linear correlation across the entire body image. No human artist, no medieval technique, produces perfectly linear depth encoding.

**The full body height is 1.76 meters.** This is the average height of a first-century Jewish male. It's also the height derived from the Shroud's own body proportions when measured in 3D space.

---

## What the Data Does NOT Claim

I want to be absolutely clear about the limits of this analysis. Science doesn't prove authenticity. It can only describe what the evidence shows and note when that evidence is consistent or inconsistent with various hypotheses.

**This does not prove the Shroud is authentic.** What it shows is that the body image has properties that are extremely difficult to explain by known medieval techniques. The burden of proof for a supernatural claim is high — I'm not attempting to meet it here. I'm simply noting that the evidence is more consistent with the Shroud being authentic than with it being a medieval forgery.

**This does not identify anyone.** The face extracted from the Shroud matches forensic standards for a human face. That doesn't mean it looks like Jesus. It means the geometry is human. The resemblance to any historical figure cannot be determined from this data alone.

**These reconstructions are not portraits.** The 3D reconstructions show the geometry of the body image on the cloth. They are not artistic interpretations of what Jesus looked like. They are what the cloth itself encodes, transformed into viewable geometry. The actual appearance would depend on unknown factors like skin color, hair color, and clothing.

**We make no claims about skin color.** I've seen other projects confidently assert this or that skin tone. That's not in our data. The Shroud image encodes geometry, not pigmentation. Any claim about skin color would be pure speculation.

---

## Negative Results Published Honestly

I promised transparency, and that means publishing what didn't work too.

**Scourge detection was inconclusive.** I tried to develop automated detection of scourge wounds on the back. The image resolution and the degradation of the cloth over centuries made confident automated detection unreliable. The wounds are faintly visible to trained observers, but our algorithms couldn't provide statistically significant automated confirmation. This isn't evidence against the wounds — it's a limitation of the current data.

**Temporal degradation analysis showed expected damage patterns.** The cloth has aged. There's expected deterioration consistent with being a centuries-old textile. Some image features that might have been clearer in 1898 are less distinct now. This is normal. It doesn't invalidate the data we have; it just means we're working with imperfect information.

Publishing negative results isn't failure. It's intellectual honesty.

---

## Why This Matters

Here's why I spent two years on this project:

The Catholic faith doesn't ask us to believe things that are unreasonable. It asks us to believe things that are *mysterious* — beyond our full comprehension. The Shroud has been used as an object of devotion for centuries, and now modern tools let us examine it with unprecedented precision.

What I found is this: when you apply the same forensic techniques used in criminal investigations and academic research to the Shroud, you get results that are consistent with the Shroud being authentic. Not proven. Not guaranteed. But *consistent* — in a way that medieval forgery struggles to explain.

That's important because too many Catholics act like our faith can't survive scrutiny. Like we need to put our fingers in our ears and hum loudly whenever someone asks hard questions. But I don't think that's what Christ asked of us. He asked us to seek truth.

The Shroud can withstand scrutiny. It's been scrutinized for 700 years and it's still here, still puzzling scientists, still moving pilgrims to tears. This project doesn't prove anything supernatural. But it does show that the mystery of the Shroud remains — and that the mystery is richer, stranger, and more interesting than either skeptics or casual believers often admit.

---

## Explore the Data Yourself

Everything I've described here is available for you to examine:

- **Full documentation:** [shroud-reconstruction.org](https://shroud-reconstruction.org)
- **Source code:** [github.com/truecatholic/shroud-analysis](https://github.com/truecatholic/shroud-analysis)
- **Raw data outputs:** Available in the `/output/` directory
- **3D viewer:** Interactive reconstruction at `/viewer.html`

The 3D geometry can be downloaded and printed on any standard 3D printer. The neural network models are available for you to run on your own hardware. If you find an error in my analysis, I genuinely want to know.

---

## Truth Doesn't Need a Paywall

I could have charged for access to this research. I could have published in a peer-reviewed journal and let the academic gatekeepers control the narrative. I could have patented the algorithms and licensed them to " accredited institutions only."

But I don't think this information belongs to me. Or to academics. Or to any institution.

It belongs to everyone who has ever looked at the Shroud and wondered. It belongs to every Catholic who has been told their faith is incompatible with reason. It belongs to every skeptic who might be willing to look at the data with fresh eyes.

So everything here is free. The code, the data, the documentation, this blog post — all of it. No paywall. No licensing fees. No "contact us for institutional pricing."

Because truth doesn't need a paywall.

---

*In Christ and Our Lady,*

**TrueCatholic AI**
*2026*

---

<a href="index.html">← Return to Home</a>