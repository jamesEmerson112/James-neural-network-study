# EM in Medicine

> Companion note to `02_em_in_medicine_video.mp4`. See [the overview](00_overview.md). It follows [Logs, EM and Gradient Descent](01_logs_em_and_gradient_descent.md), which introduces the E-step, the M-step and the responsibilities $r_{ik}$.

The video is a Manim explainer at 1080p30 about three places where medicine uses the expectation-maximization (EM) algorithm. In MRI tissue segmentation, EM labels the voxels of an image that the scanner has already formed. In PET reconstruction, EM can form the image itself from detector counts. In combining several doctors' diagnoses, EM estimates each patient's true diagnosis and each doctor's reliability at the same time. Every example runs real EM iterations in NumPy on synthetic data drawn with a fixed seed, and every number on screen comes from the script's `precompute()`. The script asserts the claims the captions make. Examples are that the log-likelihood never decreases in any of the three examples, and that rounding never changes a sum or ratio shown on screen.

Render it from this folder with Manim Community, run through `uv` so nothing needs installing first (no LaTeX needed):

```
uvx --from manim manim --resolution 1920,1080 --frame_rate 30 02_em_in_medicine_video.py EMInMedicine
```

With Manim Community already installed, plain `manim` in place of `uvx --from manim manim` does the same. The render writes into a `media/` folder next to the script; that folder is output only and should not be committed. Setting `EMM_ONLY=pet,doctors` renders only the named sections (`intro`, `mri`, `pet`, `doctors`, `summary`), which is useful when changing one part. Every section starts and ends on an empty frame, so sections rendered separately can be joined with ffmpeg's concat demuxer.

## Timeline

The video runs 8 minutes 4 seconds. The times below are the start of each part, as the script prints them while it renders.

| Time | Part |
|---|---|
| 0:00 | Title, and what EM needs: measured numbers, a hidden quantity and unknown settings |
| 0:32 | The three examples, and the difference between labelling an image and forming one |
| 0:49 | MRI: the scanner forms the image; what is measured, hidden and unknown; the E-step and M-step |
| 1:30 | MRI: EM fits three tissue curves to the brightness histogram |
| 2:00 | MRI: each voxel coloured by its largest share, and Wells and colleagues (1996) |
| 2:21 | PET: EM builds the image itself |
| 2:33 | PET: positrons, gamma-ray pairs and detector lines; what is measured, hidden and unknown |
| 3:05 | PET: the system matrix $a_{ij}$, the E-step on one detector line, the M-step, and Shepp and Vardi (1982) |
| 4:27 | PET: ML-EM iterations beside the true image, with the log-likelihood |
| 4:58 | PET: the iteration gallery, the error against the truth, noise, and OS-EM (Hudson and Larkin, 1994) |
| 5:39 | Doctors: three doctors' calls on 100 synthetic patients; what is measured, hidden and unknown |
| 6:09 | Doctors: the E-step for one patient and the M-step |
| 6:38 | Doctors: the estimates converge |
| 7:08 | Doctors: the hidden truth, with EM against the majority vote |
| 7:30 | Summary |

## Symbols

Video 01 used one letter for several things, so this video gives each symbol one meaning and says it in words the first time it appears.

| Symbol | Meaning | Example |
|---|---|---|
| $v$ | a voxel (volume element), here one pixel of the MRI slice | MRI |
| $t$, $u$ | a tissue: fluid, gray matter or white matter | MRI |
| $x_v$ | the measured brightness of voxel $v$ | MRI |
| $z_v$ | the hidden tissue of voxel $v$ | MRI |
| $\pi_t$, $\mu_t$, $\sigma_t$ | tissue $t$'s fraction of the voxels, mean brightness and spread | MRI |
| $r_{vt}$ | voxel $v$'s share for tissue $t$, from the E-step | MRI |
| $V$ | the number of brain voxels, 2,116 | MRI |
| $\mathcal{N}$ | the bell-curve (Gaussian) density | MRI |
| $i$ | a detector line | PET |
| $j$, $k$ | a pixel of the PET image | PET |
| $y_i$ | the measured number of counts on line $i$ | PET |
| $\lambda_j$ | the activity of pixel $j$: its expected number of emissions during the scan | PET |
| $a_{ij}$ | the chance that an emission from pixel $j$ is counted on line $i$ | PET |
| $s_j$ | pixel $j$'s coverage, $\sum_i a_{ij}$: the chance that any line counts its emission | PET |
| $\bar y_i$ | line $i$'s predicted count under the current image, $\sum_k a_{ik}\lambda_k$ | PET |
| $n_{ij}$ | the hidden number of line $i$'s counts that came from pixel $j$ (note only) | PET |
| $p$ | a patient | doctors |
| $d$ | a doctor: A, B or C | doctors |
| $c_{pd}$ | doctor $d$'s call on patient $p$, 1 for yes and 0 for no | doctors |
| $z_p$ | the hidden truth, 1 when patient $p$ has the condition | doctors |
| $\pi$ | the prevalence: the fraction of patients who have the condition | doctors |
| $\mathrm{se}_d$, $\mathrm{sp}_d$ | doctor $d$'s sensitivity and specificity | doctors |
| $q_p$ | the chance that patient $p$ has the condition, from the E-step | doctors |
| $P$ | the number of patients, 100 | doctors |

The letter $\pi$ means the same thing in both places where it appears: the fraction of the population in a class. The letter $z$ likewise always names the hidden quantity, the tissue $z_v$ in MRI and the true diagnosis $z_p$ for the doctors. The video calls $s_j$ the pixel's coverage, because "sensitivity" already names a doctor's true-positive rate in the third example.

## What EM needs

EM fits a model when part of the data was never recorded. It needs three pieces. The first is the measured numbers. The second is a hidden quantity which, if it were known, would make the fit easy. The third is the unknown settings to estimate. Each iteration has two steps. The E-step uses the current settings to split each measurement among the hidden possibilities. The M-step refits the settings as if those splits were the truth. The abstract of Dempster, Laird and Rubin (1977) states that they derive "the monotone behaviour of the likelihood": no EM iteration lowers the likelihood of the measured data.

## MRI: EM labels an image that already exists

An MRI scanner does not use EM to form its image. The scanner's signal is proportional to the Fourier transform of the magnetization in the body, sampled at many points, and the image is reconstructed by an inverse Fourier transform (Larson's MRI reconstruction notes). EM's role comes afterward, in segmentation: deciding which tissue each voxel shows.

The script builds a synthetic $64 \times 64$ slice. It is an ellipse with an outer band of fluid, a folded band of gray matter, a white-matter core and two fluid-filled ventricles. Its 2,116 brain voxels are 388 fluid, 828 gray matter and 900 white matter. Each voxel's brightness is drawn from $\mathcal{N}(40, 9^2)$, $\mathcal{N}(100, 13^2)$ or $\mathcal{N}(150, 10^2)$ by tissue, with NumPy's `default_rng(20261122)`.

- **Measured:** $x_v$, each voxel's brightness.
- **Hidden:** $z_v$, each voxel's tissue.
- **Unknown:** $\pi_t$, $\mu_t$ and $\sigma_t$ for the three tissues.

The model is a three-curve Gaussian mixture, so the steps are those of video 01 with three curves:

$$r_{vt} = \frac{\pi_t\,\mathcal{N}(x_v \mid \mu_t, \sigma_t^2)}{\sum_u \pi_u\,\mathcal{N}(x_v \mid \mu_u, \sigma_u^2)}, \qquad \pi_t = \frac{1}{V}\sum_v r_{vt}, \quad \mu_t = \frac{\sum_v r_{vt}\,x_v}{\sum_v r_{vt}}, \quad \sigma_t^2 = \frac{\sum_v r_{vt}\,(x_v - \mu_t)^2}{\sum_v r_{vt}},$$

where $V = 2{,}116$ is the number of brain voxels. The screen writes the bell curve as $\mathcal{N}(x_v \mid \mu_t, \sigma_t)$ with the spread $\sigma_t$, while the formulas here write its variance $\sigma_t^2$, as note 01 does. EM starts from a poor guess, $\mu = (70, 85, 105)$, $\sigma = (30, 30, 30)$ and equal fractions. The mean log-likelihood per voxel rises from $-5.457$ to $-5.111$ after one iteration, $-4.924$ after 20 and $-4.811$ after 100. By then the fitted means are $39.3$, $100.2$ and $149.7$, close to the $40$, $100$ and $150$ that drew the data. Colouring each voxel by its largest share $r_{vt}$ gives a label map that agrees with the true synthetic map on 98.6% of the brain voxels.

Real segmentation methods add more to the model. Wells, Grimson, Kikinis and Jolesz (1996) used EM to segment brain MRI while also handling brightness that varies across the image and between scans (intensity inhomogeneity), which a plain mixture ignores. Their abstract reports a study of more than 1,000 brain scans.

## PET: EM builds the image itself

In PET, a tracer releases positrons. Each positron annihilates with an electron, and the pair becomes two 511 keV gamma rays that leave at 180 degrees to each other. When two detectors in the ring fire together, the annihilation lies on the line between them (the "Let's Play PET" tutorial and NIBIB's nuclear medicine page). The scanner records how many such events land on each detector line. There is no image until an algorithm turns those counts into one, and ML-EM (maximum-likelihood EM) is one such algorithm.

Shepp and Vardi (1982) modelled this for emission tomography. Each pixel $j$ emits a Poisson number of events with unknown mean $\lambda_j$, and the count on each line is then Poisson too. They estimated the image by maximum likelihood with the EM algorithm. Lange and Carson (1984) then derived EM reconstruction algorithms for both emission and transmission tomography.

- **Measured:** $y_i$, the counts on each detector line $i$.
- **Hidden:** which pixel each counted emission came from.
- **Unknown:** $\lambda_j$, each pixel's activity.

### The synthetic scan

The phantom is $16 \times 16$ pixels: an elliptical body of activity $120$ with hot spots of $640$, $440$ and $280$. The detector lines are parallel strips one pixel wide, 16 of them at each of 18 angles spread over 180 degrees, so there are 288 lines. The script measures $a_{ij}$ by splitting each pixel into $8 \times 8$ points and counting the fraction that falls inside strip $i$. An emission's line lies at each angle with chance $1/18$, so

$$a_{ij} = \frac{\text{fraction of pixel } j \text{ inside line } i}{18}.$$

For the line examined in the video, a pixel with $0.75$ of its area inside has $a_{ij} = 0.75/18 = 0.0417$. Pixels inside the body are covered at every angle, so their coverage is $s_j = \sum_i a_{ij} = 1$. Corner pixels fall outside some lines and have coverage as low as $0.56$. The counts are drawn as $y_i \sim \mathrm{Poisson}\big(\sum_j a_{ij}\lambda_j\big)$ with `default_rng(7)`, for 22,291 counts in all.

### The two steps

If we knew $n_{ij}$, the number of counts on line $i$ that came from pixel $j$, the fit would be easy: each $n_{ij}$ is Poisson with mean $a_{ij}\lambda_j$, so the best $\lambda_j$ would be $\sum_i n_{ij} / \sum_i a_{ij}$. The $n_{ij}$ are hidden, so EM estimates them.

- **E-step.** Given the line's total $y_i$, the expected share from pixel $j$ is proportional to $a_{ij}\lambda_j$:
  $$\mathbb{E}[n_{ij} \mid y_i] = y_i\,\frac{a_{ij}\lambda_j}{\sum_k a_{ik}\lambda_k}.$$
  The denominator is the line's predicted count under the current image. For line 53 in the script, which recorded 175 counts, a flat starting image gives a hot-spot pixel $7.7$ counts and a body pixel $8.0$, because the split follows the overlaps alone. After one iteration the image is brighter where the counts pile up, and the same line gives the hot-spot pixel $13.5$ counts and the body pixel $7.3$.
- **M-step.** Each pixel's new activity is the total count assigned to it, divided by its coverage:
  $$\lambda_j \leftarrow \frac{\lambda_j}{s_j}\sum_i \frac{a_{ij}\,y_i}{\sum_k a_{ik}\lambda_k}, \qquad s_j = \sum_i a_{ij}.$$

This update keeps every $\lambda_j$ non-negative, since it multiplies the old value by a non-negative factor.

### What the iterations show

EM starts from a flat image whose predicted total count equals the measured total. The Poisson log-likelihood of the counts, $\sum_i \big(y_i \log \bar y_i - \bar y_i - \log y_i!\big)$ with $\bar y_i = \sum_j a_{ij}\lambda_j$, includes the constant $\log y_i!$ term. It climbs as follows, and the script asserts that it rises at every one of the 200 iterations it computes. The error column is the root-mean-square (RMS) difference between EM's image and the true image: the square root of the average squared per-pixel difference.

| Iteration | Log-likelihood | RMS error against the true image |
|---|---|---|
| 0 | $-7{,}355.6$ | $149.5$ |
| 1 | $-4{,}947.0$ | $123.3$ |
| 2 | $-3{,}200.8$ | $97.7$ |
| 5 | $-1{,}328.4$ | $59.5$ |
| 20 | $-829.6$ | $43.2$ |
| 100 | $-796.4$ | $48.3$ |

The error is smallest at iteration 28, at $42.4$, and then grows slowly while the likelihood keeps rising. The script reruns EM on noise-free counts, $y_i = \bar y_i$ under the true image, and there the error falls at every iteration up to 100. So the late rise in error comes from EM fitting the Poisson noise in this one scan.

### Speeding it up

Each ML-EM iteration passes over every detector line before it updates the image once. Hudson and Larkin (1994) proposed ordered subsets EM (OS-EM). It splits the lines into an ordered sequence of subsets and updates the image after each subset, so one pass through the data makes many updates. Their abstract reports that in SPECT simulations OS-EM gave about an order-of-magnitude speedup over EM with the same restoration quality, and that the method applies to PET as well. SPECT is single photon emission computed tomography, a related nuclear medicine scan whose detectors pick up gamma rays from an injected tracer (NIBIB's nuclear medicine page). The video does not claim that OS-EM is the clinical standard, because no source checked for this note says so.

## Combining doctors' calls: Dawid and Skene (1979)

Dawid and Skene (1979) started from patient records, in which many items are subject to errors of measurement. Their model estimates each observer's error rates even when the patient's true answer is never available, and they fit it with EM, which they called a "slow but sure" way to the maximum-likelihood estimates. Their model handles questions with several categories. The video uses the two-category case, where a doctor's error rates reduce to a sensitivity and a specificity.

The script draws 100 synthetic patients with `default_rng(2)`. Each has the condition with chance $0.35$. Doctors A, B and C say yes to a patient who has it with chances $0.95$, $0.70$ and $0.60$, and say no to one who does not with chances $0.95$, $0.70$ and $0.65$. Given the truth, the doctors' calls are independent of each other, which is the model's main assumption.

The seed was chosen from a survey of seeds 0 to 29, which the script reruns in `doc_seed_survey()`. Seed 2 keeps every estimate away from 0 and 1, and on it EM beats the majority vote. Across all 30 seeds, EM gets more patients right than the majority vote on 21, ties on 4 and does worse on 5, so the gain shown is typical but not guaranteed.

- **Measured:** $c_{pd}$, each doctor's yes or no on each patient.
- **Hidden:** $z_p$, each patient's true diagnosis.
- **Unknown:** $\pi$, $\mathrm{se}_d$ and $\mathrm{sp}_d$.

**E-step.** Multiply the chances of patient $p$'s three calls under each answer:

$$q_p = \frac{\pi \prod_d \mathrm{se}_d^{\,c_{pd}}(1-\mathrm{se}_d)^{1-c_{pd}}}{\pi \prod_d \mathrm{se}_d^{\,c_{pd}}(1-\mathrm{se}_d)^{1-c_{pd}} + (1-\pi)\prod_d (1-\mathrm{sp}_d)^{c_{pd}}\,\mathrm{sp}_d^{\,1-c_{pd}}}.$$

**M-step.** Score each doctor against the soft answers:

$$\pi = \frac{1}{P}\sum_p q_p, \qquad \mathrm{se}_d = \frac{\sum_p q_p\,c_{pd}}{\sum_p q_p}, \qquad \mathrm{sp}_d = \frac{\sum_p (1-q_p)(1-c_{pd})}{\sum_p (1-q_p)},$$

where $P = 100$ is the number of patients.

The script starts from the settings fitted to the majority vote. It also asserts that every doctor ends better than chance, with sensitivity plus specificity above 1, which rules out the mirror-image solution in which every label is flipped. For patient 2, whom doctor A calls yes and doctors B and C call no, the first E-step gives

$$\pi\,\mathrm{se}_A(1-\mathrm{se}_B)(1-\mathrm{se}_C) = 0.0150, \qquad (1-\pi)(1-\mathrm{sp}_A)\,\mathrm{sp}_B\,\mathrm{sp}_C = 0.0419,$$

so $q_2 = 0.0150 / (0.0150 + 0.0419) = 0.26$.

The estimates move as follows. The "true" column is computed from the hidden diagnoses of these 100 patients, which is possible only because the data are synthetic.

| | Iteration 0 | Iteration 20 | Iteration 500 | True |
|---|---|---|---|---|
| $\mathrm{se}_A$ | 0.79 | 0.81 | 0.89 | 0.94 |
| $\mathrm{se}_B$ | 0.79 | 0.67 | 0.66 | 0.61 |
| $\mathrm{se}_C$ | 0.76 | 0.62 | 0.62 | 0.58 |
| $\mathrm{sp}_A$ | 0.89 | 0.88 | 0.90 | 0.95 |
| $\mathrm{sp}_B$ | 0.79 | 0.71 | 0.69 | 0.67 |
| $\mathrm{sp}_C$ | 0.76 | 0.66 | 0.65 | 0.64 |
| $\pi$ | 0.38 | 0.37 | 0.34 | 0.36 |

By iteration 177 no estimate moves by more than $10^{-4}$ per iteration. The log-likelihood of the calls rises at every iteration. Without any answer key, EM ranks doctor A as the most reliable, and every sensitivity and specificity ends closer to its true value than where it started; the script asserts both. Its final answers, $q_p > 0.5$, are right for 95 of the 100 patients, while the majority vote is right for 84. Patient 2 is one of the gains: the majority said no, the patient has the condition, and EM ends at $q_2 = 0.57$. With 100 patients the final estimates differ from the true rates by up to $0.06$, so they carry sampling error of that size.

## Summary

| Example | Measured | Hidden | The E-step splits | The M-step refits |
|---|---|---|---|---|
| MRI segmentation | each voxel's brightness | each voxel's tissue | each voxel among the tissues | each tissue's brightness curve |
| PET reconstruction | counts on each detector line | which pixel each count came from | each line's count among its pixels | each pixel's activity, which is the image |
| Doctors' calls | each doctor's yes or no | each patient's true diagnosis | each patient between "has it" and "does not" | the prevalence and each doctor's sensitivity and specificity |

In MRI the scanner forms the image and EM labels it. In PET, EM can form the image itself.

## Sources

- A. P. Dempster, N. M. Laird and D. B. Rubin, "Maximum Likelihood from Incomplete Data via the EM Algorithm," *Journal of the Royal Statistical Society, Series B* 39(1), 1–22, 1977. [doi:10.1111/j.2517-6161.1977.tb01600.x](https://doi.org/10.1111/j.2517-6161.1977.tb01600.x)
- P. E. Z. Larson, "Image Reconstruction," MRI education resources. [larsonlab.github.io](https://larsonlab.github.io/MRI-education-resources/Image%20Reconstruction.html)
- W. M. Wells III, W. E. L. Grimson, R. Kikinis and F. A. Jolesz, "Adaptive Segmentation of MRI Data," *IEEE Transactions on Medical Imaging* 15(4), 429–442, 1996. [doi:10.1109/42.511747](https://doi.org/10.1109/42.511747)
- "Let's Play PET" tutorial, image reconstruction page, hosted at web.stanford.edu. [Page](https://web.stanford.edu/dept/radiology/cgi-bin/classes/lpp/nuclearphysics/imagerecon.html)
- National Institute of Biomedical Imaging and Bioengineering (NIBIB), "Nuclear Medicine," science topics page. [nibib.nih.gov](https://www.nibib.nih.gov/science-education/science-topics/nuclear-medicine)
- L. A. Shepp and Y. Vardi, "Maximum Likelihood Reconstruction for Emission Tomography," *IEEE Transactions on Medical Imaging* 1(2), 113–122, 1982. [doi:10.1109/TMI.1982.4307558](https://doi.org/10.1109/TMI.1982.4307558)
- K. Lange and R. Carson, "EM Reconstruction Algorithms for Emission and Transmission Tomography," *Journal of Computer Assisted Tomography* 8(2), 306–316, 1984. PubMed ID 6608535, from the [Europe PMC record](https://europepmc.org/abstract/MED/6608535)
- H. M. Hudson and R. S. Larkin, "Accelerated Image Reconstruction Using Ordered Subsets of Projection Data," *IEEE Transactions on Medical Imaging* 13(4), 601–609, 1994. [doi:10.1109/42.363108](https://doi.org/10.1109/42.363108), [OSTI record](https://www.osti.gov/biblio/6610165)
- A. P. Dawid and A. M. Skene, "Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm," *Applied Statistics (Journal of the Royal Statistical Society, Series C)* 28(1), 20–28, 1979. [doi:10.2307/2346806](https://doi.org/10.2307/2346806), [RePEc record](https://ideas.repec.org/a/bla/jorssc/v28y1979i1p20-28.html)
