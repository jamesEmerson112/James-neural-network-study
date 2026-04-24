# Nirbhay Modhe

## Overview

PhD in Computer Science from Georgia Tech (2017-2022), advised by Dhruv Batra in the Machine Learning & Perception (MLP) Lab. B.Tech in Computer Science from IIT Kanpur, where he worked with Piyush Rai on Bayesian ML. Research focus: model-based reinforcement learning, value-aware methods, unsupervised sub-goal discovery, and offline RL generalization.

## Current Status (as of April 2026)

Just left Zoox after ~2 years as a Software Engineer on the Prediction & Behavior ML team. At Zoox he led the 0-to-1 effort on language-aligned foundation models for autonomous driving behavior prediction. Before Zoox, he did a postdoc at Emory University's Center for Data Science (Nell Hodgson Woodruff School of Nursing) with Prof. Xiao Hu, working on ML for biomedical sequential decision-making.

His Google Scholar lists his affiliation as Zoox, Inc. with research interests in Machine Learning, Reinforcement Learning, and Information Theory.

## Education

| Degree | Institution | Years | Advisor |
|--------|-------------|-------|---------|
| PhD, Computer Science | Georgia Institute of Technology | 2017-2022 | Dhruv Batra |
| B.Tech, Computer Science | IIT Kanpur | ~2013-2017 | Piyush Rai (undergraduate research) |

PhD dissertation: *Leveraging Value-awareness for Online and Offline Model-based Reinforcement Learning*

## Publications

### Peer-Reviewed

1. **Scalable Generative Models for Multi-label Learning with Missing Labels**
   - Vikas Jain, **Nirbhay Modhe**, Piyush Rai
   - ICML 2017, pp. 1636-1644
   - IIT Kanpur work with Piyush Rai. Bayesian generative approach to multi-label classification that handles missing labels at scale.

2. **IR-VIC: Unsupervised Discovery of Sub-goals for Transfer in RL**
   - **Nirbhay Modhe**, Prithvijit Chattopadhyay, Mohit Sharma, Abhishek Das, Devi Parikh, Dhruv Batra, Ramakrishna Vedantam
   - IJCAI 2020, pp. 2022-2028
   - Proposes unsupervised discovery of decision states (sub-goals) that transfer across tasks in RL. Uses information-theoretic criteria to identify bottleneck states without reward supervision.

3. **Time-Aware Deep Sequential Models for In-Hospital Code Blue Prediction**
   - Ran Xiao, Matthew Clark, Cheng Ding, Duc Do, **Nirbhay Modhe**, Randall Lee, Timothy Ruchti, Xiao Hu
   - IEEE BHI 2023 (Extended Abstract)
   - Emory postdoc work. Applied sequential deep learning models with temporal awareness to predict in-hospital cardiac arrest events.

### Preprints / Workshop Papers

4. **Unsupervised Discovery of Decision States for Transfer in Reinforcement Learning**
   - **Nirbhay Modhe**, Prithvijit Chattopadhyay, Mohit Sharma, Abhishek Das, Devi Parikh, Dhruv Batra, Ramakrishna Vedantam
   - arXiv 2019
   - Earlier version of the IR-VIC / IJCAI 2020 paper.

5. **Model-Advantage Optimization for Model-Based Reinforcement Learning**
   - **Nirbhay Modhe**, Harish Kamath, Dhruv Batra, Ashwin Kalyan
   - arXiv 2021
   - Core thesis work. Introduces "model-advantage" -- a value-aware criterion for learning world models in model-based RL that focuses model capacity on states/transitions that matter for the policy, rather than reconstructing all dynamics equally.

6. **Exploiting Generalization in Offline Reinforcement Learning via Unseen State Augmentations**
   - **Nirbhay Modhe**, Qiaozi Gao, Ashwin Kalyan, Dhruv Batra, Govind Thattai, Gaurav Sukhatme
   - arXiv 2022/2023
   - Addresses generalization in offline RL by augmenting the training data with synthetically generated unseen states, improving policy robustness beyond the logged dataset.

## Specific RL Contributions

**Value-Aware Model Learning (thesis core):** Standard model-based RL learns a dynamics model by minimizing prediction error uniformly across all states. Modhe's work argues this is wasteful -- the model should focus its capacity on transitions that actually affect the value function and policy. The "model-advantage" formulation quantifies how much a model error in a given state-action pair actually hurts decision quality, and optimizes the model accordingly.

**Unsupervised Sub-goal Discovery (IR-VIC):** In transfer RL, agents need to identify reusable structure across tasks. IR-VIC discovers "decision states" (bottleneck states where the agent's future trajectory branches) without any reward signal, using mutual information criteria. These discovered sub-goals transfer to new tasks in the same environment, enabling faster learning.

**Offline RL Generalization:** Offline RL policies often fail on states not well-represented in the logged dataset. Modhe's augmentation approach generates plausible unseen states and uses them to regularize the policy, improving out-of-distribution robustness.

## Dhruv Batra's Lab & Research Group

**Lab:** Machine Learning & Perception Lab (MLP Lab)
- Website: https://mlp.cc.gatech.edu/
- GitHub: https://github.com/batra-mlp-lab
- Location: 2nd Floor, College of Computing, Georgia Tech

**Dhruv Batra** is a Professor in the School of Interactive Computing at Georgia Tech and was previously also at Meta AI (FAIR). His research spans computer vision, embodied AI, visual question answering, and robot navigation. The MLP Lab has produced graduates who went to Apple, Meta FAIR, Google DeepMind, and top academic positions.

**Notable MLP Lab alumni** (partial): Abhishek Das, Aishwarya Agrawal, Yash Goyal, Michael Cogswell, Stefan Lee, Peter Anderson.

The lab is affiliated with the broader Computer Vision Lab at Georgia Tech. Batra's group overlaps significantly with Devi Parikh's lab (Parikh is Batra's frequent collaborator and spouse) -- several of Modhe's papers include both Batra and Parikh as co-authors.

## Teaching at GT

Modhe delivered guest lectures and served as a TA at Georgia Tech in deep learning courses. There is a recorded YouTube lecture: *"L19 - Reinforcement Learning Part II"* from Georgia Tech's Deep Learning course, Fall 2020.

## Other

- Speedcuber: WCA record of 15.33 seconds on the 3x3 Rubik's Cube
- Interests: powerlifting, hiking
- Contact: nmodhe [AT] emory [DOT] edu (may be outdated post-Zoox)
- Personal site: https://nirbhayjm.github.io/
- GitHub: https://github.com/nirbhayjm
- Google Scholar: https://scholar.google.com/citations?user=g9MVi-EAAAAJ

## Sources

- https://nirbhayjm.github.io/
- https://scholar.google.com/citations?user=g9MVi-EAAAAJ
- https://dblp.org/pers/hd/m/Modhe:Nirbhay
- https://mlp.cc.gatech.edu/
- https://www.dhruvbatra.com/lab.html
- https://www.linkedin.com/in/nirbhaymodhe/
