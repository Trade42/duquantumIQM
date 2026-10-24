## Getting set up on Resonance

Take a look at the tutorial in the [IQM Resonance](https://resonance.iqm.tech/) dashboard to learn how to set up your Python environment, make a Resonance account, connect to Resonance in your preferred framework, and manage jobs on the quantum computer.

## [IQM Resonance](https://resonance.iqm.tech/)

**[YOUR SIGNUP LINK FOR RESONANCE](https://resonance.iqm.tech/sign-up/duquantumiqmchallenge): https://resonance.iqm.tech/sign-up/duquantumiqmchallenge**

IQM Resonance is our quantum computing cloud platform that gives you access to IQM’s quantum hardware to design and run quantum programs. It supports multiple programming frameworks, offers thorough insights on our QPU specifications and your quantum jobs. Tools like the open-source [Qrisp](https://www.qrisp.eu) SDK for development and the pulse-level access package PulLA give you full control over the stack. 

**[Check out our client side libraries here - Dynamical Decoupling and many more!](https://resonance.iqm.tech/docs#quantum-development)**

**[Check out the IQM PulLA documentation here](https://docs.iqm.tech/iqm-pulla/)** - **To access PulLA you need to be a part of a paid organisation, please approach your mentors about this!**

## Libraries used in the pipeline

* [Qrisp on IQM user guide](https://docs.iqm.tech/iqm-client/user_guide_qrisp/index.html)
* [Quantum error correction with Qrisp on IQM (``DetectorExperiment``)](https://www.iqmacademy.com/notebookViewer/?path=%2Fnotebooks%2Fiqm%2Fradiance%2FCrystalQrispDetectorExperiment.ipynb)
* [Qiskit on IQM user guide](https://docs.iqm.tech/iqm-client/user_guide_qiskit.html)
* [Stim](https://github.com/quantumlib/Stim) - start with the [getting started notebook](https://github.com/quantumlib/Stim/blob/main/doc/getting_started.ipynb)
* [PyMatching API documentation](https://pymatching.readthedocs.io/en/stable/api.html)

## iqm_utils 

* ``QPURepetitionCode.ipynb``: A notebook example to run repetion codes *on IQM hardware (originally written for Garnet)*.
* *``Routing_to_specific_qubits.ipynb``*: In this notebook, you will run and compare a GHZ state as implemented naively and with respect to the qubit connectivity of the IQM quantum processor. Use this as baseline how to choose specific qubit layouts on hardware.
* ``get_calibration_data``: functionality to get extract calibration data from Resonance and feed it into your code. *Qiskit-based and set to Garnet by default: set the quantum computer and the server URL (``https://resonance.iqm.tech``) yourself.*
* ``blogposts/``: blogposts describing how surface code work in general

**Suggested reading order for iqm_utils QEC material:**

* ``QPURepetitionCode.ipynb`` notebook - Surface code defintion, Repetition code definition, understanding the full pipeline for repetition codes
* *``blog-post01-planar.ipynb``* - deeper understanding on how surface codes are set up, when stabilizers fire, 
* *``blog-post02-logical-error-rate.ipynb``*- how the decoding problem arises (recommended, but not vital for building the pipeline if you trust your tooling)

Or whatever resources you find online ;)

If you have any problems, IQM mentors are here to help! Visit our table or reach out to us on the hackathon Discord!


# The non-technical stuff - stay engaged!

We believe quantum hackathons are more than just competitions — they’re spaces to:
- learn from each other   
- and turn rough ideas into real-world impact   
- engange with tomorrow's real-world innovation

Whether you’re here to win, learn, network, or just ship something cool over a short burst of focused time — you’re in the right place.
Please get in touch with us on Discord!

--- 

## Giveaways 

We have T-shirts! Come talk to us at our booth!

Also, let's have a quick chat about what a Quantum computing company can do to integrate you into our work! ;) 

## Catch us on social media!

Please tag #IQM on [LinkedIn](https://www.linkedin.com/) and talk to us about your about your experience with IQM Resonance!

Join the [IQM Discord Channel](https://discord.gg/F38bwqPYJ4) and connect with other Resonance users and quantum enthusiasts!

---


## Important links


#### [IQM Academy](https://www.iqmacademy.com/)

IQM Academy is our learning and teaching platform, where you will find plenty of material to boost your quantum know-how. This includes amazing notebooks in different SDKs that you can adjust and run out of the box, tutorials on our QPUs and and resources on algorithm development. 

#### [IQM Discord Channel](https://discord.gg/F38bwqPYJ4)

The IQM Discord channel is our community platform, where we will inform you about all upcoming community events and enable you to reach out to us. We plan to run long-term challenges like bug-hunts and feature sprints with prizes!
You are the IQM community - connect with other quantum enthusiasts!