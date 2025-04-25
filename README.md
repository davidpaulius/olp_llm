# olp_llm: Bootstrapping Object-Level Planning with Large Language Models

**NOTE:** This code is for the ICRA 2025 paper entitled ["Bootstrapping Object-Level Planning with Large Language Models"](https://davidpaulius.github.io/olp_llm) (Authors: David Paulius, Alejandro Agostini, Benedict Quartey, and George Konidaris).

## Preliminaries:

### I. Cloning Repository

You will need to clone this repository and recursively clone and download its submodules using the following commands:

```
git clone https://github.com/davidpaulius/olp_llm --recursive
git submodule update --remote --merge --recursive
```

There are several submodules that will be added to this directory:
1. [foon_to_pddl[(https://github.com/davidpaulius/foon_to_pddl)] -- this is needed to translate object-level plans (as FOONs) into PDDL as well as do other miscellaneous stuff with the generated graph.
2. [OMPLement](https://github.com/davidpaulius/OMPLement) -- this is a small library I wrote to simplify OMPL planning with CoppeliaSim
3. [fast-downward](https://github.com/aibasel/downward) -- this is the Fast-Downward automated planner.
  - You **must** build this solver prior to use. Follow the instructions [here](https://github.com/aibasel/downward/blob/main/BUILD.md) on how to do so.

### II. CoppeliaSim

You will need to download the [CoppeliaSim simulator](https://www.coppeliarobotics.com/) and set up its [Python API](https://manual.coppeliarobotics.com/en/remoteApiOverview.htm) using the following 	``pip`` command:

```
python3 -m pip install coppeliasim-zmqremoteapi-client
```

### III. Python Requirements

```
pip3 install -r requirements.txt
```

## How to Run

Simply run the Jupyter Notebook script named [``ICRA25_pipeline.py``](ICRA25_pipeline.py) once you have set everything up as above. You can change the task setting by modifying the code as well as set the OpenAI GPT model of your choice.


## Further Details on OMPL in CoppeliaSim

To deeply understand what's happening in the code, I would recommend going through the tutorials (_yes, I know..._) that are available [online](https://manual.coppeliarobotics.com/en/pathAndMotionPlanningModules.htm). There may be specific things you require for your own planning problem, so you should definitely consult the [forums](https://forum.coppeliarobotics.com/index.php).


### Questions? Comments? Feedback?

Feel free to reach out to me via [email](mailto:dpaulius@cs.brown.edu)!

