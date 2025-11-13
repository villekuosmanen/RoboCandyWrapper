# 🍬 RoboCandyWrapper

🍬🍬🍬 Sweet wrappers for extending and remixing LeRobot Datasets 🍬🍬🍬

RoboCandyWrapper provides a system for backwards-compatible wrappers for extending and remixing LeRobotDatasets. The main features are:

1. Adapter system - enrich your LeRobotDatasets with additional labels and information without breaking backwards compatibility to the `main` branch of LeRobot and all other projects using it. Remix any combination of data adapters to maintain the single responsibility principle and clean code.
2. Freely combine any number of datasets at runtime without having to merge them first, for easy multi-dataset training. Sample evenly across the datasets or over/underweigh specific datasets in your data mix using the samplers provided in the project.
