# Synthetic Application Generator

Generate synthetic applications by randomly combining 16 different types of function segments.

## Build

```
cd function_generator/
go build .
```

## Usage

```
./synthetic-function-generator generate -f ../function_segments --save --max-roll=1 --num-funcs=11
```

- `-f`: the path to the function segments (required)
- `--save`: save the generated files (names) to `generatedFunctions.txt` (can be used for playback)
- `--max-roll`: maximum number of function segments in each generated function (default 3)
- `--num-funcs`: total number of functions to generate (no repeated functions will be generated)

Generated functions are located at the `build/` folder.

## Create OpenWhisk Actions

To run the generated functions on OpenWhisk, we need to first prepare the package for registering the actions.

```
cd scripts
./create-action-zip-files.sh
```

Note that you need to have the NodeJS version consistent with the OpenWhisk lanaguage runtime version that you want to you. For example, if the function is going to be deployed in `nodejs:14` then you should also use NodeJS version 14 in your local host to generate the action package.

Then you can find the generated `function-name.zip` file under the folder for each function. To create an action:

```
wsk action create <function_name> --kind nodejs:14 <function_name>.zip
```

## MongoDB Requirement

By default, the functions will write the record of performance and system metrics during runtime to the MongoDB.
A local MongoDB server is needed to be set up to listen to `0.0.0.0:27017` so that the function can connect to it through Docker containers.

The MongoDB version that we tested is v6.0.2. Since we are using NodeJS to write to MongoDB, the NodeJS MongoDB Driver version is 4.11.0 and the NodeJS version we tested is v14.21.0.
