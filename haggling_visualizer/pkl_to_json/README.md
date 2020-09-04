# pkl\_to\_json

## description
- A python script to losslessly change the format of haggling session body data at <https://github.com/CMU-Perceptual-Computing-Lab/ssp>.
- From: pickle (with byte string keys, numpy ndarrays, loosly coupled arrays), To: JSON (json string keys, json arrays, better constraints).

## roadmap
- [x] replace bytestring keys with json string keys.
- [x] convert numpy ndarrays to nested arrays.
- [x] assert lens of scores, joints19, body\_normal, face\_normal are equal.
- [x] improve structure constraints.

## code
- `pkl_to_json.py` contains code for format conversion.
- `format.md` contains the mapping from joint indices to body part.

## documentation
- The documentation for the code is itself.

## usage
- For conversion, use `python3 pkl_to_json.py <rel-path-to-pkl-file>`.
    - This produces a `<rel-path-to-pkl-file>.json`.
    - This also reads from the json file after writing and prints the structure fuzzily.
