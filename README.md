# Pose Conversion

Parameteric pose conversion between SMPL, SMPL-X, and FLAME.

Currently supports the extraction from the module [PIXIE](https://github.com/yfeng95/PIXIE), [DECA](https://github.com/yfeng95/DECA).
Currently supports the conversion for the module [SCARF](https://github.com/yfeng95/SCARF), [HOOD](https://github.com/dolorousrtur/hood).

### Converting Pose parameters

```.bash
python main.py --load_path ./examples/smplx.pkl --save_path ./examples/ --load_source smplx --module hood
```