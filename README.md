# Pose Conversion

Parameteric pose conversion between SMPL, SMPL-X, and FLAME.

Currently supports the extraction from the module [PIXIE](https://github.com/yfeng95/PIXIE), [DECA](https://github.com/yfeng95/DECA).

Currently supports the conversion for the module [SCARF](https://github.com/yfeng95/SCARF), [HOOD](https://github.com/dolorousrtur/hood).

## Preprocessing Raw Data From Existing Models

#### Example #1: From raw DECA(in Next3D) output to pickle dictionary.
```.bash
python preprocess.py --raw_path /your/path/to/deca_results/ \
	--save_path /your/path/to/save_results/
```

## Combining Different Modules into the Single Pose

#### Example #2: Combine DECA(FLAME) output with PIXIE(SMPL-X) output.
```.bash
python combine.py --smplx_path /your/path/to/smplx_results/ \
	--flame_path /your/path/to/flame/ \
	--save_path /your/path/to/save_results/
```

## Converting Pose Parameters

#### Example #1: From PIXIE(SMPL-X) output to HOOD(SMPL) input.
```.bash
python main.py --load_path ./examples/smplx.pkl \
	--save_path ./examples/ \
	--load_source smplx --module hood
```

#### Example #2: From PIXIE(SMPL-X) output to HOOD2(SMPL-X) input.
```.bash
python main.py --load_path ./examples/smplx.pkl \
	--save_path ./examples/ \
	--load_source smplx --module hood2
```

#### Example #3: From BLENDSHAPE output to Next3D input.
```.bash
python main.py --load_path ./examples/a2f_export_bsweight.json \
	--save_path ./examples/ \
	--load_source blendshape --module next3d
```


***

## Visualization

```.bash
python visualize.py --load_path ./examples/smplx.pkl \
	--model_folder /home/june1212/scarf/data/SMPLX_NEUTRAL_2020.npz
```
