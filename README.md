# Pose Conversion

Parameteric pose conversion between SMPL, SMPL-X, and FLAME.

Currently supports the extraction from the module [PIXIE](https://github.com/yfeng95/PIXIE), [DECA](https://github.com/yfeng95/DECA).

Currently supports the conversion for the module [SCARF](https://github.com/yfeng95/SCARF), [HOOD](https://github.com/dolorousrtur/hood), [Next3D](https://github.com/MrTornado24/Next3D), and [GART](https://github.com/JiahuiLei/GART).

## Preprocessing Raw Data From Existing Models

#### From raw DECA(in Next3D) output to pickle dictionary.
```.bash
python preprocess.py --module_name next3d \
--data_path /your/path/to/deca_results --save_path /your/path/to/save_results/file_name
```

#### From raw SMPL(in GART) output to pickle dictionary.
```.bash
python preprocess.py --module_name gart \
--data_path /your/path/to/smpl_npy --save_path /your/path/to/save_results/file_name
```

## Combining Different Modules into the Single Pose

#### Combine DECA(FLAME) output with PIXIE(SMPL-X) output.
```.bash
python combine.py --smplx_path /your/path/to/smplx_results \
	--flame_path /your/path/to/flame \
	--save_path /your/path/to/save_results
```

## Converting Pose Parameters

#### From PIXIE(SMPL-X) output to HOOD(SMPL) input.
```.bash
python main.py --load_path ./examples/smplx.pkl \
	--save_path ./examples \
	--load_source smplx --module hood
```

#### From PIXIE(SMPL-X) output to HOOD2(SMPL-X) input.
```.bash
python main.py --load_path ./examples/smplx.pkl \
	--save_path ./examples \
	--load_source smplx --module hood2
```

#### From BLENDSHAPE output to Next3D input.
Currently we don't have standard output shape for BLENDSHAPE. This repo utilizes the example JSON file in ./examples.
```.bash
python main.py --load_path ./examples/a2f_export_bsweight.json \
	--save_path ./examples \
	--load_source blendshape --module next3d
```

***

## Visualization

```.bash
python visualize.py --load_path ./examples/smplx.pkl \
	--model_folder [YOUR/PATH/TO/SMPLX_NEUTRAL_2020.npz]
```

You can visualize the naive sequence of meshes as below:

```.bash
python visualize.py --load_path [YOUR/MESH/PATH] \
	--model_type mesh
```

