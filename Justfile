model := env_var_or_default("SAM3_MODEL", "facebook/sam3")
size := env_var_or_default("SAM3_SIZE", "672")
cuts := env_var_or_default("SAM3_CUTS", "8,16,24")
image := "sam3-dxnn"
out := "build"

default: build

build: export compile

export:
    uv run python export.py --model {{model}} --size {{size}} --cuts {{cuts}} --out {{out}}

compile: _image
    #!/usr/bin/env sh
    set -e
    for i in 0 1 2 3; do
        MSYS_NO_PATHCONV=1 podman run --rm --shm-size=16g \
            -v "$PWD:/work" -e PYTHONPATH=/work \
            -e SAM3_CALIB=/work/{{out}}/segment${i}_calib.npy {{image}} \
            dxcom -m /work/{{out}}/segment${i}_ready.onnx \
                  -c /work/{{out}}/segment${i}.json \
                  -o /work/{{out}}/segment${i} \
                  --opt_level 0 --aggressive_partitioning --quant_diagnosis
    done

bench picture="calib/dog.jpg": _image
    MSYS_NO_PATHCONV=1 podman run --rm --shm-size=8g -v "$PWD:/work" {{image}}         python /work/bench.py --build /work/{{out}} --image /work/{{picture}} --size {{size}}

_image:
    podman image exists {{image}} || podman build -t {{image}} -f Containerfile .

clean:
    rm -rf {{out}}
